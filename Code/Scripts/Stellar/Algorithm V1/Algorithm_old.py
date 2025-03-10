from dataclasses import dataclass
from functools import total_ordering
import networkx as nx
from scipy.stats import beta, uniform, bernoulli
import random
import functools
import time
import concurrent.futures
import threading
from typing import Optional, Tuple, List, Dict
from itertools import combinations

# A small tolerance value (to decide when an agent has finished its current edge)
EPSILON = 1e-6


def timer(func):
	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		tic = time.perf_counter_ns()
		value = func(*args, **kwargs)
		toc = time.perf_counter_ns()
		elapsed_time = (toc - tic) / 1e9  # convert nanoseconds to seconds
		print(f"Elapsed time: {elapsed_time:0.4f} seconds")
		return value

	return wrapper_timer


@dataclass
@total_ordering
class Vehicle:
	"""
	A class to represent a vehicle moving on a graph.
	"""
	id: int
	last_seen_node: int
	traversed_nodes: List[int]
	current_edge: Optional[Tuple[int, int]]
	remaining_cost: float
	heuristic_map: Dict[Tuple[int, int], float]
	velocity: float = 1.0

	def __lt__(self, other):
		return self.remaining_cost < other.remaining_cost

	def __eq__(self, other):
		return self.remaining_cost == other.remaining_cost

	def __hash__(self):
		return hash(self.id)


def create_cost_map(graph: nx.Graph) -> Dict[Tuple[int, int], Tuple[float, float]]:
	"""
	Creates a cost map for the graph.
	Each edge is assigned a tuple: (original cost, adjusted cost).
	Adjusted cost is computed using a random factor from a uniform distribution.
	"""
	cost_map = {}
	for edge in graph.edges:
		orig = graph.edges[edge]['weight']
		# Avoid division-by-zero by drawing from U(EPSILON, 1)
		adjusted = orig * (1 / uniform.rvs(EPSILON, 1))
		cost_map[edge] = (orig, adjusted)
	# For undirected graphs, mirror the cost info for the reversed edge.
	for edge in list(graph.edges):
		edge_reversed = (edge[1], edge[0])
		cost_map[edge_reversed] = cost_map[edge]
	return cost_map


def create_heuristic_map(cost_map: Dict[Tuple[int, int], Tuple[float, float]], theta: float) -> Dict[
	Tuple[int, int], float]:
	"""
	Creates a heuristic map by combining the original and adjusted costs.
	"""
	heuristic_map = {}
	for edge, (orig, adjusted) in cost_map.items():
		heuristic_map[edge] = adjusted * theta + orig * (1 - theta)
	return heuristic_map


@timer
def algorithm(graph, cost_map, realised_edges, start, target, n_agents):
	"""
	Runs the simulation using a discrete‐event (time‐advance) approach.

	At each iteration the simulation does:

	  1. Compute the next time step (dt) as the minimum remaining time among all agents.
	  2. For every agent, decrement its remaining cost by (dt * velocity).
	  3. For each agent that finishes its current edge (i.e. remaining_cost <= EPSILON),
		 update its state (mark the edge as explored, update its location, and
		 choose the next edge via a shortest path search using a custom weight function).

	Agent updates in steps (2) and (3) are run in parallel.
	"""
	agents = []
	theta = 0.0
	for i in range(n_agents):
		# Each agent gets its own heuristic map based on a different weight (theta).
		heuristic = create_heuristic_map(cost_map, theta)
		# Initialize: start at the 'start' node.
		agent = Vehicle(i, start, [start], None, 0.0, heuristic)
		agents.append(agent)
		theta += 1 / n_agents

	# Shared data structure to hold explored edges.
	explored_edges = set()
	explored_edges_lock = threading.Lock()

	# --- INITIALIZATION: assign each agent its first edge ---
	for agent in agents:
		# Use a simple custom weight that only consults the agent’s heuristic
		def custom_weight(u, v, d):
			return agent.heuristic_map.get((u, v), agent.heuristic_map.get((v, u), 1))

		path = nx.shortest_path(graph, start, target, weight=custom_weight)
		if len(path) < 2:
			continue  # should not occur if start != target
		next_edge = (path[0], path[1])
		agent.current_edge = next_edge
		agent.remaining_cost = cost_map[next_edge][realised_edges[next_edge]]

	# Function to update the state of an agent that has finished traversing its current edge.
	def update_agent_state(agent: Vehicle) -> bool:
		"""
		Update the agent if it has finished its edge.
		Returns True if the agent has reached the target.
		"""
		if agent.current_edge is not None:
			with explored_edges_lock:
				explored_edges.add(agent.current_edge)
				explored_edges.add((agent.current_edge[1], agent.current_edge[0]))
			agent.last_seen_node = agent.current_edge[1]
		else:
			agent.last_seen_node = start

		agent.traversed_nodes.append(agent.last_seen_node)

		# Check if the target has been reached.
		if agent.last_seen_node == target:
			return True

		# Define a custom weight function that uses both the agent’s heuristic
		# and the explored_edges information.
		def custom_weight(u, v, d):
			with explored_edges_lock:
				if (u, v) in explored_edges:
					return cost_map[(u, v)][realised_edges[(u, v)]]
			return agent.heuristic_map.get((u, v), agent.heuristic_map.get((v, u), 1))

		path = nx.shortest_path(graph, agent.last_seen_node, target, weight=custom_weight)
		if len(path) < 2:
			return True  # Already at target (should not normally happen)
		next_edge = (path[0], path[1])
		agent.current_edge = next_edge
		agent.remaining_cost = cost_map[next_edge][realised_edges[next_edge]]
		return False

	# Use a ThreadPoolExecutor to update agents in parallel.
	with concurrent.futures.ThreadPoolExecutor(max_workers=n_agents) as executor:
		while True:
			# --- PHASE 1: Determine the next time step ---
			# For each agent, the time to finish its current edge is (remaining_cost / velocity).
			# (Agents that are in the middle of an edge will have positive times.)
			dt_candidates = [agent.remaining_cost / agent.velocity for agent in agents]
			dt = min(dt_candidates)

			# --- PHASE 2: Advance time for all agents in parallel ---
			def advance_agent(agent: Vehicle, delta: float) -> None:
				agent.remaining_cost -= delta * agent.velocity

			list(executor.map(lambda a: advance_agent(a, dt), agents))

			# --- PHASE 3: For agents that have finished their edge, update their state in parallel ---
			# An agent is considered to have finished if its remaining_cost is (almost) zero.
			finished_flags = list(
				executor.map(lambda a: update_agent_state(a) if a.remaining_cost <= EPSILON else False, agents))
			# If any agent has reached the target, return it.
			for agent, reached in zip(agents, finished_flags):
				if reached:
					return agent


# Helper function to create a connected directed graph, along with cost maps.
def create_graph_and_realise(n_nodes, p_edge, min_weight, max_weight):
	graph = nx.erdos_renyi_graph(n_nodes, p_edge)
	while not nx.is_connected(graph):
		graph = nx.erdos_renyi_graph(n_nodes, p_edge)

	# Assign random integer weights to each edge.
	for edge in graph.edges:
		graph.edges[edge]['weight'] = random.randint(min_weight, max_weight)

	cost_map = create_cost_map(graph)

	# For each edge, realise it with a random Bernoulli outcome.
	realised_edges = {}
	for edge in graph.edges:
		realised_edges[edge] = bernoulli.rvs(uniform.rvs(0, 1))
	for edge in list(graph.edges):
		edge_reversed = (edge[1], edge[0])
		realised_edges[edge_reversed] = realised_edges[edge]

	# Convert to a directed graph.
	directed_graph = graph.to_directed()
	return directed_graph, cost_map, realised_edges


if __name__ == "__main__":
	# Create a graph with 100 nodes, edge probability 0.1, and weights between 1 and 10.
	n=100
	graph, cost_map, realised_edges = create_graph_and_realise(n, 0.1, 1, 10)

	def custom_weight(u, v, d):
		return cost_map[(u, v)][realised_edges[(u, v)]]

	combinations = list(combinations(range(n), 2))

	results=[]
	for comb in combinations:
		start = comb[0]
		target = comb[1]

		print(f"Start: {start}, Target: {target}")
		path=nx.shortest_path(graph, start, target, weight=custom_weight)


		result_agent = algorithm(graph, cost_map, realised_edges, start, target, n_agents=10)
		results.append(result_agent.traversed_nodes==path)

	print(f"Success rate: {sum(results)/len(results)}")