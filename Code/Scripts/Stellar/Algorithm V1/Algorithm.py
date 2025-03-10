from dataclasses import dataclass, field
from functools import total_ordering
import networkx as nx
from scipy.stats import beta, uniform, bernoulli
import random
import functools
import time
import concurrent.futures
import threading
from typing import Optional, Tuple, List, Dict, Any, Set
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import cProfile
import heapq
from tqdm import tqdm

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

@total_ordering
@dataclass
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
    # Fields for caching the incremental Dijkstra information:
    cached_dists: Optional[Dict[Any, float]] = None
    cached_preds: Optional[Dict[Any, Any]] = None
    cached_explored: Set[Tuple[Any, Any]] = field(default_factory=set)

    def __lt__(self, other):
        return self.remaining_cost < other.remaining_cost

    def __eq__(self, other):
        return self.remaining_cost == other.remaining_cost

    def __hash__(self):
        return hash(self.id)

# ---------------------------------------
# Helper functions for Dijkstra updates
# ---------------------------------------

def full_dijkstra(graph: nx.Graph, start: Any, cost_func) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """
    Compute the full shortest–path tree from start using the given cost function.
    """
    dist = {start: 0}
    preds = {}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in graph.neighbors(u):
            alt = dist[u] + cost_func(u, v)
            if alt < dist.get(v, float('inf')):
                dist[v] = alt
                preds[v] = u
                heapq.heappush(pq, (alt, v))
    return dist, preds

def incremental_dijkstra(
    graph: nx.Graph,
    start: Any,
    cost_func,
    old_dist: Dict[Any, float],
    old_pred: Dict[Any, Any],
    # Set of edges (u, v) whose cost has changed
    changed_edges: Set[Tuple[Any, Any]]
) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """
    Update a previously computed shortest–path tree (old_dist and old_pred)
    from start when some edges have changed weight.
    """
    # Make a copy of the old distances and predecessors.
    new_dist = dict(old_dist)
    new_pred = dict(old_pred)
    pq = []

    # For each changed edge, try to relax the affected neighbor.
    for u, v in changed_edges:
        if u not in new_dist or new_dist[u] == float('inf'):
            continue
        alt = new_dist[u] + cost_func(u, v)
        if alt < new_dist.get(v, float('inf')):
            new_dist[v] = alt
            new_pred[v] = u
            heapq.heappush(pq, (alt, v))

    # Propagate the changes.
    while pq:
        d, u = heapq.heappop(pq)
        if d > new_dist[u]:
            continue
        for v in graph.neighbors(u):
            alt = new_dist[u] + cost_func(u, v)
            if alt < new_dist.get(v, float('inf')):
                new_dist[v] = alt
                new_pred[v] = u
                heapq.heappush(pq, (alt, v))
    return new_dist, new_pred

def extract_path(preds: Dict[Any, Any], start: Any, target: Any) -> Optional[List[Any]]:
    """
    Given a predecessor dictionary, extract the path from start to target.
    Returns None if no path is found.
    """
    path = []
    current = target
    while current != start:
        path.append(current)
        if current not in preds:
            return None
        current = preds[current]
    path.append(start)
    path.reverse()
    return path

# ---------------------------------------
# Cost and heuristic map functions
# ---------------------------------------

def create_cost_map(graph: nx.Graph) \
        -> Dict[Tuple[int, int], Tuple[float, float]]:
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

def create_heuristic_map(cost_map: Dict[Tuple[int, int], Tuple[float, float]], theta: float) -> Dict[Tuple[int, int], float]:
    """
    Creates a heuristic map by combining the original and adjusted costs.
    """
    heuristic_map = {}
    for edge, (orig, adjusted) in cost_map.items():
        heuristic_map[edge] = adjusted * theta + orig * (1 - theta)
    return heuristic_map

# ---------------------------------------
# The simulation algorithm with incremental updates
# ---------------------------------------

def Stellar(graph, cost_map, realised_edges, start, target, n_agents):
    """
    Runs the simulation using a discrete‐event (time‐advance) approach.

    Instead of recalculating the full shortest–path at every update,
    each agent caches its current shortest–path tree and uses an incremental
    update when some edges become explored.
    """
    agents = []
    theta = 0.0
    cost_taken = 0
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
        # Initially, no edges are explored.
        agent.cached_explored = set()
        def cost_func(u, v):
            # Initially, use only the heuristic.
            return agent.heuristic_map.get((u, v), agent.heuristic_map.get((v, u), 1))
        # Compute full Dijkstra from the start node.
        agent.cached_dists, agent.cached_preds = full_dijkstra(graph, start, cost_func)
        path = extract_path(agent.cached_preds, start, target)
        if path is None or len(path) < 2:
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
        # Mark the current edge as explored.
        if agent.current_edge is not None:
            with explored_edges_lock:
                explored_edges.add(agent.current_edge)
                # For undirected behavior, also mark the reverse.
                explored_edges.add((agent.current_edge[1], agent.current_edge[0]))
            agent.last_seen_node = agent.current_edge[1]
        else:
            agent.last_seen_node = start

        agent.traversed_nodes.append(agent.last_seen_node)

        # Check if the target has been reached.
        if agent.last_seen_node == target:
            return True

        # --- Incremental Dijkstra update ---
        with explored_edges_lock:
            new_explored = set(explored_edges)
        # Determine which edges have newly become explored.
        changed_edges = new_explored - agent.cached_explored
        agent.cached_explored = new_explored

        # Define the cost function that uses both explored and unexplored edge weights.
        def cost_func(u, v):
            if (u, v) in new_explored:
                return cost_map[(u, v)][realised_edges[(u, v)]]
            return agent.heuristic_map.get((u, v), agent.heuristic_map.get((v, u), 1))

        # If the cached tree is missing or outdated for the new source, compute from scratch.
        if agent.cached_dists is None or agent.last_seen_node not in agent.cached_dists:
            agent.cached_dists, agent.cached_preds = full_dijkstra(graph, agent.last_seen_node, cost_func)
        else:
            agent.cached_dists, agent.cached_preds = incremental_dijkstra(
                graph, agent.last_seen_node, cost_func, agent.cached_dists, agent.cached_preds, changed_edges
            )

        # Extract the path from agent.last_seen_node to target.
        path = extract_path(agent.cached_preds, agent.last_seen_node, target)
        if path is None or len(path) < 2:
            return True  # no path found, assume target reached or error
        next_edge = (path[0], path[1])
        agent.current_edge = next_edge
        agent.remaining_cost = cost_map[next_edge][realised_edges[next_edge]]
        return False

    # Use a ThreadPoolExecutor to update agents in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_agents) as executor:
        while True:
            # --- PHASE 1: Determine the next time step ---
            # For each agent, the time to finish its current edge is (remaining_cost / velocity).
            dt_candidates = [agent.remaining_cost / agent.velocity for agent in agents]
            dt = min(dt_candidates)

            # --- PHASE 2: Advance time for all agents in parallel ---
            def advance_agent(agent: Vehicle, delta: float) -> None:
                agent.remaining_cost -= delta * agent.velocity
            list(executor.map(lambda a: advance_agent(a, dt), agents))

            # --- PHASE 3: For agents that have finished their edge, update their state in parallel ---
            finished_flags = list(
                executor.map(lambda a: update_agent_state(a) if a.remaining_cost <= EPSILON else False, agents)
            )
            # If any agent has reached the target, return it.
            for agent, reached in zip(agents, finished_flags):
                if reached:
                    total_cost_taken=0

                    for a in agents:
                        total_cost_taken+=a.remaining_cost
                        path= a.traversed_nodes
                        for i in range(len(path)-1):
                            total_cost_taken+=cost_map[(path[i], path[i+1])][realised_edges[(path[i], path[i+1])]]

                    cost_taken = agent.remaining_cost
                    path= agent.traversed_nodes
                    for i in range(len(path)-1):
                        cost_taken+=cost_map[(path[i], path[i+1])][realised_edges[(path[i], path[i+1])]]

                    return agent, cost_taken, total_cost_taken

# ---------------------------------------
# Helper function to create a graph and realise edge outcomes
# ---------------------------------------

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
    edge_bernouilli_probs={}
    for edge in graph.edges:
        prob=uniform.rvs(0, 1)
        edge_bernouilli_probs[edge] = prob
        realised_edges[edge] = bernoulli.rvs(prob)
    for edge in list(graph.edges):
        edge_reversed = (edge[1], edge[0])
        edge_bernouilli_probs[edge_reversed] = edge_bernouilli_probs[edge]
        realised_edges[edge_reversed] = realised_edges[edge]

    # Convert to a directed graph.
    directed_graph = graph.to_directed()
    return directed_graph, cost_map, realised_edges, edge_bernouilli_probs

# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":
    n = 50
    graph, cost_map, realised_edges = create_graph_and_realise(n, 0.1, 1, 10)

    combinations = list(combinations(range(n), 2))

    costs = []
    total_costs = []

    for n_agents in tqdm(range(1, 19)):
        n_costs = []
        n_total_costs = []
        for source,target in combinations:
            agent, cost,total_cost = algorithm(graph, cost_map, realised_edges, source, target, n_agents)
            n_costs.append(cost)
            n_total_costs.append(total_cost)
        costs.append(n_costs)
        total_costs.append(n_total_costs)


    agents = [i for i in range(1, 19)]
    average_time = [sum(cost) / len(cost) for cost in costs]
    average_total_cost = [sum(cost) / len(cost) for cost in total_costs]

    sns.scatterplot(x=agents, y=average_time)

    plt.title('Average Time vs. Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Time (s)')
    plt.grid(True)
    plt.show()

    sns.scatterplot(x=agents, y=average_total_cost)
    plt.title('Average Total Cost vs. Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Total Cost')
    plt.grid(True)
    plt.show()

    print(average_total_cost)
