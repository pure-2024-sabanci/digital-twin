import heapq
from dataclasses import dataclass
from functools import total_ordering
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import uniform, bernoulli
import random
import functools
import time
import concurrent.futures
import threading
from typing import Optional, Tuple, List, Dict
from itertools import combinations
import plotly.graph_objects as go
from Utility import create_random_manhattan_graph, draw_graph,get_node_node_id,visualise_algorithm
import matplotlib
matplotlib.use('TkAgg')

@dataclass
@total_ordering
class Vehicle:
	"""
	A class to represent a vehicle moving on a graph.
	"""
	id: int
	last_seen_node: int
	traversed_nodes: List[Tuple[int, float]]
	current_edge: Optional[Tuple[int, int]]
	remaining_cost: float
	heuristic_map: Dict[Tuple[int, int], float]
	velocity: float = 1.0

	prev_route = List[int]

	def get_route(self):
		route = []
		for i in self.traversed_nodes:
			route.append(i[0])
		return route
	def __lt__(self, other):
		return self.remaining_cost < other.remaining_cost

	def __eq__(self, other):
		return self.remaining_cost == other.remaining_cost

	def __hash__(self):
		return hash(self.id)

def graph_same(heuristic_map,graph):
	for u, v in graph.edges:
		if(heuristic_map[(u, v)]!=graph[u][v]['weight']):
			return False
		if(heuristic_map[(v, u)]!=graph[v][u]['weight']):
			return False
	return True


def algorithm(graph, edge_binom_probs, n_agents, cost_multiplier, source, target, EXPLORATION_FAVOR):

	# Agent Initialization Phase
	agents=[]
	for i in range(n_agents):
		heuristic_map = {}

		for u, v in graph.edges:

			edge_prob = edge_binom_probs[(u, v)]#*(1-EXPLORATION_FAVOR) + uniform.rvs(0,1)*EXPLORATION_FAVOR

			if edge_prob > 1:
				edge_prob = 1
			if edge_prob < 0:
				edge_prob = 0

			state = bernoulli.rvs(edge_prob)
			if (state == 1):
				heuristic_map[(u, v)] = graph[u][v]['base'] * cost_multiplier
				heuristic_map[(v, u)] = graph[u][v]['base'] * cost_multiplier
			else:
				heuristic_map[(u, v)] = graph[u][v]['base']
				heuristic_map[(v, u)] = graph[u][v]['base']

		agents.append(Vehicle(i, source, [], None, 0, heuristic_map))

	"""	while graph_same(heuristic_map,graph):
			heuristic_map = {}

			for u, v in graph.edges:

				edge_prob=edge_binom_probs[(u, v)]*(1-EXPLORATION_FAVOR) + uniform.rvs(0,1)*EXPLORATION_FAVOR

				if edge_prob>1:
					edge_prob=1
				if edge_prob<0:
					edge_prob=0

				state=bernoulli.rvs(edge_prob)
				if(state==1):
					heuristic_map[(u, v)] = graph[u][v]['base']*cost_multiplier
					heuristic_map[(v, u)] = graph[u][v]['base']*cost_multiplier
				else:
					heuristic_map[(u, v)] = graph[u][v]['base']
					heuristic_map[(v, u)] = graph[u][v]['base']

			agents.append(Vehicle(i,source , [], None, 0, heuristic_map))
"""
	# Algorithm Phase

	# Initialize the priority queue with the agents

	explored_edges = set()
	priority_queue = []
	heapq.heapify(priority_queue)

	min_time=None
	Epoch=1

	total_time=0

	route_change=0

	finished_agent_ids = set()
	agent_ids = set([a.id for a in agents])

	while finished_agent_ids != agent_ids:

		if(min_time!=None):
			total_time += min_time
		for i in range(len(agents)):

			print(finished_agent_ids)

			if i in finished_agent_ids:
				print("Agent ",agents[i].id," has already reached the target")
				continue

			agent = agents[i]
			if(min_time!=None):

				agent.remaining_cost -= agent.velocity*min_time

			if agent.remaining_cost <= 0:
				if(agent.current_edge!=None):

					explored_edges.add(agent.current_edge)
					explored_edges.add((agent.current_edge[1],agent.current_edge[0]))
					agent.last_seen_node = agent.current_edge[1]

				else:
					agent.last_seen_node = source

				agent.traversed_nodes.append((agent.last_seen_node,total_time))

				if agent.last_seen_node == target:
					finished_agent_ids.add(agent.id)
					continue

				def custom_weight(u, v, data):
					if (u, v) in explored_edges or (v, u) in explored_edges:

						return graph[u][v]['weight']
					else:

						return agent.heuristic_map[(u, v)]


				#### This Function is Problematic !Implement Pathfinding Manually
				path = nx.dijkstra_path(graph, agent.last_seen_node, target, weight=custom_weight)

				if(agent.prev_route!=None):
					if(agent.prev_route!=path):
						route_change+=1

				print(path)
				agent.prev_route = path
				print("Agent ",agent.id," is going to ",path[1])

				agent.current_edge = (path[0], path[1])

				agent.remaining_cost = graph[path[0]][path[1]]['weight']


			time=agent.remaining_cost/agent.velocity
			heapq.heappush(priority_queue, (time, i))

		min_time, i = heapq.heappop(priority_queue)

	def sort_agents(agent):
		return agent.traversed_nodes[-1][1]

	agents.sort(key=sort_agents)

	return agents,route_change


if __name__ == "__main__":

	improvement=0
	improvement_perc=0

	decline=0

	# SYNTHETIC GRAPH PARAMETERS
	N_ROWS = 6
	N_COLS = 6
	NUMBER_OF_NODES = N_ROWS * N_COLS
	MIN_WEIGHT = 15
	MAX_WEIGHT = 100

	# ALGORITHM PARAMETERS
	EXPLORATION_FAVOR=0 # 0 ->Heuristics are same as the graph, 1->Heuristics are random
	SOURCE_NODE=0
	TARGET_NODE=NUMBER_OF_NODES-1
	NUMBER_OF_AGENTS=10
	COST_MULTIPLIER=200

	true_paths=[]
	true_path_lengths=[]
	heuristic_paths=[]
	heuristic_path_lengths=[]
	agent_paths=[]
	agent_path_lengths=[]


	"""
	Here we create a syntetic graph
	"""
	"""graph = nx.erdos_renyi_graph(NUMBER_OF_NODES, ERDOS_RENYI_PROB)
	while not nx.is_connected(graph):
		graph = nx.erdos_renyi_graph(NUMBER_OF_NODES, ERDOS_RENYI_PROB)"""

	graph=create_random_manhattan_graph(N_ROWS,N_COLS)

	edge_binom_probs = {(u, v): uniform.rvs(0,1) for u, v in graph.edges}


	for u, v in graph.edges:
		graph[u][v]['base'] = random.uniform(MIN_WEIGHT, MAX_WEIGHT)

		state = bernoulli.rvs(edge_binom_probs[(u, v)])
		if (state == 1):
			graph[u][v]['weight'] = graph[u][v]['base'] * COST_MULTIPLIER
			graph[v][u]['weight'] = graph[u][v]['base'] * COST_MULTIPLIER
		else:
			graph[u][v]['weight'] = graph[u][v]['base']
			graph[v][u]['weight'] = graph[u][v]['base']



	agents,route_change=algorithm(graph,edge_binom_probs,
	                             NUMBER_OF_AGENTS,
	                             COST_MULTIPLIER,
	                             SOURCE_NODE,TARGET_NODE,
	                             EXPLORATION_FAVOR)

	print("Route Change: ",route_change)


	# Visualize the graph and the agents
	visualise_algorithm(graph, agents)