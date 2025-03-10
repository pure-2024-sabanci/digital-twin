"""
This Script will be used for testing Stellar Algorithm

METRICS:
    First Found Path length
        Comparison with Dijkstra's Algorithm (Blind)
        Comparison with Dijkstra's Algorithm (Full Knowledge)
    Total Length Traversed
    Average Length Traversed Per Agent

INDEPENDENT VARIABLES:
    Number of Agents
    Heuristics Used (?)
    Graph Size
    Edge Probability
"""
import concurrent.futures

from Algorithm import Stellar, create_graph_and_realise
import networkx as nx
import pandas as pd
from tqdm import tqdm
from itertools import combinations

MIN_WEIGHT = 10
MAX_WEIGHT = 100

EPSILON = 1e-6

def test_stellar(n_nodes, p_edge, n_agents):
    graph, cost_map, realised_edges, bernouilli_probs = create_graph_and_realise(n_nodes, p_edge, MIN_WEIGHT, MAX_WEIGHT)

    # Benchmarking Dijkstra's Algorithm
    def dijkstra_full(source, target):
        def custom_cost(u, v, d):
            return cost_map[(u, v)][realised_edges[(u, v)]]

        path = nx.shortest_path(graph, source, target, weight=custom_cost, method='dijkstra')
        path_length = sum(custom_cost(path[i], path[i + 1], 0) for i in range(len(path) - 1))
        return len(path), path_length

    def dijkstra_blind_expected(source, target):
        def custom_cost(u, v, d):
            return cost_map[(u, v)][1] * bernouilli_probs[(u, v)] + cost_map[(u, v)][0] * (1 - bernouilli_probs[(u, v)])

        path = nx.shortest_path(graph, source, target, weight=custom_cost, method='dijkstra')
        expected_length = sum(custom_cost(path[i], path[i + 1], 0) for i in range(len(path) - 1))
        return expected_length



    combs = combinations(range(n_nodes), 2)
    """
    results format
    
    {
    n_nodes
    p_edge
    
    }
    
    
    """

if __name__ == "__main__":
    # Define the parameters for the experiments.
    n_nodes = [10, 20]
    p_edges = [0.05, 0.1, 0.2]
    n_agents = list(range(1, 16))


    def tasks_generator():
        for n in n_nodes:
            for p in p_edges:
                for a in n_agents:
                    for source, target in combinations(range(n), 2):
                        yield (n, p, a, source, target)


    # Calculate the total number of tasks for tqdm progress bar
    total_tasks = sum((n * (n - 1) // 2) * len(p_edges) * len(n_agents) for n in n_nodes)

    results = []
    # Adjust max_workers as appropriate for your system.
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # executor.map returns results in order; wrap it with tqdm for progress
        for res in tqdm(executor.map(lambda params: test_stellar(*params), tasks_generator()),
                        total=total_tasks, desc="Running Tests"):
            results.append(res)

    # Save results to an Excel file
    df = pd.DataFrame(results)
    df.to_excel("stellar_results.xlsx", index=False)