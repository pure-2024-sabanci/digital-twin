#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bu betik, Algorithm.py'deki 'algorithm' fonksiyonunu ve
Utility.py'deki 'create_random_manhattan_graph' fonksiyonunu
kullanarak farklı parametreler altında test verisi üretir ve kaydeder.

Kullanım:
    python test_script.py

Çıktı:
    - test_results.json dosyası
"""

import random
import json

import networkx as nx

from Algorithm import algorithm
from Utility import create_random_manhattan_graph, visualise_algorithm
from scipy.stats import bernoulli, uniform

# TQDM kütüphanesini ekliyoruz
from tqdm.auto import tqdm
def run_experiments():
    # Deney sonuçlarını tutacağımız sözlük
    results = {}

    # Parametre aralıklarını tanımlayalım:
    # Farklı grid boyutları
    row_options = [4, 5, 6]
    col_options = [4, 5, 6]

    # Kaç ajan kullanacağımız
    agent_options = [5, 10]

    # Farklı cost multiplier değerleri
    cost_multipliers = [50, 100, 200, 500]

    # Kenar ağırlıkları
    min_weight = 10
    max_weight = 50

    # Kaç tekrar yapacağımız (farklı rastgele tohumlar)
    num_repeats = 5

    # Tüm parametre kombinasyonlarını gözlemlemek için iç içe döngüler:
    # Her döngüyü tqdm ile sarıyoruz.
    # Toplam kombinasyon sayısı
    total_steps = (
        len(row_options)
        * len(col_options)
        * len(agent_options)
        * len(cost_multipliers)
        * num_repeats
    )

    # Tek bir ilerleme çubuğu oluştur
    pbar = tqdm(total=total_steps, desc="Tüm Deneyler", leave=True)

    # İç içe döngüler:
    for n_rows in row_options:
        for n_cols in col_options:
            for n_agents in agent_options:
                for cmult in cost_multipliers:
                    for rep in range(num_repeats):

                        desc_text = (
                            f"Rows={n_rows}, "
                            f"Cols={n_cols}, "
                            f"Agents={n_agents}, "
                            f"Cost={cmult}, "
                            f"Repeat={rep + 1}/{num_repeats}"
                        )
                        pbar.set_description(desc_text)

                        # Rastgele tohum
                        seed_value = 1000 * rep + n_rows + n_cols + n_agents
                        random.seed(seed_value)

                        # Rastgele Manhattan graph oluştur
                        graph = create_random_manhattan_graph(n_rows, n_cols)

                        # Kenar ağırlıklarını (base) rastgele belirleyelim
                        edge_binom_probs = {}
                        for u, v in graph.edges():
                            prob = random.uniform(0, 1)
                            edge_binom_probs[(u, v)] = prob
                            edge_binom_probs[(v, u)] = prob
                            base_val = random.uniform(min_weight, max_weight)
                            graph[u][v]['base'] = base_val
                            graph[v][u]['base'] = base_val

                        # Kenarların fiili weight değerlerini ayarlayalım:
                        for u, v in graph.edges():
                            edge_prob = edge_binom_probs[(u, v)]
                            state = bernoulli.rvs(edge_prob)

                            if state == 1:
                                graph[u][v]['weight'] = graph[u][v]['base'] * cmult
                                graph[v][u]['weight'] = graph[v][u]['base'] * cmult
                            else:
                                graph[u][v]['weight'] = graph[u][v]['base']
                                graph[v][u]['weight'] = graph[v][u]['base']

                        # Algorithm.py'deki 'algorithm' fonksiyonunu çağıralım
                        source_node = 0
                        target_node = n_rows * n_cols - 1
                        cost_multiplier = cmult
                        EXPLORATION_FAVOR = 0

                        agents, route_change = algorithm(
                            graph,
                            edge_binom_probs,
                            n_agents,
                            cost_multiplier,
                            source_node,
                            target_node,
                            EXPLORATION_FAVOR
                        )

                        # Sonuçları kaydedelim
                        result_dict = {
                            "graph": {
                                "seed_value": seed_value,
                                "n_rows": n_rows,
                                "n_cols": n_cols,
                                "nodes" : [{
                                    "id": node,
                                    "pos": pos
                                } for node, pos in graph.nodes(data='pos')],
                                "edges": [
                                    {
                                        "source": u,
                                        "target": v,
                                        "base_cost": graph[u][v]['base'],
                                        "actual_cost": graph[u][v]['weight']
                                    } for u, v in graph.edges()
                                ],
                            },
                            "algorithm": {
                                "num_agents": n_agents,
                                "cost_multiplier": cost_multiplier,
                                "rep_index": rep,
                                "seed_value": seed_value,
                                "n_route_change": route_change,
                                "agents": [
                                    {
                                        "agent_id": ag.id,
                                        "route": ag.traversed_nodes,
                                        "heuristic_map": {str(k): v for k, v in ag.heuristic_map.items()},
                                        "velocity": ag.velocity
                                    } for ag in agents
                                ],
                            }
                        }

                        key_name = f"{n_rows}x{n_cols}_agents{n_agents}_cmult{cmult}_rep{rep}"
                        results[key_name] = result_dict
                        pbar.update(1)
                        
                        visualise_algorithm(graph, agents)

    # Bütün sonuçları JSON olarak yazalım
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Tüm deneyler tamamlandı. test_results.json dosyası oluşturuldu.")

if __name__ == "__main__":
    run_experiments()
