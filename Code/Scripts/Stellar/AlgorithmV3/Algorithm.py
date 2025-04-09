import heapq
from dataclasses import dataclass
from functools import total_ordering
import networkx as nx
import random
from enum import Enum
from typing import Optional, Tuple, List, Dict

from matplotlib import pyplot as plt
from scipy.stats import uniform, bernoulli

import random
from Utility import create_random_manhattan_graph, visualise_algorithm

class AgentPolicy(Enum):
    """
    Farklı ajan politika türleri.
    """
    GRAPH_SAMPLING = 1
    CONSTANT = 2
    RANDOM = 3
    SHARED = 4


@dataclass
@total_ordering
class Vehicle:
    """
    Represents a single vehicle (agent).
    """
    id: int
    last_seen_node: int
    traversed_nodes: List[Tuple[int, float]]
    current_edge: Optional[Tuple[int, int]]
    remaining_cost: float
    heuristic_map: Dict[Tuple[int,int], float]
    velocity: float = 1.0
    prev_route = List[int]

    def get_route(self) -> List[int]:
        return [node for (node, _) in self.traversed_nodes]

    def __lt__(self, other):
        return self.remaining_cost < other.remaining_cost

    def __eq__(self, other):
        return self.remaining_cost == other.remaining_cost


def check_policy_args(
    policy: AgentPolicy,
    policy_args: dict,
    n_agents: int
):
    """
    Her bir AgentPolicy türü için gerekli parametrelerin policy_args içinde
    olup olmadığını kontrol eder. Eğer eksikse ValueError fırlatılır.
    """
    if policy == AgentPolicy.GRAPH_SAMPLING:
        # edge_bernoulli_probs yoksa hata ver
        if "edge_bernoulli_probs" not in policy_args:
            raise ValueError(
                "GRAPH_SAMPLING policy requires 'edge_bernoulli_probs' in policy_args."
            )

    elif policy == AgentPolicy.CONSTANT:
        # const_p yoksa hata
        if "const_p" not in policy_args:
            raise ValueError(
                "CONSTANT policy requires 'const_p' in policy_args."
            )
        # Ayrıca 0..1 arası mı diye kontrol edebilirsiniz
        const_p = policy_args["const_p"]
        if not (0 <= const_p <= 1):
            raise ValueError(f"const_p must be in [0,1], got {const_p}.")

    elif policy == AgentPolicy.RANDOM:
        # RANDOM için özel arg gerekmiyor, isterseniz pass diyebilirsiniz
        pass

    elif policy == AgentPolicy.SHARED:
        # SHARED'ta da ek arg gerekmez, ama n_agents > 1 olmalı
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1 for SHARED policy.")
        # Tek ajan varsa da bir uyarı basılabilir, ama durdurmasak da olur
        if n_agents == 1:
            print("Warning: SHARED policy with only 1 agent doesn't make sense.")
    else:
        # Tanımsız policy
        raise ValueError(f"Unsupported AgentPolicy: {policy}")


def get_agent_edge_probability(
    agent_id: int,
    edge: Tuple[int,int],
    policy: AgentPolicy,
    n_agents: int,
    policy_args: dict
) -> float:
    """
    Verilen policy'ye göre bu ajanın kenar yıkılma olasılığını (0..1) belirler.
    check_policy_args fonksiyonu ile zaten argümanlar kontrol ediliyor.
    """
    if policy == AgentPolicy.GRAPH_SAMPLING:
        # Bu politikada 'edge_bernoulli_probs' sözlüğünü kullanıyoruz
        edge_probs = policy_args["edge_bernoulli_probs"]
        # Kenar (u,v) yoksa default 0.5 alalım
        return edge_probs.get(edge, 0.5)

    elif policy == AgentPolicy.CONSTANT:
        # Sabit bir olasılık
        const_p = policy_args["const_p"]
        return const_p

    elif policy == AgentPolicy.RANDOM:
        # Her edge için random
        return random.uniform(0, 1)

    elif policy == AgentPolicy.SHARED:
        # n_agents'a göre paylaştırma
        # agent_id = 0 -> prob=0
        # agent_id = n_agents-1 -> prob=1
        # Diğerleri eşit adımlarla
        if n_agents > 1:
            prob = agent_id / (n_agents - 1)
            return prob
        else:
            # Tek ajan varsa 0.5 diyebilirsiniz ya da 0.0
            return 0.5

    # Varsayılan
    return 0.5


def algorithm(
    graph,
    n_agents: int,
    cost_function,
    source: int,
    target: int,
    policy: AgentPolicy,
    policy_args: dict = None,
    share: bool = True
):
    """
    Çoklu ajan algoritması.
    Ajanların edge probability'si, 'policy' ve 'policy_args' doğrultusunda belirlenir.

    Parametreler:
    - graph: NetworkX graph
    - n_agents: Kaç ajan kullanılacak
    - cost_function: Kenar hasar çarpanı için fonksiyon (örn. lambda: 1/uniform.rvs(...))
    - source, target: Başlangıç ve hedef node
    - policy: AgentPolicy Enum değeri (GRAPH_SAMPLING, CONSTANT, RANDOM, SHARED)
    - policy_args:
        * GRAPH_SAMPLING -> {"edge_bernoulli_probs": dict}
        * CONSTANT -> {"const_p": 0.5}
        * RANDOM -> {}
        * SHARED -> {} (n_agents>1 önerilir)
    """
    if policy_args is None:
        policy_args = {}

    # Önce kontrol
    check_policy_args(policy, policy_args, n_agents)

    agents = []
    # Her ajan için heuristic_map oluştur
    for i in range(n_agents):
        heuristic_map = {}
        for (u, v) in graph.edges():
            # Politika'ya göre bu ajanın (u,v) yıkılma olasılığını al
            p = get_agent_edge_probability(
                agent_id=i,
                edge=(u, v),
                policy=policy,
                n_agents=n_agents,
                policy_args=policy_args
            )
            # Bernoulli sample
            state = bernoulli.rvs(p)
            if state == 1:
                cmult = cost_function()
                heuristic_map[(u, v)] = graph[u][v]['base'] * cmult
                heuristic_map[(v, u)] = graph[u][v]['base'] * cmult
            else:
                heuristic_map[(u, v)] = graph[u][v]['base']
                heuristic_map[(v, u)] = graph[u][v]['base']

        # Ajan nesnesi
        agents.append(Vehicle(
            id=i,
            last_seen_node=source,
            traversed_nodes=[],
            current_edge=None,
            remaining_cost=0,
            heuristic_map=heuristic_map
        ))

    explored_edges = set()
    priority_queue = []
    heapq.heapify(priority_queue)

    total_time = 0.0
    route_change = 0

    finished_agents = set()
    agent_ids = set(a.id for a in agents)

    min_time = None

    while finished_agents != agent_ids:
        if min_time is not None:
            total_time += min_time

        for i, agent in enumerate(agents):
            if agent.id in finished_agents:
                continue

            if min_time is not None:
                agent.remaining_cost -= agent.velocity * min_time

            # Kenar bitti mi?
            if agent.remaining_cost <= 0:
                # Kenarın sonuna vardı
                if agent.current_edge is not None:
                    (n1, n2) = agent.current_edge

                    if(share):


                        explored_edges.add((n1, n2))
                        explored_edges.add((n2, n1))
                    agent.last_seen_node = n2
                else:
                    agent.last_seen_node = source

                # traversed_nodes güncelle
                agent.traversed_nodes.append((agent.last_seen_node, total_time))

                # Hedefe ulaştı mı?
                if agent.last_seen_node == target:
                    finished_agents.add(agent.id)
                    continue

                # Yeni rota
                def custom_weight(u, v, data):
                    if (u, v) in explored_edges or (v, u) in explored_edges:
                        # Bu kenarın gerçek weight'i
                        return graph[u][v]['weight']
                    else:
                        # Ajanın tahmini
                        return agent.heuristic_map[(u, v)]

                path = nx.dijkstra_path(graph, agent.last_seen_node, target, weight=custom_weight)

                if agent.prev_route is not None:
                    if agent.prev_route != path:
                        route_change += 1

                agent.prev_route = path
                if len(path) > 1:
                    agent.current_edge = (path[0], path[1])
                    agent.remaining_cost = graph[path[0]][path[1]]['weight']
                else:
                    # Tek node'luk path -> demek ki source==target
                    agent.current_edge = None
                    agent.remaining_cost = 0

            time_to_finish = agent.remaining_cost / agent.velocity if agent.remaining_cost>0 else float('inf')
            heapq.heappush(priority_queue, (time_to_finish, i))

        min_time, i_pop = heapq.heappop(priority_queue)
        if min_time == float('inf'):
            # Demek ki tüm ajanlar bitmiş veya route yok
            break

    agents.sort(key=lambda ag: (ag.traversed_nodes[-1][1] if ag.traversed_nodes else 0))
    return agents, route_change

def compute_path_cost(graph: nx.Graph, node_list: List[int]) -> float:
    total = 0.0
    for i in range(len(node_list) - 1):
        u = node_list[i]
        v = node_list[i+1]
        total += graph[u][v]['weight']
    return total
def compute_no_earthquake_path_cost(graph: nx.Graph, source: int, target: int) -> float:
    def base_weight(u, v, data):
        return graph[u][v]['base']

    path = nx.dijkstra_path(graph, source, target, weight=base_weight)
    return compute_path_cost(graph, path)


def compute_agent_route_cost(graph: nx.Graph, agent: Vehicle) -> float:
    node_list = [n for (n, t) in agent.traversed_nodes]
    return compute_path_cost(graph, node_list)

if __name__ == "__main__":
    """
    Örnek kullanım:
    1) Graf oluştur.
    2) Kenarlara base weight ver, 
    3) Gerçek (final) weight hesapla -> 'edge_bernoulli_probs' 
    4) Seçilen poltikaya uygun policy_args oluştur.
    5) algorithm(...) çağır.
    """


    # Parametreler
    N_ROWS =5
    N_COLS = 4
    N_AGENTS = 3
    SOURCE_NODE = 0
    TARGET_NODE = N_ROWS*N_COLS - 1
    MIN_WEIGHT = 5
    MAX_WEIGHT = 25

    # Maliyet fonksiyonu
    COST_FUNCTION = lambda: 1 / uniform.rvs(1e-15, 1)



    # Graph oluştur
    graph = create_random_manhattan_graph(N_ROWS, N_COLS)

    # Kenarlar için base tanımla, random bir p de atayalım
    # (Gerçek senaryo: p -> yıkılma olasılığı)
    edge_bernoulli_probs = {}
    for (u,v) in graph.edges():
        base_val = random.uniform(MIN_WEIGHT, MAX_WEIGHT)
        graph[u][v]['base'] = base_val
        graph[v][u]['base'] = base_val

        p = random.uniform(0, 1)
        edge_bernoulli_probs[(u,v)] = p
        edge_bernoulli_probs[(v,u)] = p

        # Gerçek weight
        st = bernoulli.rvs(p)
        if st == 1:
            mul = COST_FUNCTION()
            graph[u][v]['weight'] = base_val * mul
            graph[v][u]['weight'] = base_val * mul
        else:
            graph[u][v]['weight'] = base_val
            graph[v][u]['weight'] = base_val




    # policy_args
    # GRAPH_SAMPLING -> "edge_bernoulli_probs" gerekli
    # CONSTANT -> "const_p" gerekli
    # RANDOM -> ek arg. gerekmez
    # SHARED -> ek arg. gerekmez (n_agents>1 durumu)

    PROB_SAMPLING_POLICY = AgentPolicy.SHARED
    policy_args = {
         "edge_bernoulli_probs": edge_bernoulli_probs,
        # "const_p": 0.3
        # "random": {}
        # "shared": {}
    }

    # 1) Hiç deprem olmasa
    no_eq_cost = compute_no_earthquake_path_cost(graph, SOURCE_NODE, TARGET_NODE)

    # 2) Bilgi paylaşımı yok
    agents_no_sharing, _ = algorithm(
        graph=graph,
        n_agents=N_AGENTS,
        cost_function=COST_FUNCTION,
        source=SOURCE_NODE,
        target=TARGET_NODE,
        policy=PROB_SAMPLING_POLICY,
        policy_args=policy_args,
        share=False
    )
    total_no_sharing = 0.0
    for ag in agents_no_sharing:
        total_no_sharing += compute_agent_route_cost(graph, ag)
    avg_no_sharing = total_no_sharing / N_AGENTS

    # 3) Sharing açık -> ilk varan ajan
    agents_sharing, _ = algorithm(
        graph=graph,
        n_agents=N_AGENTS,
        cost_function=COST_FUNCTION,
        source=SOURCE_NODE,
        target=TARGET_NODE,
        policy=PROB_SAMPLING_POLICY,
        policy_args=policy_args,
        share=True
    )
    first_agent = agents_sharing[0]
    first_agent_cost = compute_agent_route_cost(graph, first_agent)

    # 4) Tüm ajanlar (sharing açık) -> ortalama rota
    total_sharing = 0.0
    for ag in agents_sharing:
        total_sharing += compute_agent_route_cost(graph, ag)
    avg_sharing = total_sharing / N_AGENTS

    # Ekrana basalım
    print(f"1) Hiç deprem yok varsayımı -> en kısa yol uzunluğu: {no_eq_cost:.2f}")
    print(f"2) Paylaşım yok -> ortalama rota: {avg_no_sharing:.2f}")
    print(f"3) Paylaşım açık -> ilk varan ajanın rota uzunluğu: {first_agent_cost:.2f}")
    print(f"4) Paylaşım açık -> ortalama rota: {avg_sharing:.2f}")

    # Çıktıları matplotlib ile görselleştirelim.
    # Birden fazla veriyi tek plotta çubuk grafikte gösterebiliriz.
    # Dilersen her bir değeri ayrı plotla da gösterebilirsin; burada hepsini tek çubuk grafikte gösteriyorum.

    labels = [
        "No Eq (base)",
        "No Sharing (avg)",
        "First Arrival (sharing)",
        "Avg Sharing",
    ]
    values = [
        no_eq_cost,
        avg_no_sharing,
        first_agent_cost,
        avg_sharing,
    ]

    # Tek çubuk grafik
    plt.figure()
    x_positions = range(len(values))
    plt.bar(x_positions, values)
    plt.xticks(x_positions, labels, rotation=25)
    plt.ylabel("Rota Uzunluğu")
    plt.title("Senaryo Kıyaslaması (Matplotlib)")

    # Grafiği göster
    plt.tight_layout()
    plt.show()

    # Ayrıca sharing açık olan senaryoyu animasyonla da izleyebiliriz
    visualise_algorithm(graph, agents_sharing, save_path="Videos")
