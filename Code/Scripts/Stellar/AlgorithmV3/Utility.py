import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import numpy as np
import random
import time
import os

import matplotlib
# Rastgele mesafelerle Manhattan graph oluşturma fonksiyonu
def create_random_manhattan_graph(rows, cols, min_dist=1, max_dist=3):
    G = nx.Graph()

    node_id = 0
    # Düğümleri ekle
    for i in range(rows):
        for j in range(cols):
            G.add_node(node_id, pos=(j, -i))
            node_id += 1

    for i in range(rows):
        for j in range(cols):

            if j < cols - 1:
                G.add_edge(i * cols + j, i * cols + j + 1)
                G.add_edge(i * cols + j + 1, i * cols + j)


            if i < rows - 1:
                G.add_edge(i * cols + j, (i + 1) * cols + j)
                G.add_edge((i + 1) * cols + j, i * cols + j)
    return G

def draw_graph(graph,weight_name='weight'):
    pos=nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(6, 6))
    edges = graph.edges(data=True)
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray")

    edge_labels = {(u, v): f"{d[weight_name]:.1f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Random Weighted Manhattan Graph (5x5 Grid)")

    plt.show()

def get_node_node_id(N_ROW,N_COL,pos):
    return pos[1]+pos[0]*N_ROW


import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import numpy as np

def visualise_algorithm(graph, agents,save_path):
    """
    Tüm ajanların (node, time) kayıtlarını tek bir listede [agent_id, node, time]
    olarak toplayıp time'a göre sıralar. Her olay (frame) için ilgili ajanın
    rotasına yeni node eklenir ve animasyon adım adım gösterilir.

    Varsayımlar:
      - graph: node'larında "pos" = (x, y) attribute'u var (örn. create_random_manhattan_graph).
      - agents: her agent'da:
           agent.agent_id (str veya benzeri)
           agent.traversed_nodes = [(node, time), (node, time), ...] (SIRAYI KORUMUYORUZ;
             buradan tek tek alıp global listede time'a göre SIRALIYORUZ)
    """

    # 1) Node konumları
    pos = nx.get_node_attributes(graph, 'pos')

    # 2) Kenar ağırlıkları (2 ondalıklı gösterim)
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    formatted_edge_labels = {
        e: f"{edge_weights[e]:.2f}" for e in edge_weights
    } if edge_weights else {}

    # 3) Tüm ajanların verilerini [agent_id, node, time] formatında tek listede topla
    combined_events = []
    for agent in agents:
        agent_id = getattr(agent, 'id', None)
        for (node, t) in agent.traversed_nodes:
            combined_events.append((agent_id, node, t))

    # time'a göre sırala
    combined_events.sort(key=lambda x: x[2])  # x[2] = time

    # 4) animasyon çerçeve sayısı => her olay bir frame
    num_frames = len(combined_events)

    # 5) Figure oluştur
    fig, ax = plt.subplots(figsize=(7,6))
    plt.title("Ajan Rotaları (Tüm agentlar time'a göre birleşik)")

    # Graf çizimi

    edge_colors = []

    for u, v in graph.edges():
        if graph[u][v]['weight'] == graph[u][v]['base']:
            edge_colors.append('black')

        else:
            edge_colors.append('red')


    formatted_edge_labels = {}
    not_formatted_edge_labels = {}
    for (u, v) in graph.edges():
        if graph[u][v]['weight'] != graph[u][v]['base']:
            formatted_edge_labels[(u, v)] = f"{graph[u][v]['weight']:.2f}"
        else:
            not_formatted_edge_labels[(u, v)] = f"{graph[u][v]['weight']:.2f}"

    nx.draw_networkx_nodes(graph, pos, node_color='lightgray', node_size=400, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=ax, edge_color=edge_colors)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)
    if formatted_edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=formatted_edge_labels, ax=ax,font_color='red')
    if not_formatted_edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=not_formatted_edge_labels, ax=ax,font_color='black')

    # 6) Her agent için scatter, line, text objesi
    agent_ids = [getattr(a, 'id', f"A{i}") for i,a in enumerate(agents)]
    # agent_ids benzersiz olduğunu varsayalım.
    agent_ids_unique = list(set(agent_ids))

    # agent'a dair verileri kaydetmek için dict
    agent_scatter_dict = {}
    agent_line_dict = {}
    agent_text_dict = {}
    visited_nodes_dict = {}  # her agent_id icin o ana dek gidilen node listesi

    import random
    color_map = {}
    for i, aid in enumerate(agent_ids_unique):
        color_map[aid] = (random.random(), random.random(), random.random())

    for aid in agent_ids_unique:
        sc = ax.scatter([], [], c=[color_map[aid]], s=150, edgecolor='black', zorder=3)
        ln, = ax.plot([], [], color=color_map[aid], linewidth=2, alpha=0.7, zorder=2)
        txt = ax.text(0,0, aid,
                      fontsize=10, fontweight='bold',
                      horizontalalignment='center',
                      verticalalignment='bottom')
        agent_scatter_dict[aid] = sc
        agent_line_dict[aid] = ln
        agent_text_dict[aid] = txt
        visited_nodes_dict[aid] = []

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=12, fontweight='bold')

    # 7) update fonksiyonu
    def update(frame_idx):
        # Her frame bir olay
        if frame_idx >= num_frames:
            return

        # Olay:
        (aid, node, t) = combined_events[frame_idx]

        # Ajana bu node'u ekle
        visited_nodes_dict[aid].append(node)

        # Zamanı yaz
        time_text.set_text(f"Time: {t:.2f}")

        # Tüm agentları güncelle
        for agentid in agent_ids_unique:
            sc = agent_scatter_dict[agentid]
            ln = agent_line_dict[agentid]
            txt = agent_text_dict[agentid]

            visited_list = visited_nodes_dict[agentid]
            if len(visited_list) == 0:
                # Scatter boş
                sc.set_offsets(np.empty((0,2)))
                ln.set_data([], [])
                txt.set_position((0, 0))
            else:
                last_node = visited_list[-1]
                sc.set_offsets([pos[last_node]])
                # rota
                coords = [pos[n] for n in visited_list]
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                ln.set_data(xs, ys)
                # etiket
                txt.set_position((xs[-1], ys[-1]+0.15))

        # Return
        # (matplotlib animasyonu icin guncelledigimiz artisleri döndürelim)
        # sc, ln, txt...
        result_artists = list(agent_scatter_dict.values()) \
                        + list(agent_line_dict.values()) \
                        + list(agent_text_dict.values()) \
                        + [time_text]
        return result_artists

    ani = animation.FuncAnimation(fig, update,
                                  frames=num_frames, interval=800,
                                  blit=False, repeat=False)


    # write a path to save the video


    # Create the folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create a unique filename for the animation
    unique_filename = os.path.join(save_path, f"Ajan_Rotalari_{int(time.time())}.mp4")

    # kaydet
    writervideo = animation.FFMpegWriter(fps=1)
    ani.save(unique_filename, writer=writervideo)
    
    print(f"Video successfully saved at: {unique_filename}")

    plt.close(fig)





def main():
    """
    Örnek main fonksiyonu, create_random_manhattan_graph ile bir graf oluşturup
    içine basit bir edge ağırlığı atar, sonra 2 tane örnek agent oluşturur
    ve en sonunda visualise_algorithm ile animasyonu görüntüler.
    """

    # 1) Parametreler
    n_rows = 5
    n_cols = 5

    # 2) Graf oluştur
    G = create_random_manhattan_graph(n_rows, n_cols)

    # Örnek olarak her edge'e rastgele bir weight ekleyelim
    for (u, v) in G.edges():
        w = random.randint(1, 10)
        G[u][v]["weight"] = w
        G[v][u]["weight"] = w  # iki yönlü

    # 3) Örnek Ajan Nesneleri
    # Ajanlarda traversed_nodes = [(node_id, time), ...]
    # ve isterseniz agent_id gibi bir alan olsun.
    class Agent:
        def __init__(self, agent_id, traversed_nodes):
            self.id = agent_id
            self.traversed_nodes = traversed_nodes

    # Örnek olarak 2 tane basit rota oluşturuyoruz
    # Agent A0 => 3 node ziyaret
    # Agent A1 => 4 node ziyaret
    A0 = Agent(
        "A0",
        [
            (0, 0),  # (node_id=0, t=0)
            (1, 2),  # 2. saniyede node=1
            (6, 5)  # 5. saniyede node=6
        ]
    )

    A1 = Agent(
        "A1",
        [
            (24, 0),  # en altta-right'ta
            (19, 1),  # ...
            (14, 3),
            (9, 6)
        ]
    )

    agents = [A0, A1]

    # 4) Görselleştir
    visualise_algorithm(G, agents)


if __name__ == "__main__":
    main()
