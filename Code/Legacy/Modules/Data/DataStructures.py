import random
import  networkx as nx
import osmnx
import matplotlib.pyplot as plt
import pprint
from enum import Enum
from typing import List, Tuple, Dict
import numpy.random as npr

class EdgeStatus(Enum):
    OPEN = 1
    CLOSED = 0



class DTCity:
    """
    A class to represent a city using OpenStreetMap data.

    Attributes:
    ----------
    _graph : networkx.MultiGraph
        The graph representation of the city.
    _adjacency_list : dict
        The adjacency list representation of the graph.
    _adjacency_matrix : numpy.ndarray
        The adjacency matrix representation of the graph.
    nodes : List[int]
        List of nodes in the graph.
    edges : List[Tuple[int, int, EdgeStatus]]
        List of edges in the graph with their status.
    """

    def __init__(self, address: str,only_reachability=True,bernouilli_closedness=False) -> None:
        """
        Constructs all the necessary attributes for the DTCity object.

        Parameters:
        ----------
        address : str
            The address to generate the city graph from.
        """
        self._graph = osmnx.graph_from_place(address, network_type='drive',retain_all=False).to_undirected()
        self._graph=nx.convert_node_labels_to_integers(self._graph)

        self._adjacency_list = nx.to_dict_of_lists(self._graph,)



        self.nodes: List[int] = [node for node in self._adjacency_list.keys()]
        self.edges: Dict[Tuple[int, int], Dict] = {}

        for node in self._adjacency_list:
            for neighbor in self._adjacency_list[node]:
                length = self._graph[node][neighbor][0]['length']

                if(only_reachability):
                    length=1 if length>0 else 0

                if(bernouilli_closedness):
                    status=npr.binomial(1,0.5)
                    edge_status=EdgeStatus.CLOSED if status==0 else EdgeStatus.OPEN

                    self.edges[(node, neighbor)] = {"length": length, "status": edge_status}
                    self.edges[(neighbor, node)] = {"length": length, "status": edge_status}

                    self._graph[node][neighbor][0]['length']=status
                    self._graph[neighbor][node][0]['length']=status
                else:
                    self.edges[(node, neighbor)] = {"length": length, "status": EdgeStatus.OPEN}
                    self.edges[(neighbor, node)] = {"length": length, "status": EdgeStatus.OPEN}

        apsp = dict(nx.all_pairs_shortest_path(self._graph))
        self.reachable_pairs={k:[] for k in self.nodes}
        for k,v in apsp.items():
            for k2 in v:
                if k!=k2:
                    self.reachable_pairs[k].append(k2)


    def get_neighbors(self, node: int) -> List[int]:
        """
        Returns the neighbors of a given node.

        Parameters:
        ----------
        node : int
            The node to get the neighbors of.

        Returns:
        -------
        List[int]
            The neighbors of the given node.
        """
        return self._adjacency_list[node]

    def generate_random_path(self, start: int, length: int) -> List[int]:
        """
        Generates a random path of a given length starting from a given node.

        Parameters:
        ----------
        start : int
            The starting node of the path.
        length : int
            The length of the path.

        Returns:
        -------
        List[int]
            A list of nodes representing the path.
        """
        path: List[int] = [start]
        for _ in range(length):
            path.append(random.choice(self._adjacency_list[path[-1]]))
        return path

    def info(self) -> None:
        """
        Prints the nodes and edges of the graph.
        """
        for row in self.adjacency_matrix:
            pprint.pprint(row)
    def plot(self) -> None:
        """
        Plots the graph with edges colored based on their status.
        """
        edge_colors: List[str] = ['green' if self.edges[edge] == EdgeStatus.OPEN else 'red' for edge in self.edges]
        osmnx.plot_graph(self._graph, edge_color=edge_colors)
        plt.show()

    def generate_shortest_path(self, start: int, end: int) -> List[int]:
        import random
        import networkx as nx
        import osmnx
        import matplotlib.pyplot as plt
        import pprint
        from enum import Enum
        from typing import List, Tuple, Dict

        class EdgeStatus(Enum):
            OPEN = 1
            CLOSED = 0

        class DTCity:
            """
            A class to represent a city using OpenStreetMap data.

            Attributes:
            ----------
            _graph : networkx.MultiGraph
                The graph representation of the city.
            _adjacency_list : dict
                The adjacency list representation of the graph.
            adjacency_matrix : numpy.ndarray
                The adjacency matrix representation of the graph.
            nodes : List[int]
                List of nodes in the graph.
            edges : Dict[Tuple[int, int], EdgeStatus]
                Dictionary of edges in the graph with their status.
            """

            def __init__(self, address: str) -> None:
                """
                Constructs all the necessary attributes for the DTCity object.

                Parameters:
                ----------
                address : str
                    The address to generate the city graph from.
                """
                self._graph = osmnx.graph_from_place(address, network_type='drive').to_undirected()
                self._adjacency_list = nx.to_dict_of_lists(self._graph)
                self.adjacency_matrix = nx.to_numpy_array(self._graph)
                self.nodes: List[int] = [node for node in self._adjacency_list.keys()]
                self.edges: Dict[Tuple[int, int], EdgeStatus] = {}

                for node in self._adjacency_list:
                    for neighbor in self._adjacency_list[node]:
                        self.edges[(node, neighbor)] = EdgeStatus.OPEN
                        self.edges[(neighbor, node)] = EdgeStatus.OPEN

            def get_neighbors(self, node: int) -> List[int]:
                """
                Returns the neighbors of a given node.

                Parameters:
                ----------
                node : int
                    The node to get the neighbors of.

                Returns:
                -------
                List[int]
                    The neighbors of the given node.
                """
                return self._adjacency_list[node]

            def generate_random_path(self, start: int, length: int) -> List[int]:
                """
                Generates a random path of a given length starting from a given node.

                Parameters:
                ----------
                start : int
                    The starting node of the path.
                length : int
                    The length of the path.

                Returns:
                -------
                List[int]
                    A list of nodes representing the path.
                """
                path: List[int] = [start]
                for _ in range(length):
                    path.append(random.choice(self._adjacency_list[path[-1]]))
                return path

            def info(self) -> None:
                """
                Prints the nodes and edges of the graph.
                """
                for row in self.adjacency_matrix:
                    pprint.pprint(row)

            def plot(self) -> None:
                """
                Plots the graph with edges colored based on their status.
                """
                edge_colors: List[str] = ['green' if self.edges[edge] == EdgeStatus.OPEN else 'red' for edge in
                                          self.edges]
                osmnx.plot_graph(self._graph, edge_color=edge_colors)
                plt.show()

            def generate_shortest_path(self, start: int, end: int) -> List[int]:
                """
                Generates the shortest path between two nodes.

                Parameters:
                ----------
                start : int
                    The starting node of the path.
                end : int
                    The ending node of the path.

                Returns:
                -------
                List[int]
                    A list of nodes representing the shortest path.
                """


                start = self.nodes[start]
                end = self.nodes[end]

                return nx.shortest_path(self._graph, start, end)

            def plot_path(self, path: List[int]) -> None:
                """
                Plots a given path on the graph.

                Parameters:
                ----------
                path : List[int]
                    The path to plot.
                """
                fig, ax = osmnx.plot_graph_route(self._graph, path)
                plt.show()

        if __name__ == '__main__':
            city = DTCity('Moda, Istanbul, Turkey')

            city.plot()
            city.plot_path(city.generate_random_path(1457277967, 60))
        start= self.nodes[start]
        end = self.nodes[end]

        return nx.shortest_path(self._graph, start, end)



    def plot_path(self, path: List[int]) -> None:
        """
        Plots a given path on the graph.

        Parameters:
        ----------
        path : List[int]
            The path to plot.
        """



        fig, ax = osmnx.plot_graph_route(self._graph, path)
        plt.show()


if __name__ == '__main__':
    city = DTCity('Moda, Istanbul, Turkey')
    print(city.reachable_pairs[0])