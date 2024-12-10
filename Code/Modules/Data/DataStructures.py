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
    _graph : networkx.MultiDiGraph
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

    def __init__(self, address: str) -> None:
        """
        Constructs all the necessary attributes for the DTCity object.

        Parameters:
        ----------
        address : str
            The address to generate the city graph from.
        """
        self._graph = osmnx.graph_from_place(address, network_type='drive')
        self._adjacency_list = nx.to_dict_of_lists(self._graph)
        self._adjacency_matrix = nx.to_numpy_array(self._graph)
        self.nodes: List[int] = [node for node in self._adjacency_list.keys()]
        self.edges: Dict[Tuple[int, int], EdgeStatus] = {(edge[0], edge[1]): EdgeStatus.OPEN for edge in
                                                         self._graph.edges}

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
        pprint.pprint(self.nodes)
        pprint.pprint(self.edges)

    def plot(self) -> None:
        """
        Plots the graph with edges colored based on their status.
        """
        edge_colors: List[str] = ['green' if edge[2] == EdgeStatus.OPEN else 'red' for edge in self.edges]
        osmnx.plot_graph(self._graph, edge_color=edge_colors)
        plt.show()

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
    city.info()
    city.plot()
    city.plot_path(city.generate_random_path(1457277967, 60))
