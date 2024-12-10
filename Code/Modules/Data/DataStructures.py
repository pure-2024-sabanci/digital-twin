import random

import networkx as nx
import osmnx
import matplotlib.pyplot as plt

import pprint
class DTCity():

    def __init__(self,address,initial_probability=0.5):

        self._graph=osmnx.graph_from_place(address,network_type='drive')

        self.adjacency_list=nx.to_dict_of_lists(self._graph)
        self.adjacency_matrix=nx.to_numpy_array(self._graph)

        self.nodes=[node for node in self.adjacency_list.keys()]
        self.edges=[(node,neighbor,initial_probability)
                    for node in self.adjacency_list.keys()
                    for neighbor in self.adjacency_list[node]]






    def generate_random_path(self,start,length):

        path=[start]
        for i in range(length):
            path.append(random.choice(self.adjacency_list[path[-1]]))
        return path

    def info(self):
        pprint.pprint(self.nodes)
        pprint.pprint(self.edges)
    def plot(self):
        osmnx.plot_graph(self._graph)
        plt.show()

    def plot_path(self,path):


        fig, ax = osmnx.plot_graph_route(self._graph, path)
        plt.show()



if __name__ == '__main__':
    city=DTCity('Moda, Istanbul, Turkey')

    city.info()

    city.plot()
    city.plot_path(city.generate_random_path(1457277967,60))
