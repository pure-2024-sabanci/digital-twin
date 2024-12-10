import random

import networkx as nx
import osmnx
import matplotlib.pyplot as plt

import pprint
class DTCity():

    def __init__(self,address):

        self._graph=osmnx.graph_from_place(address,network_type='drive')

        self.adjacency_list=nx.to_dict_of_lists(self._graph)
        self.adjacency_matrix=nx.to_numpy_array(self._graph)

    def generate_random_path(self,start,length):

        path=[start]
        for i in range(length):
            path.append(random.choice(self.adjacency_list[path[-1]]))
        return path

    def info(self):
        pprint.pprint(self.adjacency_list)
        pprint.pprint(self.adjacency_matrix)
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
