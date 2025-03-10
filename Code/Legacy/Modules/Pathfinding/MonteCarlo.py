import random

from Code.Modules.Data import EdgeStatus, DTCity
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import seaborn as sns

from concurrent.futures import ProcessPoolExecutor

class GraphMonteCarlo():

    def __init__(self, graph: DTCity, start, end, alpha,gamma,epsilon) -> None:

        self.graph = graph

        self.start = graph.nodes[start]
        self.end = graph.nodes[end]

        self.q_dict = {(node, neighbour): 0 for node in self.graph.nodes for neighbour in self.graph.get_neighbors(node)}
        self.q_dict_history = [self.q_dict.copy()]

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon

    def reward(self, node):
        if node == self.end:
            return 1

        else:
            return 0

    def state_action(self, node):

        neighbours = self.graph.get_neighbors(node)

        return [neighbour for neighbour in neighbours if self.graph.edges[(node, neighbour)]["status"] == EdgeStatus.OPEN]

    def epsilon_greedy_policy(self, node):

        actions = self.state_action(node)
        if len(actions) == 0:
            return None

        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        else:

            values=[self.q_dict[node,action] for action in actions]
            return actions[np.argmax(values)]

    def episode(self):

        history=[self.start]

        current_state = self.start

        while current_state != self.end:

            next_state = self.epsilon_greedy_policy(current_state)

            if next_state is None:
                print("No path found")
                break

            history.append(next_state)

            current_state = next_state

        return history

    def update_q_table(self, history):

        g=0

        for i in range(len(history)-1,0,-1):

            g=self.gamma*g+self.reward(history[i])
            state,action=history[i-1],history[i]

            self.q_dict[state,action]=self.q_dict[state,action]+self.alpha*(g-self.q_dict[state,action])


    def train(self, episodes):

        for _ in range(episodes):

            history=self.episode()

            self.update_q_table(history)
            self.q_dict_history.append(self.q_dict.copy())

    def generate_path(self):
        current_node=self.start
        path=[current_node]

        while current_node!=self.end:
            actions=self.state_action(current_node)
            values = [self.q_dict[current_node, action] for action in actions]

            next_node=actions[np.argmax(values)]

            if next_node is None or next_node in path:
                return []

            path.append(next_node)
            current_node=next_node


        return path

