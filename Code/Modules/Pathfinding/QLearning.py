
from Code.Modules.Data import EdgeStatus, DTCity
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import seaborn as sns


class GraphQLearning():

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

            reward=self.reward(next_state)

            update=self.q_dict[current_state,next_state]

            neighbours_of_next=self.state_action(next_state)

            max_q_next=max([self.q_dict[next_state,neighbour] for neighbour in neighbours_of_next])

            update+=self.alpha*(reward+self.gamma*max_q_next-self.q_dict[current_state,next_state])

            self.q_dict[current_state,next_state]=update

            history.append(next_state)

            current_state = next_state

        return history

    def train(self, episodes):

        for _ in tqdm(range(episodes)):

            history=self.episode()
            self.q_dict_history.append(self.q_dict.copy())

    def generate_path(self):
        current_node=self.start
        path=[current_node]

        while current_node!=self.end:
            actions=self.state_action(current_node)
            values = [self.q_dict[current_node, action] for action in actions]

            next_node=actions[np.argmax(values)]

            if next_node is None:
                print("No path found")
                break
            path.append(next_node)
            current_node=next_node


        return path




if __name__ == "__main__":

    city = DTCity("Moda, Istanbul, Turkey")
    pairs=city.reachable_pairs


    q_learning = GraphQLearning(city, 0, 3, 0.005, 0.8, 0.1)

    q_learning.train(100)
    real_path=city.generate_shortest_path(0,3)
    q_learning_path=q_learning.generate_path()

    print(real_path)
    print(q_learning_path)

    city.plot_path(real_path)
    city.plot_path(q_learning_path)


