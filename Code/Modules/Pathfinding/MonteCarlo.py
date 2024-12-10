from ..Data import EdgeStatus, DTCity
import numpy as np

class GraphMonteCarlo():

    def __init__(self, graph: DTCity, start, end, alpha,gamma,epsilon) -> None:

        self.graph = graph

        self.start = start
        self.end = end

        self.q_dict = {(node, neighbour): 0 for node in self.graph.nodes for neighbour in self.graph.get_neighbors(node)}

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

        return [neighbour for neighbour in neighbours if self.graph.edges[(node, neighbour)] == EdgeStatus.OPEN]

    def epsilon_greedy_policy(self, node, epsilon):

        if np.random.rand() < epsilon:
            return np.random.choice(self.state_action(node))
        else:
            return np.argmax(self.q_dict[node])

    def episode(self):

        history=[self.start]

        current_state = self.start

        while current_state != self.end:

            next_state = self.epsilon_greedy_policy(current_state, self.epsilon)

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






















