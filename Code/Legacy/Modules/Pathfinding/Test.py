import pandas as pd

from MonteCarlo import GraphMonteCarlo
from QLearning import GraphQLearning

import random

from Code.Modules.Data import EdgeStatus, DTCity
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import seaborn as sns

from concurrent.futures import ProcessPoolExecutor




EPOCH=10000
city = DTCity("Moda, Istanbul, Turkey", bernouilli_closedness=False)


def run_monte_carlo( start, end):
    print("Training")
    monte_carlo = GraphMonteCarlo(city, start, end, 0.05, 0.95, 0.2)
    real_path = city.generate_shortest_path(start, end)
    length_diff = []
    length_ratio = []


    for _ in range(EPOCH):
        monte_carlo.train(1)

        mc_path = monte_carlo.generate_path()
        if(len(mc_path)!=0):

            length_diff.append(len(mc_path) - len(real_path))
            length_ratio.append(len(mc_path) / len(real_path))
        else:
            length_diff.append(1000)
            length_ratio.append(1000)
    print("Training done")

    return length_diff, length_ratio

def run_qlearning(start,end):

    print("Training")
    q_learning = GraphQLearning(city, start, end, 0.05, 0.95, 0.2)
    real_path = city.generate_shortest_path(start, end)
    length_diff = []
    length_ratio = []

    for _ in range(EPOCH):
        q_learning.train(1)

        q_path = q_learning.generate_path()
        if(len(q_path)!=0):
            length_diff.append(len(q_path) - len(real_path))
            length_ratio.append(len(q_path) / len(real_path))
        else:
            length_diff.append(1000)
            length_ratio.append(1000)
    print("Training done")

    return length_diff, length_ratio


if __name__ == "__main__":

    reachable_pairs = city.reachable_pairs
    reachable_pairs = [(start, end) for start in reachable_pairs for end in reachable_pairs[start]]

    subset = random.choices(reachable_pairs, k=100)


    with ProcessPoolExecutor(max_workers=18) as executor:
        results_mc=executor.map(run_monte_carlo,[pair[0] for pair in subset],[pair[1] for pair in subset])

    length_diffs_mc=[]
    length_ratios_mc=[]

    for result in results_mc:
        length_diffs_mc.append(result[0])
        length_ratios_mc.append(result[1])

    length_diffs_mc=np.array(length_diffs_mc)
    length_ratios_mc=np.array(length_ratios_mc)

    mean_diffs_mc=np.mean(length_diffs_mc,axis=0)
    mean_ratios_mc=np.mean(length_ratios_mc,axis=0)




    with ProcessPoolExecutor(max_workers=18) as executor:
        results_q=executor.map(run_qlearning,[pair[0] for pair in subset],[pair[1] for pair in subset])

    length_diffs_q=[]
    length_ratios_q=[]

    for result in results_q:
        length_diffs_q.append(result[0])
        length_ratios_q.append(result[1])

    length_diffs_q=np.array(length_diffs_q)
    length_ratios_q=np.array(length_ratios_q)

    mean_diffs_q=np.mean(length_diffs_q,axis=0)
    mean_ratios_q=np.mean(length_ratios_q,axis=0)




    sns.scatterplot(x=range(EPOCH),y=mean_ratios_mc)
    sns.scatterplot(x=range(EPOCH),y=mean_ratios_q)

    plt.xlim(0,EPOCH)
    plt.ylim(1,2)
    plt.title("Mean Path Length/Real Path Length")

    plt.legend(["Monte Carlo","Q Learning"])

    plt.show()


    sns.scatterplot(x=range(EPOCH),y=mean_diffs_mc)
    sns.scatterplot(x=range(EPOCH),y=mean_diffs_q)

    plt.xlim(0,EPOCH)
    plt.ylim(0,10)


    plt.title("Mean Path Length-Real Path Length")
    plt.legend(["Monte Carlo","Q Learning"])

    plt.show()

