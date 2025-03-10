import random
import seaborn as sns
from matplotlib import pyplot as plt

from Code.Modules.Data import Map,Generator

def random_walk(pos: [int,int],num_steps,map:Map) -> [int]:
    """
    Generate a random walk from start to end
    """

    visited=set()
    for i in range(num_steps):


        possible_directions = map.get_possible_directions(pos)

        if len(possible_directions)==0:
            break

        direction=random.choice(possible_directions)

        pos[0]+=direction[0]
        pos[1]+=direction[1]

        print(pos,"type(pos")


        visited.add((pos[0],pos[1]))


    return visited





if __name__=="__main__":

    random.seed(42)


    generator = Generator(42, 10)
    map=generator.generate((1000,1000),0,100,70,10)

    sns.heatmap([[cell.value for cell in row] for row in map.cells])
    plt.show()

    vehicle=None
    for i in range(450,550):
        for j in range(450,550):
            if map.cells[i][j].value==0:
                vehicle=[i, j]

                break
        if vehicle!=None:
            break


    v=random_walk(vehicle, 100000, map)

    print(list(v))
    for vehicle in v:

        map.cells[vehicle[0]][vehicle[1]].value=-100



    sns.heatmap([[cell.value for cell in row] for row in map.cells])
    plt.show()

