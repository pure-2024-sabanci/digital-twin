import random

import pygame
from matplotlib import pyplot as plt
import seaborn as sns

from Code.Modules.Data import Generator

SCREEN_SIZE=(1440,1080)
CELL_SIZE=5

pygame.init()

screen=pygame.display.set_mode(SCREEN_SIZE)
clock=pygame.time.Clock()

running=True


def draw_grid(map,screen,vehicle):

    n=len(map.cells)

    pos=[20,20]

    for i in range(n):
        for j in range(n):
            col="red" if map.cells[i][j].value>0 else "black"
            pygame.draw.rect(screen,col,(pos[1],pos[0],CELL_SIZE,CELL_SIZE))
            pos[1]+=CELL_SIZE
        pos[0]+=CELL_SIZE
        pos[1]=20

    pygame.draw.rect(screen,
                     "yellow",
                     (20+CELL_SIZE*vehicle[1],
                           20+CELL_SIZE*vehicle[0],
                     CELL_SIZE,CELL_SIZE))

generator = Generator(42, 4)
map = generator.generate((100, 100), 1, 100, 70, 10)

sns.heatmap([[cell.value for cell in row] for row in map.cells])
plt.show()

vehicle = None
for i in range(0, 100):
    for j in range(0, 100):
        if map.cells[i][j].value == 0:
            vehicle = [i, j]

            break
    if vehicle != None:
        break





while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    screen.fill("purple")

    possible_directions = map.get_possible_directions(vehicle)


    if(len(possible_directions)!=0):
        direction = random.choice(possible_directions)

        vehicle[0] += direction[0]
        vehicle[1] += direction[1]


    if(map.cells[vehicle[0]][vehicle[1]].value!=0):
        print(map.cells[vehicle[0]][vehicle[1]].value)


    draw_grid(map,screen,vehicle)

    pygame.display.flip()
    clock.tick(25)

pygame.quit()