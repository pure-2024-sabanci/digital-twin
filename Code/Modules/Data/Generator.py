import numpy as np
from Dataset import Map, Cell
from typing import Tuple, List
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from queue import Queue
import enum


def clamp(n, min_val, max_val):
    return max(min_val, min(n, max_val))

class ParcelType(enum.Enum):
    ROAD = 1
    BUILDING = 2


class Generator():

    def __init__(self, seed: int, min_build_area: int) -> None:

        self._seed = seed
        self._min_build_area = min_build_area

    def generate(self, shape: Tuple[int, int], low: int, high: int,
                 road_width: int,
                 road_height: int,
                 building_width: int,
                 building_height: int,
                 ) -> Map:

        mat = self.__initialize_map(shape, low, high)

        self.__generate_helper((0, shape[0]), (0, shape[1]), mat, low, high,
                               road_width,
                               road_height,
                               building_width,
                               building_height)

        return Map(shape, mat)

    def __generate_helper(self, row_range: Tuple[int, int],
                          col_range: Tuple[int, int],
                          mat: [List[Cell]],
                          low: int, high: int,
                          road_width: int,
                          road_height: int,
                          building_width: int,
                          building_height: int,
                          ) -> None:

        area = (row_range[1] - row_range[0]) * (col_range[1] - col_range[0])
        if area <= self._min_build_area:
            print("Area too small")
            return
        else:
            print("Area: ", area)

        parcel_queue = []  # queue of parcels to be divided

        p_first = col_range[0]
        p_second = col_range[0]

        current = ParcelType.BUILDING

        while p_first < col_range[1]:

            if current == ParcelType.BUILDING:

                p_first = clamp(p_first + building_width, p_first, col_range[1])
                parcel=(p_second, p_first)
                parcel_queue.append(parcel)

                current = ParcelType.ROAD
                p_second = p_first

            else:
                p_first = clamp(p_first + road_width, p_first, col_range[1])

                for i in range(row_range[0], row_range[1]):
                    for j in range(p_second, p_first):

                        mat[i][j].value = 0

                p_second = p_first
                current = ParcelType.BUILDING


        sns.heatmap(np.array([[cell.value for cell in row] for row in mat]))
        plt.legend()
        plt.show()

        # Draw horizontal road



        num_vertical_parcel = len(parcel_queue)

        for _ in range(num_vertical_parcel):

            v_parcel= parcel_queue.pop(0)

            p_first = row_range[0]
            p_second = row_range[0]

            current = ParcelType.BUILDING

            while p_first < row_range[1]:

                if current == ParcelType.BUILDING:
                    p_first = clamp(p_first + building_height, p_first, row_range[1])
                    parcel_queue.append(((p_second, p_first),v_parcel))

                    current = ParcelType.ROAD
                    p_second = p_first

                else:

                    p_first = clamp(p_first + road_height, p_first, row_range[1])


                    for i in range(v_parcel[0], v_parcel[1]):
                        for j in range(p_second, p_first):

                            mat[j][i].value = 0

                    p_second = p_first
                    current = ParcelType.BUILDING





        sns.heatmap(np.array([[cell.value for cell in row] for row in mat]))
        plt.legend()
        plt.show()





    def __initialize_map(self, shape: Tuple[int, int], low: int, high: int) -> List[List[Cell]]:

        map = []

        for i in range(shape[0]):
            line = []
            for j in range(shape[1]):
                line.append(Cell((i, j), 1))
            map.append(line)

        return map


if (__name__ == "__main__"):
    generator = Generator(42, 5)

    map = generator.generate((500, 500), 1, 5, 2, 2, 10, 15)
