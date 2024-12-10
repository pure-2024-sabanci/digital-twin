import numpy as np
from Code.Modules.Data.MapStructures import Map, Cell
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
                 max_road_distance: int,
                 max_road_size: int,
                 ) -> Map:

        mat = self.__initialize_map(shape, low, high)

        self.__generate_helper((0, shape[0]), (0, shape[1]), mat, low, high,
                               max_road_distance,
                               max_road_size)
        return Map(shape, mat)

    def __generate_helper(self, row_range: Tuple[int, int],
                          col_range: Tuple[int, int],
                          mat: [List[Cell]],
                          low: int, high: int,
                          max_road_distance: int,
                          max_road_size: int,
                          ) -> None:

        area = (row_range[1] - row_range[0]) * (col_range[1] - col_range[0])

        if area < self._min_build_area:
            return
        if max_road_distance <= 1 or max_road_size <= 1:
            return

        cursor = col_range[0]

        parcels = []

        parcel_start = col_range[0]  #previous parcel start

        while cursor < col_range[1]:

            road_size = 1
            road_distance = np.random.randint(1, max_road_distance)

            parcels.append((parcel_start,
                            clamp(parcel_start + road_distance, col_range[0], col_range[1])))

            cursor += road_distance

            if cursor + road_size >= col_range[1]:
                break

            for i in range(row_range[0], row_range[1]):
                for j in range(cursor, cursor + road_size):
                    mat[i][j].value = 0

            cursor += road_size
            parcel_start = cursor

        for _ in range(len(parcels)):
            parcel = parcels.pop(0)

            cursor = row_range[0]

            parcel_start = row_range[0]

            while cursor < row_range[1]:

                road_size = 1
                road_distance = np.random.randint(1, max_road_distance)

                cursor += road_distance

                parcels.append(((parcel_start,
                                 clamp(cursor, row_range[0], row_range[1])),

                                parcel))

                if cursor + road_size >= row_range[1]:
                    break

                for i in range(parcel[0], parcel[1]):
                    for j in range(cursor, cursor + road_size):

                        mat[j][i].value = 0

                cursor += road_size + road_distance

                parcel_start = cursor

        for parcel in parcels:
            self.__generate_helper(parcel[0],
                                   parcel[1],
                                   mat,
                                   low,
                                   high,
                                   max_road_distance - 1,
                                   max_road_size)

    def __initialize_map(self, shape: Tuple[int, int], low: int, high: int) -> List[List[Cell]]:

        map = []

        for i in range(shape[0]):
            line = []
            for j in range(shape[1]):
                line.append(Cell((i, j), np.random.randint(low, high)))
            map.append(line)

        return map


if (__name__ == "__main__"):
    generator = Generator(42, 10)



    map = generator.generate((15, 15), 1, 100, 20, 1)

    sns.heatmap([[cell.value for cell in row] for row in map.cells])
    plt.show()
    print("Done")
