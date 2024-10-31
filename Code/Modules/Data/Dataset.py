from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

@dataclass
class Cell():

    coordinates: Tuple[int,int]
    value: int
    is_visited: bool

    def __init__(self, coordinates: Tuple[int,int], value: int, is_visited: bool=False) -> None:

        self.coordinates = coordinates
        self.value = value
        self.is_visited = is_visited


    def __repr__(self) -> str:

        return f"{self.value}"




@dataclass
class Map():

    shape: Tuple[int,int]
    cells: List[List[Cell]]

    def __init__(self, shape: Tuple[int,int], cells: List[List[Cell]]) -> None:

        self.shape = shape
        self.cells = cells



    def __repr__(self) -> str:

        rep=""

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                rep+=f"{self.cells[i][j]} "
            rep+="\n"

        return rep



