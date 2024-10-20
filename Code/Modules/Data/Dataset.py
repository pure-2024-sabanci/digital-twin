from dataclasses import dataclass
from typing import Tuple
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
    cells: np.ndarray

    def __init__(self, shape: Tuple[int,int], cells: np.ndarray) -> None:

        self.shape = shape
        self.cells = cells

    def __repr__(self) -> str:

        return f"{self.cells}"