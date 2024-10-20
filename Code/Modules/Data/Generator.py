import numpy as np
from typing import Tuple


class Generator():

    def __init__(self, seed:int) -> None:

        self.seed=seed

    def generate(self, shape:Tuple[int,int], low:int, high:int) -> np.ndarray:











