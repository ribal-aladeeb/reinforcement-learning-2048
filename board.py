from __future__ import annotations # in order to allow type hints for a class referring to itself
import numpy as np
from typing import List
import random


class Board2048:
    

    def __init__(self, k: int = 4):
        self.state: np.array = np.zeros(shape=(k, k))
        self._empty_spot_numbers: List[int] = [2,4]
        self._populate_empty_cell()
        self._populate_empty_cell()


    def __repr__(self):
        return str(self.state)


    def _populate_empty_cell(self) -> Board2048:
        """
        This method checks the board for an empty cell (contains 0)
        and populates one empty cell chosen at random with either a 2 or a 4
        """
        indices = np.array(np.where(self.state == 0)).T
        i = random.randint(0, len(indices)-1)
        x,y = indices[i]
        number = np.random.choice(self._empty_spot_numbers,1)
        self.state[x,y] = number
        return self


    def up(self):
        pass


    def down(self):
        pass


    def left(self):
        pass


    def right(self):
        pass

if __name__ == "__main__":
    board = Board2048()
    for i in range(9):
        print(Board2048())
        print(i-1)