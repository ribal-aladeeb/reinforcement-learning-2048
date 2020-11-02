from __future__ import annotations  # in order to allow type hints for a class referring to itself
import numpy as np
from typing import List
import random

from numpy.core.fromnumeric import nonzero


class Board2048:

    def __init__(self, k: int = 4):
        self.state: np.array = np.zeros(shape=(k, k))
        self._empty_spot_numbers: List[int] = [2, 4]
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
        x, y = indices[i]
        number = np.random.choice(self._empty_spot_numbers, 1)
        self.state[x, y] = number
        return self

    def _apply_action_to_vector(self, vector: np.array) -> np.array:

        vector = np.copy(vector)
        current = 0

        while current < len(vector)-1:

            non_zero_indices = np.where(vector != 0)[0]  # non zero indices (updated)
            # ensure that value of current is non-zero
            if len(non_zero_indices) == 0 or non_zero_indices[-1] <= current:
                return vector
            else:
                non_zero_indices = non_zero_indices[current < non_zero_indices]
                if len(non_zero_indices) == 0:
                    return vector

            if vector[current] == 0:
                # we know that there is a non_zero value at an index further than current
                #print(current, vector, non_zero_indices[0], vector[current])
                vector[current] += vector[non_zero_indices[0]]
                vector[non_zero_indices[0]] = 0

            elif vector[current] == vector[non_zero_indices[0]]:
                # we know that there is a non_zero value at an index further than current
                #print(current, vector, non_zero_indices[0], vector[current])
                vector[current] += vector[non_zero_indices[0]]
                vector[non_zero_indices[0]] = 0
                current += 1

            elif current + 1 == non_zero_indices[0]:
                current += 1
                continue

            else:
                # you have a non-zero entry at vector[current] and you have a non-zero entry somewhere further than current, but you don't know how many 0 are in between
                vector[current+1] = vector[non_zero_indices[0]]
                vector[non_zero_indices[0]] = 0
                current += 1

        return vector

    def up(self):
        pass

    def down(self):
        pass

    def left(self):
        # apply_along_axis(func1d, axis, arr, *args, **kwargs)
        result_matrix = np.apply_along_axis(self._apply_action_to_vector, axis=1, arr=self.state)
        if not np.equal(result_matrix,self.state).all():
            self.state = result_matrix
            self._populate_empty_cell()  
        return self

    def right(self):
        pass


if __name__ == "__main__":
    board = Board2048()
    print(board)
    for i in range(100000):
        print(i)
        print(board.left())
        print()
    # examples = [
    #     ([0, 0, 0, 0], [0, 0, 0, 0]),
    #     ([0, 0, 0, 2], [2, 0, 0, 0]),
    #     ([0, 0, 2, 2], [4, 0, 0, 0]),
    #     ([2, 0, 0, 0], [2, 0, 0, 0]),
    #     ([2, 0, 2, 0], [4, 0, 0, 0]),
    #     ([2, 2, 2, 2], [4, 4, 0, 0]),
    #     ([2, 2, 4, 4], [4, 8, 0, 0]),
    #     ([2, 2, 0, 0], [4, 0, 0, 0]),
    #     ([2, 0, 0, 2], [4, 0, 0, 0]),
    #     ([0, 0, 2, 2], [4, 0, 0, 0]),
    #     ([2, 4, 2, 4], [2, 4, 2, 4]),
    #     ([2, 2, 4, 2], [4, 4, 2, 0]),
    #     ([2, 4, 4, 2], [2, 8, 2, 0]),
    #     ([2, 4, 4, 4], [2, 8, 4, 0]),
    #     ([4, 8, 16, 32], [4, 8, 16, 32])
    # ]
    # for example in examples:
    #     result = board._apply_action_to_vector(np.array(example[0]))
    #     print(example[0], result, example[1], result == example[1])
