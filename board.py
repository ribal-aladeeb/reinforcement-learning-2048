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

    def _reverse_vector(self, vector: np.array) -> np.array:
        return np.flip(vector)

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
        result_matrix = np.apply_along_axis(self._apply_action_to_vector, axis=1, arr=self.state.T).T
        if not np.equal(result_matrix,self.state).all():
            self.state = result_matrix
            self._populate_empty_cell()  
        return self

    def down(self):
        result_matrix = np.apply_along_axis(lambda v: self._apply_action_to_vector(self._reverse_vector(v)), axis=1, arr=self.state.T)
        result_matrix = np.apply_along_axis(self._reverse_vector, axis=1, arr=result_matrix).T
        if not np.equal(result_matrix,self.state).all():
            self.state = result_matrix
            self._populate_empty_cell()  
        return self

    def left(self):
        result_matrix = np.apply_along_axis(self._apply_action_to_vector, axis=1, arr=self.state)
        if not np.equal(result_matrix,self.state).all():
            self.state = result_matrix
            self._populate_empty_cell()  
        return self

    def right(self):
        result_matrix = np.apply_along_axis(lambda v: self._apply_action_to_vector(self._reverse_vector(v)), axis=1, arr=self.state)
        result_matrix = np.apply_along_axis(self._reverse_vector, axis=1, arr=result_matrix)
        if not np.equal(result_matrix,self.state).all():
            self.state = result_matrix
            self._populate_empty_cell()  
        return self
    
    def perform_action(self, action):
        action = action.lower()[0]
        actions = {"u":"up", "d":"down", "l":"left","r":"right"}
        if action in actions:
           move_to_perform = getattr(self, actions[action]) 
           return move_to_perform()
        raise ValueError(f"Action: {action} is invalid.")
    
    def _simple_score(self):
        return self.state.flatten().sum(axis=0)
    
    def show(self):
        print(f"Simple Score: {board._simple_score()}")
        print(self)



if __name__ == "__main__":
    board = Board2048()
    board.show()
    while x:=input("What is your next move: "):
        board.perform_action(x)
        board.show()
    print(f"Final Score: {board._simple_score()}")
    print(board)