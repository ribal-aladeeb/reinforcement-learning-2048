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
        self._score = 0
        self._action_list = []

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
                vector[current] += vector[non_zero_indices[0]]
                vector[non_zero_indices[0]] = 0
            elif vector[current] == vector[non_zero_indices[0]]:
                # we know that there is a non_zero value at an index further than current
                vector[current] += vector[non_zero_indices[0]]
                self._score += vector[current]
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
        self._action_list.append("up")
        result_matrix = np.apply_along_axis(self._apply_action_to_vector, axis=1, arr=self.state.T).T
        if not np.equal(result_matrix,self.state).all():
            self.state = result_matrix
            self._populate_empty_cell()  
        return self

    def down(self):
        self._action_list.append("down")
        result_matrix = np.apply_along_axis(lambda v: self._apply_action_to_vector(self._reverse_vector(v)), axis=1, arr=self.state.T)
        result_matrix = np.apply_along_axis(self._reverse_vector, axis=1, arr=result_matrix).T
        if not np.equal(result_matrix,self.state).all():
            self.state = result_matrix
            self._populate_empty_cell()  
        return self

    def left(self):
        self._action_list.append("left")
        result_matrix = np.apply_along_axis(self._apply_action_to_vector, axis=1, arr=self.state)
        if not np.equal(result_matrix,self.state).all():
            self.state = result_matrix
            self._populate_empty_cell()  
        return self

    def right(self):
        self._action_list.append("right")
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
    
    def _merge_score(self):
        return self._score
    
    def show(self, ignore_zeros=False):
        print(f"Simple Score: {self._simple_score()}")
        print(f"Merge Score: {self._merge_score()}")
        if ignore_zeros:
            print(self.__repr__().replace("0.", "_."))
        else:
            print(self)

def basic_updown_algorithm():
    board = Board2048()
    simple_score = board._simple_score()
    while True:
        new_state = board.perform_action("up").state
        board.show(ignore_zeros=True)
        new_state = board.perform_action("left").state
        board.show(ignore_zeros=True)
        if simple_score == board._simple_score():
            new_state = board.perform_action('down').state
            board.show(ignore_zeros=True)
            new_state = board.perform_action('right').state
            if simple_score == board._simple_score():
                break
        board.show(ignore_zeros=True)
        simple_score = board._simple_score()
    board.show()
    print(board._action_list)
    return board

if __name__ == "__main__":
    # basic_updown_algorithm()
    
    board = Board2048()
    board.show(ignore_zeros=True)
    while x:=input("What is your next move: "):
        board.perform_action(x)
        board.show(ignore_zeros=True)
    print(f"Final Score: {board._merge_score()}")
    print(board)