from __future__ import annotations  # in order to allow type hints for a class referring to itself
import numpy as np
from typing import List
import random

from numpy.core.fromnumeric import nonzero


class Board2048:

    def __init__(self, k: int = 4, populate_empty_cells=True):
        self.state: np.array = np.zeros(shape=(k, k),dtype=int)
        self._empty_spot_numbers: List[int] = [2, 4]
        self._mergescore = 0
        self._action_history = []
        self.k = k
        self.populate_empty_cells = populate_empty_cells
        
        if populate_empty_cells:
            self._populate_empty_cell()
            self._populate_empty_cell()
    
    def clone(self) -> Board2048:
        board = Board2048(k=self.k, populate_empty_cells=self.populate_empty_cells)
        board.state: np.array = np.copy(self.state)
        board._mergescore: int = self._mergescore
        board._action_history: List[str] = self._action_history.copy()
        return board

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
                self._mergescore += vector[current]
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
    
    def available_moves(self) -> Set[str]:
        moves = ['up', 'down', 'left','right']
        available = set()

        for move in moves:
            if not (self.state == self.peek_action(move)).all():
                available.add(move)
        return available
    
    def up(self) -> Board2048:
        board = self.clone()
        board._action_history.append('up')
        result_matrix = np.apply_along_axis(board._apply_action_to_vector, axis=1, arr=board.state.T).T
        if not np.equal(result_matrix, board.state).all():
            board.state = result_matrix
            board._populate_empty_cell()  
        return board

    def down(self) -> Board2048:
        board = self.clone()
        board._action_history.append('down')
        result_matrix = np.apply_along_axis(lambda v: board._apply_action_to_vector(board._reverse_vector(v)), axis=1, arr=board.state.T)
        result_matrix = np.apply_along_axis(board._reverse_vector, axis=1, arr=result_matrix).T
        if not np.equal(result_matrix, board.state).all():
            board.state = result_matrix
            board._populate_empty_cell()  
        return board


    def left(self) -> Board2048:
        board = self.clone()
        board._action_history.append('left')
        result_matrix = np.apply_along_axis(board._apply_action_to_vector, axis=1, arr=board.state)
        if not np.equal(result_matrix, board.state).all():
            board.state = result_matrix
            board._populate_empty_cell()  
        return board

    def right(self) -> Board2048:
        board = self.clone()
        board._action_history.append('right')
        result_matrix = np.apply_along_axis(lambda v: board._apply_action_to_vector(board._reverse_vector(v)), axis=1, arr=board.state)
        result_matrix = np.apply_along_axis(board._reverse_vector, axis=1, arr=result_matrix)
        if not np.equal(result_matrix,board.state).all():
            board.state = result_matrix
            board._populate_empty_cell()  
        return board
    
    def peek_action(self, action: str) -> Board2048:
        '''
        Returns the would-be state of the board if you were to save the state after performing the <action> argument
        '''
        action = action.lower()[0]
        actions = {"u":"up", "d":"down", "l":"left","r":"right"}
        
        if action in actions:
            move_to_perform = getattr(self, actions[action])
            new_board: Board2048 = move_to_perform()
            return new_board
        raise ValueError(f"Action: {action} is invalid.")
    
    def simple_score(self):
        return self.state.flatten().sum(axis=0)
    
    def merge_score(self):
        return self._mergescore
    
    def show(self, ignore_zeros=False):
        print(f"Simple Score: {self.simple_score()}")
        print(f"Merge Score: {self.merge_score()}")
        if ignore_zeros:
            print(self.__repr__().replace("0", "_"))
        else:
            print(self)

def basic_updown_algorithm(k=4):
    board = Board2048(k=k)
    simple_score = board.simple_score()
    while True:
        board = board.peek_action("up")
        board.show(ignore_zeros=True)
        board = board.peek_action("left")
        board.show(ignore_zeros=True)
        if simple_score == board.simple_score():
            board = board.peek_action('down')
            board.show(ignore_zeros=True)
            board = board.peek_action('right')
            if simple_score == board.simple_score():
                break
        board.show(ignore_zeros=True)
        simple_score = board.simple_score()
    board.show()
    print(board._action_history)
    return board

if __name__ == "__main__":
    merge_scores = []
    maxes = []
    simple_scores = []
    for i in range(100):
        board = basic_updown_algorithm(k=4)
        board.show(ignore_zeros=True)
        maxes.append(np.log2(np.max(board.state))) 
        merge_scores.append(board.merge_score())
        simple_scores.append(board.simple_score())

    print(np.mean(merge_scores))
    print(np.mean(maxes), f"{2**np.mean(maxes)}")
    print(np.mean(simple_scores))
    
    
    board = Board2048()
    board.show(ignore_zeros=True)
    while x:=input("What is your next move: "):
        board = board.peek_action(x)
        board.show(ignore_zeros=True)
    print(f"Final Score: {board._merge_score()}")
    print(board)