from __future__ import annotations  # in order to allow type hints for a class referring to itself
from board import Board2048
from queue import PriorityQueue
import numpy as np
from typing import List


class SSS:

    def __init__(self, board):
        self.board = board
        self.tiebreaker = 0  # monotonically increasing id
        '''
        when adding boards to the priority queue, F(n) scores might be equal,
        the tiebreaker will be used as a second key for comparision instead of
        letter the priority queue object try to perform a '<' operation on the
        Board2048 obj (which will crash since '>' is not implemented for
        Board2048).
        '''

    def generate_GofN_score(self, board: Board2048) -> int:
        return - board._mergescore

    def generate_HofN_score(self, board: Board2048) -> int:
        zero_blocks_count: int = len(np.where(board.state == 0)[0])  # get index[0] because np.where returns a tuple of size D where D is dimension of the array argument (in this case D==2 always)
        return - zero_blocks_count

    def generate_FofN_score(self, board: Board2048) -> int:
        return self.generate_GofN_score(board) + self.generate_HofN_score(board)

    def generate_children(self, board) -> List[Board2048]:
        moves: dict = board.available_moves()
        return list(moves.values())

    def add_children_to_open_list(self, children: List[Board2048], open_list: List[Board2048]):
        assert type(children) == list, f'Children obj, is of wrong type: wanted list, got {type(children)}'

        for child in children:
            item = ((self.generate_FofN_score(child), self.tiebreaker), child)
            open_list.put(item)
            self.tiebreaker += 1

    def A_star(self):

        # Used as Priority Queue
        openlist = PriorityQueue()
        openlist.put(((0, 1), self.board))
        closed_list = []
        visited_nodes = 0

        print("A*. . . .")
        i = 1
        while openlist.qsize() > 0:
            cost: int
            next_board: Board2048
            cost, next_board = openlist.get()

            msg = ""

            if 2**i in next_board:
                # print("SUCCESS, 2048 reached")
                # next_board.show(ignore_zeros=True)
                # exit()
                i += 1
                msg = f"{2**i} already reached"

            children = self.generate_children(next_board)
            visited_nodes += len(children)
            print(f'Visited {visited_nodes} so far. Current cost is {cost}. {i}, {msg}', end='\r')

            if len(children) >= 1:
                self.add_children_to_open_list(children, openlist)  # g(n) + h(n) = f(n)

            else:
                closed_list.append(next_board)

        best = Board2048()

        print(f'Found {len(closed_list)} blocking configurations, now searching for the best one...')

        for board in closed_list:
            if self.generate_FofN_score(board) < self.generate_FofN_score(best):
                best = board

        print(f'The best board configuration reached had a merge_score of {self.generate_FofN_score(best)*-1}')
        best.show()

        return best

    def best_first(self):

        children = self.generate_children(self.board)
        dummy_board = Board2048()
        best_child = dummy_board

        print('Playing the game...')
        while len(children) > 0:


            best_child = children[0]
            if len(children) <= 1:
                self.board = best_child
                continue

            children = children[1:]
            for c in children:
                if self.generate_FofN_score(c) <= self.generate_FofN_score(best_child):  # the range of f is [-inf, -4] and smaller is better
                    best_child = c

            self.board = best_child

            children: List[Board2048] = self.generate_children(self.board)
            # print(f'{self.board}')

        print(f'No more moves available')

        print(f'Highscore reached is {np.max(self.board.state)}')
        print(f'Game score is {self.board.merge_score()}')
        self.board.show()


if __name__ == "__main__":

    sss = SSS(Board2048())
    sss.A_star()
    #sss.best_first()
