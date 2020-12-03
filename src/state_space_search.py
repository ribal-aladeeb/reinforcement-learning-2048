from __future__ import annotations  # in order to allow type hints for a class referring to itself
from board import Board2048
from queue import PriorityQueue
import numpy as np
from typing import List
import math
from collections import deque
from dqn_lib import reward_func_merge_score

class Node():
    def __init__(self, b: Board2048, parent=None, cost=0, move=None, reward = 0):
        self.parent = parent
        self.board = b
        self.cost = cost
        self.move = move
        self.reward = reward

    def is_root(self):
        return self.parent == None

    def HofN(self, goal_tile=2048):
        return self.board.merge_score() // 2

    # def HofN(self, goal_tile=2048):
    #     'computes heuristic func used for informed search'
    #     return goal_tile - self.board.get_max_tile()

    # def HofN_2(self, goal_tile=2048):
    #     'computes heuristic func used for informed search'
    #     return (goal_tile - self.board.get_max_tile()) // len(np.where(self.board.state == 0)[0])

    # def HofN_3(self, goal_tile=2048):
    #     'computes heuristic func used for informed search'
    #     if (goal_tile - self.board.get_max_tile()) == 0:
    #         return 0
    #     return (self.board._mergescore) // len(np.where(self.board.state == 0)[0])  # may not be admissible. still lets try it out and see which ones perform better and make it admissible afterwrds

    def FofN(self, goal_tile=2048):
        # return (self.cost - self.HofN(goal_tile))
        return -self.board.merge_score() // 2

    def generate_children(self) -> List[Board2048]:
        moves: dict = self.board.available_moves()
        return moves

def A_star(b: Board2048):

    current_node = Node(b)

    current_max_tile = np.max(current_node.board.state)

    if current_max_tile >= 2048:
        goal_tile = current_max_tile * 2
    else:
        goal_tile = 2048

    openlist = PriorityQueue()
    visited_nodes = 1
    openlist.put((0, 0, current_node))
    closed_list = {}
    expanded_nodes = 0

    while not openlist.empty():

        #-- Pop() from open list --#
        _, _, current_node = openlist.get()
        visited_nodes += 1

        # if visited_nodes % 1000 == 0:
        #     print(f'visited {visited_nodes}, goal tile: {goal_tile}, openlist.size: {openlist.qsize()}, board:\n{current_node.board}',)

        #-- Goal state reached! --#
        if goal_tile in current_node.board:
            return {'success': True,
                    'current_node': current_node,
                    'path_length': current_node.cost,
                    'visited_nodes': visited_nodes,
                    'expanded_nodes': expanded_nodes}

        #-- Do not add node in closed list if the state has already been encountered with a less g(n) --#
        hashed = tuple(current_node.board.state.flatten())
        if hashed in closed_list:
            if closed_list[hashed].FofN() > current_node.FofN():
                continue

        #-- Add node in closed list --#
        closed_list[hashed] = current_node

        #-- Add children in open list --#
        moves: dict = current_node.generate_children()
        for m in moves:
            move: str = m
            next_board: Board2048 = moves[m]
            child_node = Node(b=next_board, parent=current_node, cost=current_node.cost+1, move=move)
            expanded_nodes += 1
            openlist.put((child_node.FofN(), expanded_nodes, child_node))

    return {'success': True,
            'current_node': current_node,
            'path_length': np.inf,
            'visited_nodes': visited_nodes,
            'expanded_nodes': expanded_nodes}

def generate_replay_buffer_using_A_star(batch_size, maxlen):
    from tqdm import tqdm

    #-- Set a max replay buffer size --#
    replay_buffer = deque(maxlen=maxlen)

    #-- Mapping of action to int --#
    move = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

    #-- Run a_star for each batch --#
    for _ in tqdm(range(batch_size)):
        b = Board2048()
        current_node = A_star(b)['current_node']
        print(current_node, current_node.parent)

        #-- Trace back to parent nodes and register all moves --#
        while current_node.parent != None:
            current = current_node
            parent = current_node.parent
            done = int(current.is_root())
            action = move[current.move]
            reward = reward_func_merge_score(current.board, parent.board, action, done)

            #-- Append games to replay_buffer --#
            replay_buffer.append((current.board, action, reward, current.board, done))

            current_node = current_node.parent
    return replay_buffer

if __name__ == "__main__":
    x = generate_replay_buffer_using_A_star(batch_size=1, maxlen=1000)
    print("First state")
    print(x[-1])
    print("Last state")
    print(x[0])
    # b = Board2048()
    # print('Searching for solution in initial board:')
    # print(b)
    # print()
    # result = A_star(b)

    # print(f'Solution found:\n{result["current_node"].board}')
    # print(result)
