from board import Board2048
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

if not torch.cuda.is_available():
    logging.warning("No GPU: Cuda is not utilized")

model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
)

print(model)
def epsilon_greedy_policy(board, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random



# transition_probs = {}
# rewards  = {}

# def rewards(b, a, n):
#     return None

# def exploration_policy(board) -> str:
#     available_moves = board.available_moves()
#     return np.random.choice(list(available_moves.keys()))

# def step(board, action):
#     new_board = board.peek_action(action)
#     reward = rewards(board, action, new_board)
#     return new_board, reward

# alpha0 = 0.05
# decay = 0.005
# gamma = 0.9

# board = Board2048()
# for i in range(1000):
#     action = exploration_policy(board)
#     new_board, reward = step(board, action)
#     next_value = np.max(q[new_board])
#     alpha = alpha0 / (1 + i*decay)
#     q[board, action] *= 1 - alpha
#     q[board, action] += alpha * (reward + gamma * next_value)
#     board = new_board
