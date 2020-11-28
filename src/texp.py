from collections import deque
from copy import Error
from logging import error
import pprint
from torch import tensor
from torch.nn.modules.container import Sequential
from board import Board2048
import torch
import torch.nn as nn
import numpy as np
import logging
import copy
import os
from experiments import Experiment

if not torch.cuda.is_available():
    logging.warning("No GPU: Cuda is not utilized")
    device = "cpu"
else:
    device = "cuda:0"

pp = pprint.PrettyPrinter(indent=4)

model = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),  # each feature map is 2x2 with 128 features
    nn.Linear(2*2*64, 64),
    nn.Linear(64, 4)
).double().to(device=device)

batch_size = 5000  # number of experiences to sample
discount_factor = 0.80  # used in q-learning equation (Bellman equation)
target_model = copy.deepcopy(model)
replay_buffer = deque(maxlen=15000)  # [(state, action, reward, next_state, done),...]
learning_rate = 1e-4  # optimizer for gradient descent
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
no_episodes = 50000
no_episodes_to_reach_epsilon = 10000
min_epsilon = 0.01
no_episodes_before_training = 2000
no_episodes_before_updating_target = 100
use_double_dqn = True

job_name = input("What is the job name: ")

if job_name:
    experiment = Experiment(
        python_file_name = os.path.basename(__file__),
        folder_name=job_name,
        model=model,
        loss=loss_fn,
        optimizer=optimizer)
else:
    experiment = Experiment(
        python_file_name = os.path.basename(__file__),
        model=model,
        loss=loss_fn,
        optimizer=optimizer)

experiment.add_hyperparameter({
    'batch_size': batch_size,
    'discount_factor' :discount_factor,
    'model': str(model),
    'replay_buffer': replay_buffer.maxlen,
    'learning_rate' : learning_rate,
    'loss_fn': str(loss_fn),
    'optimizer': str(optimizer),
    'no_episodes': no_episodes,
    'no_episodes_to_reach_epsilon': no_episodes_to_reach_epsilon,
    'no_episodes_before_training': no_episodes_before_training,
    'no_episodes_before_updating_target': no_episodes_before_updating_target,
    'min_epsilon': min_epsilon,
    'use_double_dqn': use_double_dqn
})

board = Board2048()
model(board.normalized().state_as_4d_tensor())
experiment.add_episode(board, 0.01, 0, 80085)
experiment.save()


import time
time.sleep(2)
print("Sleeping...")

new_experiment = Experiment(
        python_file_name = os.path.basename(__file__),
        folder_name=job_name,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        resumed=True)


board = Board2048()
model(board.normalized().state_as_4d_tensor())
new_experiment.add_episode(board, 0.01, 0, 80085)
new_experiment.save()


print(experiment.model.parameters())
print(new_experiment.model.parameters())
