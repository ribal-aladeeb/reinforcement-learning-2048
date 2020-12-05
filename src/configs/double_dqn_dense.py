from collections import deque
from torch import nn
import torch
import copy
from device import device

model = nn.Sequential(
    torch.nn.Linear(in_features=16, out_features=512),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=512, out_features=512),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=512, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256, out_features=4),
).double().to(device=device)

batch_size = 5000  # number of experiences to sample
discount_factor = 0.80  # used in q-learning equation (Bellman equation)
target_model = copy.deepcopy(model)
replay_buffer_length = 100000  # [(state, action, reward, next_state, done),...]
learning_rate = 1e-2  # optimizer for gradient descent
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
no_episodes = 50000
no_episodes_to_reach_epsilon = 1000
min_epsilon = 0.01  # meaning at a certain point, there is no random moves
no_episodes_before_training = 700
no_episodes_before_updating_target = 100
use_double_dqn = True
snapshot_game_every_n_episodes = 500
no_episodes_to_fill_up_existing_model_replay_buffer = 0

HYPERPARAMS = {
    'batch_size': batch_size,
    'discount_factor': discount_factor,
    'model': str(model),
    'replay_buffer_length': replay_buffer_length,
    'learning_rate': learning_rate,
    'loss_fn': str(loss_fn),
    'optimizer': str(optimizer),
    'no_episodes': no_episodes,
    'no_episodes_to_reach_epsilon': no_episodes_to_reach_epsilon,
    'no_episodes_before_training': no_episodes_before_training,
    'no_episodes_before_updating_target': no_episodes_before_updating_target,
    'no_episodes_to_fill_up_existing_model_replay_buffer': no_episodes_to_fill_up_existing_model_replay_buffer,
    'min_epsilon': min_epsilon,
    'use_double_dqn': use_double_dqn,
    'snapshot_game_every_n_episodes': snapshot_game_every_n_episodes
}
