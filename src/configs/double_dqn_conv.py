from collections import deque
from torch import nn
import torch
import copy
from device import device

'''
When using ConvNets, the following formula is useful for knowing a convolution's feature map output shape

SHAPE = ((w + 2p) - k + s)/s
where
w: width of input feature (assuming width == height)
p: padding value (a padding of 1 means adds 2 pixels to each axis)
k: kernel size (assuming kernel is square)
s: stride

the resulting output feature map (assuming they are all squares) will be SHAPE x SHAPE
'''
model = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),  # each feature map is 2x2 with 128 features
    nn.Linear(2*2*64, 64),
    nn.ReLU(),
    nn.Linear(64, 4)
).double().to(device=device)

'''
Hyperparameters are defined here
'''
batch_size = 5000  # Number of experiences to sample when training
discount_factor = 0.80  # Used in q-learning equation (Bellman equation) to determine how much of the future rewards to apply to the current Q-values
target_model = copy.deepcopy(model) # A duplicate model, acting as the target model, it is updated at set intervals to compute loss
replay_buffer_length = 15000  # contains experiences (or episodes) [(state, action, reward, next_state, done),...]
learning_rate = 1e-2  # In what increments the optimizer is using when doing gradient descent
loss_fn = nn.MSELoss(reduction='sum') # Loss function used to compute the loss values
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Variant of SGD
no_episodes = 30000  # Number of Episodes to run
no_episodes_to_reach_epsilon = 1000 # Number of episodes before minimum epsilon is always used
min_epsilon = 0.01  # Minimum epsilon, epsilon is a probably of doing a random action instead of using the model's predicted best action
no_episodes_before_training = 700  # Number of episodes to wait before waiting
no_episodes_before_updating_target = 100 # Number of episodes before target model is updated and clones the online model
no_episodes_to_fill_up_existing_model_replay_buffer = 0 # Set to 0 if you want to not fill up the replay buffer.
use_double_dqn = True  # Use the Double DQN Variant
snapshot_game_every_n_episodes = 500 # Snapshot the game, every n episodes so that playback of games is possible.

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
