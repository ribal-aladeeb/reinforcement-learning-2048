# Reinforcement Learning 2048

We build a reinforcement learning agent using Deep Q-learning neural networks.

## Running an Experiment

#### Model Configuration
```
cd ./src/configs/
```

Currently there are 3 config files set up, each can be modified seperately. An example config looks like:
```python
## example config similar to src/double_dqn_conv.py
from collections import deque
from torch import nn
import torch
import copy
from device import device

model = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),  # each feature map is 2x2 with 128 features
    nn.Linear(2*2*64, 64),
    nn.Relu(),
    nn.Linear(64, 4)
).double().to(device=device)

batch_size = 5000  # Number of experiences to sample when training
discount_factor = 0.95  # Used in q-learning equation (Bellman equation) to determine how much of the future rewards to apply to the current Q-values
target_model = copy.deepcopy(model) # A duplicate model, acting as the target model, it is updated at set intervals to compute loss
replay_buffer_length = 15000  # contains experiences (or episodes) [(state, action, reward, next_state, done),...]
learning_rate = 1e-4  # In what increments the optimizer is using when doing gradient descent
loss_fn = nn.MSELoss(reduction='sum') # Loss function used to compute the loss values
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Variant of SGD
no_episodes = 30000  # Number of Episodes to run
no_episodes_to_reach_epsilon = 500 # Number of episodes before minimum epsilon is always used
min_epsilon = 0.01  # Minimum epsilon, epsilon is a probably of doing a random action instead of using the model's predicted best action
no_episodes_before_training = 500  # Number of episodes to wait before waiting
no_episodes_before_updating_target = 100 # Number of episodes before target model is updated and clones the online model
no_episodes_to_fill_up_existing_model_replay_buffer = 50 # Set to 0 if you want to not fill up the replay buffer.
use_double_dqn = True  # Use the Double DQN Variant
snapshot_game_every_n_episodes = 500 # Snapshot the game, every n episodes so that playback of games is possible.
```

#### Execute Experiment
Execute the file associated with the config file.
```
cd src
python3 double_dqn_conv.py
```

```
What is the job name:
Episode: 890: 1860, 128, 144
```
