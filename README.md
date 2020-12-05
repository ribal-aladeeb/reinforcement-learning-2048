# Reinforcement Learning 2048

We implemented a reinforcement learning algorithm capable of training an agent in playing the sliding tile game 2048 using Deep Q-learning neural networks.

Though our max tile reached was only 1024, we found that our models were in fact learning and performed better than random and a simple up-left policy.

We utilize Deep-Q-Learning on both convolutional and linear networks using torch for training and numpy for the environment representation.

## Running an Experiment

### Model Configuration
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

### Execute Experiment
Execute the file associated with the config file.
```
cd src
python3 double_dqn_conv.py
```

The program will begin and ask you for a job name. Every N episodes it prints the merge score, max tile and number of actions takens so that you can see live results in the terminal.
```
What is the job name: job1
...
Episode: 890: 1860, 128, 144
```

### Experiment Analysis
Once the experiment has completed, an folder with the name of the job is created in the `/experiments` folder.

Inside the experiment folder there are 2 subfolders `/binary` and `/text`. Inside `/binary` there are a collection of episode information, hyperparameter information, snapshotted board histories and a snapshot of the model at the end of the experiment.

Models can be re-loaded into an experiment by using the `resumed` flag on the `Experiment` class constructor with a file path to the existing experiment folder.

Analysis of these experiments is done in the notebook `experiment_analysis.ipynb` where we conduct an analysis on any particular experiment. Plots for merge score, number of moves and max tiles are generated as well as a histogram showing the frequency distribution of max tiles within all the episodes played.

## Repository Structure
```
src/
    board.py => Board2048 game implemented using numpy.

    double_dqn_conv_sss.py => Driver that utilizes A* to load replay buffer with successful games

    double_dqn_conv.py => Driver that utilizes convolution neural networks to train an agent to play 2048

    double_dqn_dense.py => Driver that utilizes linear neural networks to train an agent to play 2048

    dqn_lib.py => Deep-Q-Networks Library which houses functionality used by the drivers to train, back-propagate, choose optional actions and sample experiences from the replay buffer.

    experiments.py => Experiments class used to create and resume running of jobs

    player.py => Player class is used to load existing policys from an experiment and simply running games using the existing policy (includes random and up-left algorithm)

    state_space_search.py => A* implementation to baseline our model against.

experiments/ -> notebooks and experiments
```
