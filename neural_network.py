from collections import deque
from board import Board2048
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import backward
import numpy as np
import logging

if not torch.cuda.is_available():
    logging.warning("No GPU: Cuda is not utilized")

batch_size = 32  # hyper paramter we need to tune
discount_factor = 0.95
# n_outputs = 4

model = nn.Sequential(
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 4),
)
replay_buffer = deque(maxlen=3000)
learning_rate = 1e-4
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)

puzzle = Board2048()


def epsilon_greedy_policy(board, epsilon=0):  # p.634
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    else:
        Q_values = model(board.flattened_state_as_tensor())
        next_action: torch.Tensor = torch.argmax(Q_values)
        return next_action


def sample_experiences(size):
    random_sample = np.random.randint(len(replay_buffer), size=size)
    batch = [replay_buffer[index] for index in random_sample]
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for experience in batch:
        board, action, reward, next_board, done = experience
        states.append(list(board.flattened_state_as_tensor()))
        actions.append(action)
        rewards.append(reward)
        next_states.append(list(next_board.flattened_state_as_tensor()))
        dones.append(int(done))
    states = torch.tensor(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states)
    dones = torch.tensor(dones)
    return states, actions, rewards, next_states, dones


def play_one_step(board, epsilon):
    action = epsilon_greedy_policy(board, epsilon)

    # take the board and perform action
    next_board = board.peek_action(action)
    reward = next_board.merge_score()
    done = (len(next_board.available_moves()) == 0)
    if done:
        next_board.show(ignore_zeros=True)
    replay_buffer.append((board, action, reward, next_board, done))
    return next_board, reward, done


def one_hot(tensor, no_outputs):
    assert tensor.max().item() + 1 <= no_outputs, "One hot encoded array size has to be bigger or equal than max scalar value"
    assert len(tensor.shape) == 1, "should be 1D"
    encoded = torch.zeros(tensor.shape[0], no_outputs)
    encoded[torch.arange(tensor.shape[0]), tensor] = 1
    return encoded



def train_step(batch_size):  # 636
    # sample some experiences
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    # compute Q-value Equation: Q_target(s,a) = r + discount_factor * max Q_theta(s', a')
    next_Q_theta_values = model(next_states)  # compute Q_theta(s',a')
    max_next_Q_theta_value = torch.max(next_Q_theta_values, axis=1).values  # compute max
    # apply discount rate to Q_theta and sum up with rewards. Note: (1-dones) is used to cancel out Q_theta when the environment is done
    target_Q_values = rewards + (1 - dones) + discount_factor * max_next_Q_theta_value
    mask = one_hot(actions, 4) # mask will be used to zero out the q values predictions from the model
    q_values_with_zeros = model(states)
    q_values = torch.sum(q_values_with_zeros * mask, axis=1, keepdim=True) # tensor of q_values for each sampled experience
    loss = loss_fn(target_Q_values, q_values)
    # loss = torch.mean(loss_fn(target_Q_values, q_values))
    loss.backward() # compute gradients
    optimizer.step() # update weights
    optimizer.zero_grad() # release gradients


def main():
    no_episodes = 300
    no_episodes_before_training = 5

    for ep in range(no_episodes):
        board = Board2048()
        done = False
        while not done:
            epsilon = max((1 - ep) / no_episodes, 0.01)
            board, reward, done = play_one_step(board, epsilon)
            if done:
                break
        print(f"Episode: {ep}: {board.merge_score()}, {np.max(board.state.flatten())}")
        if ep > no_episodes_before_training:
            train_step(batch_size)

"""Notes:
=> Merge score as a reward is ever increasing
=> Values are not normalized: 2,4,8,16
=> Hidden Layers, different types of layers etc
=> Run it on GPU
=> Identify Performance issues and tackle them
=> Start collecting metrics
*=> Implement Variants of DQN: fixed q-target, double DQN, prioritized experience replay, dueling DQN
=> Hyper parameter Tuning: DQNs are extremely succeptible to hyper parameters
=> Unit Tests for one_hot and maybe other functionalities
*=> Gradients: No magic
=> Decide whether the board can keep its underlying numpy state or if it needs to be reimplemented as tensors
"""


if __name__ == "__main__":
    main()
