from collections import deque


from board import Board2048
from experiments import Experiment
import torch
import torch.nn as nn
import numpy as np
import logging


if not torch.cuda.is_available():
    logging.warning("No GPU: Cuda is not utilized")
    device = "cpu"
else:
    device = "cuda:0"

batch_size = 32  # number of experiences to sample
discount_factor = 0.95  # used in qlearning equation (Bellman equation)
model = nn.Sequential(
    nn.Linear(16, 16),  # takes 16 inputs
    nn.ReLU(),
    nn.Linear(16, 4),  # outputs 4 actions
).double()
model.to(device)

replay_buffer = deque(maxlen=3000)  # [(state, action, reward, next_state, done),...]
learning_rate = 1e-4  # optimizer for gradient descent
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)


def epsilon_greedy_policy(board, epsilon=0) -> int:  # p.634
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    else:
        Q_values = model(board.flattened_state_as_tensor().to(device))
        next_action: torch.Tensor = torch.argmax(Q_values)
        return int(next_action)


def sample_experiences(batch_size):
    random_sample = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in random_sample]  # list of sampled experiences
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for experience in batch:
        board, action, reward, next_board, done = experience
        states.append(list(board.flattened_state_as_tensor().to(device)))
        actions.append(action)
        rewards.append(reward)
        next_states.append(list(next_board.flattened_state_as_tensor().to(device)))
        dones.append(int(done))

    states = torch.tensor(states, device=device)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device)
    next_states = torch.tensor(next_states, device=device)
    dones = torch.tensor(dones, device=device)
    return states, actions, rewards, next_states, dones


def compute_reward(board, next_board):
    '''
    Here a reward is refine as the number of merges. If the max value of the
    board has increase, add np.log2(next_max) * 0.1 to the reward.
    '''
    previous_max = np.max(board.state)
    next_max = np.max(next_board.state)
    no_empty_previous = board.number_of_empty_cells()
    no_empty_next = next_board.number_of_empty_cells()
    number_of_merges = (no_empty_next - no_empty_previous)+1  # number of merges done
    reward = number_of_merges
    if next_max > previous_max:
        reward += np.log2(next_max)*0.1
    return reward


def play_one_step(board, epsilon):
    action = epsilon_greedy_policy(board, epsilon)

    # take the board and perform action
    next_board = board.peek_action(action)

    reward = compute_reward(board, next_board)
    # reward = next_board.merge_score()  # define a better reward than merge
    done = (len(next_board.available_moves()) == 0)  # indicates whether you have any moves you can do
    if done:
        next_board.show(ignore_zeros=True)
    replay_buffer.append((board, action, reward, next_board, done))
    return next_board, reward, done


def one_hot(tensor, no_outputs):
    assert tensor.max().item() + 1 <= no_outputs, "One hot encoded array size has to be bigger or equal than max scalar value"
    assert len(tensor.shape) == 1, "should be 1D"
    encoded = torch.zeros(tensor.shape[0], no_outputs, device=device)
    encoded[torch.arange(tensor.shape[0]), tensor] = 1
    return encoded


def train_step(batch_size):  # 636
    # sample some experiences
    experiences = sample_experiences(batch_size)  # (state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = experiences

    # compute Q-value Equation: Q_target(s,a) = r + discount_factor * max Q_theta(s', a')
    next_Q_theta_values = model(next_states)  # compute Q_theta(s',a')
    max_next_Q_theta_value = torch.max(next_Q_theta_values, axis=1).values  # compute max

    # apply discount factor to Q_theta and sum up with rewards. Note: (1-dones) is used to cancel out Q_theta when the environment is done
    target_Q_values = rewards + (1 - dones) * discount_factor * max_next_Q_theta_value
    target_Q_values.double()

    # create a mask
    mask = one_hot(actions, 4)  # mask will be used to zero out the q values predictions from the model
    q_values_with_zeros = model(states)
    q_values = torch.sum(q_values_with_zeros * mask, axis=1, keepdim=True, dtype=torch.double)  # tensor of q_values for each sampled experience

    # q_values shape was [N,1] transpose q_values and get first element so shape is [N] to match target_q_values
    q_values = torch.transpose(q_values, 0, 1)[0]

    # compute loss
    loss = loss_fn(q_values, target_Q_values)

    # back propogate
    loss.backward()  # compute gradients
    optimizer.step()  # update weights
    optimizer.zero_grad()  # release gradients


def main():
    try:
        experiment = Experiment()

        no_episodes = 500
        no_episodes_before_training = 50

        experiment.add_hyperparameter({
            'no_episodes': no_episodes,
            'no_episodes_before_training': no_episodes_before_training,
        })

        for ep in range(no_episodes):
            board = Board2048()
            done = False
            while not done:
                epsilon = max((1 - ep) / no_episodes, 0.3)  # value to determine how greedy the policy should be for that step
                board, reward, done = play_one_step(board, epsilon)

            experiment.add_episode(board, epsilon, ep, reward)
            print(f"Episode: {ep}: {board.merge_score()}, {np.max(board.state.flatten())}")
            if ep > no_episodes_before_training:
                train_step(batch_size)

        experiment.save()

    except KeyboardInterrupt or Exception as e:
        try:
            print('Keyboard interupt caught, saving current experiment')
            experiment.save()
        except Exception as e:
            print(e)
            print("Can save your experiment to disk.")


"""Notes:
* => Merge score as a reward is ever increasing
=> Values are not normalized: 2,4,8,16
* => Hidden Layers, different types of layers etc
DONE => Run it on GPU
=> Identify Performance issues and tackle them
=> Start collecting metrics
*=> Implement Variants of DQN: fixed q-target, double DQN, prioritized experience replay, dueling DQN
=> Unit Tests for one_hot and maybe other functionalities
=> Gradients: No magic
x => Decide whether the board can keep its underlying numpy state or if it needs to be reimplemented as tensors
"""
if __name__ == "__main__":
    main()
