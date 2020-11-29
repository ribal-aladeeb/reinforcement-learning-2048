'''
This will become a convolutional network
'''
from collections import deque
import pprint
from board import Board2048
import torch
import torch.nn as nn
import numpy as np
import logging
import copy
from experiments import Experiment
import os


if not torch.cuda.is_available():
    logging.warning("No GPU: Cuda is not utilized")
    device = "cpu"
else:
    device = "cuda:0"

pp = pprint.PrettyPrinter(indent=4)

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
    torch.nn.Linear(in_features=16, out_features=1024),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=1024, out_features=512),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=512, out_features=4),
).double().to(device=device)

batch_size = 5000  # number of experiences to sample
discount_factor = 0.95  # used in q-learning equation (Bellman equation)
target_model = copy.deepcopy(model)
replay_buffer = deque(maxlen=50000)  # [(state, action, reward, next_state, done),...]
learning_rate = 1e-3  # optimizer for gradient descent
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
no_episodes = 50000
no_episodes_to_reach_epsilon = 2000
min_epsilon = 0.01
no_episodes_before_training = 500
no_episodes_before_updating_target = 50
use_double_dqn = True
snapshot_game_every_n_episodes = 500

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
    'discount_factor': discount_factor,
    'model': str(model),
    'replay_buffer': replay_buffer.maxlen,
    'learning_rate': learning_rate,
    'loss_fn': str(loss_fn),
    'optimizer': str(optimizer),
    'no_episodes': no_episodes,
    'no_episodes_to_reach_epsilon': no_episodes_to_reach_epsilon,
    'no_episodes_before_training': no_episodes_before_training,
    'no_episodes_before_updating_target': no_episodes_before_updating_target,
    'min_epsilon': min_epsilon,
    'use_double_dqn': use_double_dqn
})
pp.pprint(experiment.hyperparameters)


def epsilon_greedy_policy(board, epsilon=0) -> int:  # p.634
    available_moves = board.available_moves_as_torch_unit_vector(device=device)
    done = torch.max(available_moves) == 0

    if np.random.rand() < epsilon:
        return np.random.randint(4), done, torch.zeros(size=(1,), device=device)
    else:
        state = board.normalized().state_as_4d_tensor().to(device)
        Q_values = model(state)

        available_Q_values = available_moves * Q_values

        next_action: torch.Tensor = torch.argmax(available_Q_values)
        return int(next_action), int(done), torch.max(Q_values)


def sample_experiences(batch_size):
    random_sample = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in random_sample]  # list of sampled experiences
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    states = torch.zeros((batch_size, 16), device=device).double()
    next_states = torch.zeros((batch_size, 16), device=device).double()

    for experience in batch:
        board, action, reward, next_board, done = experience
        state = board.normalized().flattened_state_as_tensor().to(device)
        next_state = next_board.normalized().flattened_state_as_tensor().to(device)
        actions.append(action)
        rewards.append(reward)
        dones.append(int(done))
        states = torch.cat((states, state), dim=0)
        next_states = torch.cat((next_states, next_state), dim=0)

    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device)
    dones = torch.tensor(dones, device=device)

    return states, actions, rewards, next_states, dones


def compute_reward(board, next_board, action, done):
    return next_board.merge_score() - board.merge_score()


def play_one_step(board, epsilon):
    action, done = epsilon_greedy_policy(board, epsilon)

    # take the board and perform action
    next_board = board.peek_action(action)

    reward = compute_reward(board, next_board, action, done)

    replay_buffer.append((board, action, reward, next_board, done))
    return next_board, action, reward, done


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

    if use_double_dqn:
        next_Q_theta_values = model(next_states)
        best_next_actions = torch.argmax(next_Q_theta_values, axis=1)
        next_mask = one_hot(best_next_actions, 4)
        next_best_q_values = torch.sum((target_model(next_states) * next_mask), axis=1)
        target_Q_values = (rewards + (1-dones) * discount_factor * next_best_q_values).double()
    else:
        # compute Q-value Equation: Q_target(s,a) = r + discount_factor * max Q_theta(s', a')
        next_Q_theta_values = target_model(next_states)  # compute Q_theta(s',a')

        max_next_Q_theta_value = torch.max(next_Q_theta_values, axis=1).values  # compute max

        # apply discount factor to Q_theta and sum up with rewards. Note: (1-dones) is used to cancel out Q_theta when the environment is done
        target_Q_values = rewards + ((1 - dones) * discount_factor * max_next_Q_theta_value)
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
    optimizer.zero_grad()  # release gradients
    optimizer.step()  # update weights
    return loss


def main():
    try:
        for ep in range(no_episodes):
            board = Board2048()
            done = False
            board_history = []
            rewards = []
            q_values = []
            while not done:
                epsilon = max((no_episodes_to_reach_epsilon - ep) / no_episodes_to_reach_epsilon, min_epsilon)  # value to determine how greedy the policy should be for that step
                new_board, action, reward, done, max_q_value = play_one_step(board, epsilon)
                board_history.append((board.state, ['u', 'd', 'l', 'r'][int(action)], reward))
                rewards.append(reward)
                q_values.append(max_q_value)
                board = new_board
            mean_of_rewards = np.mean(np.array(rewards))
            mean_of_q_values = torch.mean(torch.tensor(q_values))
            experiment.add_episode(board, epsilon, ep, mean_of_rewards, mean_of_q_values)
            if ep % snapshot_game_every_n_episodes == 0:
                experiment.snapshot_game(board_history, ep)
            if ep % 50 == 0:
                print(f"Episode: {ep}: {board.merge_score()}, {np.max(board.state.flatten())}, {len(board._action_history)}")
            if ep > no_episodes_before_training:
                train_step(batch_size)
            if ep % no_episodes_before_updating_target == 0:
                target_model.load_state_dict(copy.deepcopy(model.state_dict()))
            if ep % 1000 == 0:
                experiment.save()
                print("Saved game")

        experiment.save()

    except KeyboardInterrupt as e:
        print(e)
        print(f'\nKeyboard interrut caught. Saving current experiment in {experiment.folder}')
        experiment.save()

    except Error or Exception as e:
        experiment.save()
        print(f'\nSaving current experiment in {experiment.folder}\n')
        raise e


if __name__ == "__main__":
    main()
