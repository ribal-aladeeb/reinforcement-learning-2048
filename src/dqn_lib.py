from typing import Callable
import numpy as np
import torch
from board import Board2048
from collections import deque
import copy

def board_as_4d_tensor(board: Board2048, device: str) -> torch.tensor:
    return board.log_scale().state_as_4d_tensor().to(device)


def board_as_flattened_tensor(board: Board2048, device: str) -> torch.tensor:
    return board.log_scale().flattened_state_as_tensor().to(device)


def epsilon_greedy_policy(board, epsilon, model: torch.nn.Sequential, device, board_to_tensor_function: Callable = board_as_4d_tensor) -> int:  # p.634
    available_moves = board.available_moves_as_torch_unit_vector(device=device)
    done = torch.max(available_moves) == 0

    if np.random.rand() < epsilon:
        return np.random.randint(4), done, torch.zeros(size=(1,), device=device)
    else:
        state = board_to_tensor_function(board, device)
        Q_values = model(state)
        Q_values_normal = Q_values - \
            torch.min(Q_values) * torch.max(Q_values) - torch.min(Q_values)

        available_Q_values = available_moves * Q_values_normal
        next_action: torch.Tensor = torch.argmax(available_Q_values)
        return int(next_action), int(done), torch.max(Q_values)


def extract_samples_conv(batch_size, batch, device, actions, rewards, dones, board_to_tensor_function):
    states = torch.tensor([], device=device).double()
    next_states = torch.tensor([], device=device).double()

    for experience in batch:
        board, action, reward, next_board, done = experience
        state = board_to_tensor_function(board, device)
        next_state = board_to_tensor_function(next_board, device)
        actions.append(action)
        rewards.append(reward)
        dones.append(int(done))
        states = torch.cat((states, state), dim=0)
        next_states = torch.cat((next_states, next_state), dim=0)
    return states, actions, rewards, next_states, dones


def extract_samples_dense(batch_size, batch, device, actions, rewards, dones, board_to_tensor_function):
    states = torch.zeros((batch_size, 16), device=device).double()
    next_states = torch.zeros((batch_size, 16), device=device).double()

    i = 0
    for experience in batch:
        board, action, reward, next_board, done = experience
        state = board_to_tensor_function(board, device)
        next_state = board_to_tensor_function(next_board, device)
        actions.append(action)
        rewards.append(reward)
        dones.append(int(done))
        states[i] = state
        next_states[i] = next_state
        i += 1
    return states, actions, rewards, next_states, dones


def sample_experiences(batch_size: int, replay_buffer: deque, device: str, board_to_tensor_function: Callable, extract_sample_function: Callable):
    random_sample = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index]
             for index in random_sample]  # list of sampled experiences
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    states, actions, rewards, next_states, dones = extract_sample_function(
        batch_size, batch, device, actions, rewards, dones, board_to_tensor_function)

    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device)
    dones = torch.tensor(dones, device=device)

    return states, actions, rewards, next_states, dones


def reward_func_merge_score(board: Board2048, next_board: Board2048, action: int, done: int) -> int:
    return next_board.merge_score() - board.merge_score()


def play_one_step(
        board: Board2048,
        epsilon: float,
        model: torch.nn.Sequential,
        replay_buffer: deque,
        device: str,
        reward_function: Callable = reward_func_merge_score, board_to_tensor_function: Callable = board_as_4d_tensor):

    action, done, max_q_value = epsilon_greedy_policy(
        board, epsilon=epsilon, model=model, device=device, board_to_tensor_function=board_to_tensor_function)

    next_board = board.peek_action(action)

    reward = reward_function(board, next_board, action, done)

    replay_buffer.append((board, action, reward, next_board, done))
    return next_board, action, reward, done, max_q_value


def one_hot(tensor: torch.tensor, no_outputs: int, device: str):
    assert tensor.max().item() + \
        1 <= no_outputs, "One hot encoded array size has to be bigger or equal than max scalar value"
    assert len(tensor.shape) == 1, "should be 1D"
    encoded = torch.zeros(tensor.shape[0], no_outputs, device=device)
    encoded[torch.arange(tensor.shape[0]), tensor] = 1
    return encoded


def train_step(batch_size: int, discount_factor: int, model: torch.nn.Sequential, target_model: torch.nn.Sequential, replay_buffer: deque, loss_fn: Callable, optimizer: object, device: str, use_double_dqn: bool = True, board_to_tensor_function: Callable = board_as_4d_tensor, extract_samples_function: Callable = extract_samples_conv):  # 636
    # sample some experiences
    experiences = sample_experiences(batch_size, replay_buffer, device, board_to_tensor_function,
                                     extract_samples_function)  # (state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = experiences

    if use_double_dqn:
        next_Q_theta_values = model(next_states)
        best_next_actions = torch.argmax(next_Q_theta_values, axis=1)
        next_mask = one_hot(best_next_actions, 4, device=device)
        next_best_q_values = torch.sum(
            (target_model(next_states) * next_mask), axis=1)
        target_Q_values = (rewards + (1-dones) *
                           discount_factor * next_best_q_values).double()
    else:
        # compute Q-value Equation: Q_target(s,a) = r + discount_factor * max Q_theta(s', a')
        next_Q_theta_values = target_model(
            next_states)  # compute Q_theta(s',a')

        max_next_Q_theta_value = torch.max(
            next_Q_theta_values, axis=1).values  # compute max

        # apply discount factor to Q_theta and sum up with rewards. Note: (1-dones) is used to cancel out Q_theta when the environment is done
        target_Q_values = rewards + \
            ((1 - dones) * discount_factor * max_next_Q_theta_value)
        target_Q_values.double()

    # create a mask
    # mask will be used to zero out the q values predictions from the model
    mask = one_hot(actions, 4, device=device)
    q_values_with_zeros = model(states)
    # tensor of q_values for each sampled experience
    q_values = torch.sum(q_values_with_zeros * mask, axis=1,
                         keepdim=True, dtype=torch.double)

    # q_values shape was [N,1] transpose q_values and get first element so shape is [N] to match target_q_values
    q_values = torch.transpose(q_values, 0, 1)[0]

    # compute loss
    loss = loss_fn(q_values, target_Q_values)

    # back propogate
    loss.backward()  # compute gradients
    optimizer.zero_grad()  # release gradients
    optimizer.step()  # update weights
    return loss


def training_loop(replay_buffer_length, no_episodes, no_episodes_to_reach_epsilon, no_episodes_to_fill_up_existing_model_replay_buffer, min_epsilon, model, reward_function, board_to_tensor_function, device, experiment, snapshot_game_every_n_episodes, no_episodes_before_training, batch_size, discount_factor, target_model, loss_fn, optimizer, use_double_dqn, no_episodes_before_updating_target, extract_samples_function):
    try:
        replay_buffer = deque(maxlen=replay_buffer_length)
        for ep in range(no_episodes):
            print(ep)
            board = Board2048()
            done = False
            board_history = []
            rewards = []
            q_values = []
            epsilon = None
            while not done:
                # value to determine how greedy the policy should be for that step
                epsilon = max((no_episodes_to_reach_epsilon - ep) /
                              no_episodes_to_reach_epsilon, min_epsilon)

                if ep < no_episodes_to_fill_up_existing_model_replay_buffer:
                    epsilon = 0

                new_board, action, reward, done, max_q_value = play_one_step(
                    board,
                    epsilon,
                    model,
                    replay_buffer,
                    reward_function=reward_function,
                    board_to_tensor_function=board_to_tensor_function,
                    device=device
                )
                board_history.append(
                    (board.state, ['u', 'd', 'l', 'r'][int(action)], reward))
                rewards.append(reward)
                q_values.append(float(max_q_value))
                board = new_board
            mean_of_rewards = np.mean(np.array(rewards))
            mean_of_q_values = np.mean(np.array(q_values))
            experiment.add_episode(
                board, epsilon, ep, mean_of_rewards, mean_of_q_values)
            if ep % snapshot_game_every_n_episodes == 0:
                experiment.snapshot_game(board_history, ep)
            if ep % 10 == 0:
                print(
                    f"Episode: {ep}: {board.merge_score()}, {np.max(board.state.flatten())}, {len(board._action_history)}")
            if ep > no_episodes_before_training:
                train_step(
                    batch_size,
                    discount_factor,
                    model,
                    target_model,
                    replay_buffer,
                    loss_fn,
                    optimizer,
                    device=device,
                    use_double_dqn=use_double_dqn,
                    board_to_tensor_function=board_to_tensor_function,
                    extract_samples_function=extract_samples_function
                )
            if ep % no_episodes_before_updating_target == 0 and ep >= no_episodes_to_fill_up_existing_model_replay_buffer:
                target_model.load_state_dict(copy.deepcopy(model.state_dict()))
            if ep % 1000 == 0:
                experiment.save()
                print("Saved game")

        experiment.save()

    except KeyboardInterrupt as e:
        print(e)
        print(
            f'\nKeyboard interrut caught. Saving current experiment in {experiment.folder}')
        experiment.save()

    except Exception as e:
        experiment.save()
        print(f'\nSaving current experiment in {experiment.folder}\n')
        raise e
