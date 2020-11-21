from neural_network import *
from board import *

target_model = copy.deepcopy(model)

def train_step_q_learning(batch_size):  # 636
    # sample some experiences
    experiences = sample_experiences(batch_size) # (state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = experiences

    # compute Q-value Equation: Q_target(s,a) = r + discount_factor * max Q_theta(s', a')
    next_Q_theta_values = target_model(next_states)  # compute Q_theta(s',a')
    max_next_Q_theta_value = torch.max(next_Q_theta_values, axis=1).values  # compute max

    # apply discount factor to Q_theta and sum up with rewards. Note: (1-dones) is used to cancel out Q_theta when the environment is done
    target_Q_values = rewards + (1 - dones) * discount_factor * max_next_Q_theta_value
    target_Q_values.double()

    # create a mask
    mask = one_hot(actions, 4) # mask will be used to zero out the q values predictions from the model
    q_values_with_zeros = model(states)
    q_values = torch.sum(q_values_with_zeros * mask, axis=1, keepdim=True,dtype=torch.double) # tensor of q_values for each sampled experience

    # q_values shape was [N,1] transpose q_values and get first element so shape is [N] to match target_q_values
    q_values = torch.transpose(q_values, 0, 1)[0]

    # compute loss
    loss = loss_fn(q_values, target_Q_values)

    # back propogate
    loss.backward() # compute gradients
    optimizer.step() # update weights
    optimizer.zero_grad() # release gradients


def main_variation1():
    no_episodes = 500
    no_episodes_before_training = 50
    no_episodes_before_update = 50
    for ep in range(no_episodes):
        board = Board2048()
        done = False
        while not done:
            epsilon = max((1 - ep) / no_episodes, 0.3)  #  value to determine how greedy the policy should be for that step
            board, reward, done = play_one_step(board, epsilon)
            if done:
                break
        print(f"Episode: {ep}: {board.merge_score()}, {np.max(board.state.flatten())}")
        if ep > no_episodes_before_training:
            train_step(batch_size)
        if ep % no_episodes_before_update == 0:
            target_model = copy.deepcopy(model)


if __name__ == '__main__':
    main_variation1()
