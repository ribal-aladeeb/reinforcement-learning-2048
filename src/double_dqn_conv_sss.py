'''
This will become a convolutional network
'''
from collections import deque
import pprint
from board import Board2048
from dqn_lib import play_one_step,  train_step, reward_func_merge_score, board_as_4d_tensor, board_as_flattened_tensor, extract_samples_dense, extract_samples_conv, training_loop
import torch
import torch.nn as nn
import numpy as np
import copy
from experiments import Experiment
import os
from device import device
from state_space_search import generate_replay_buffer_using_A_star
from configs.double_dqn_sss import HYPERPARAMS, model, target_model, loss_fn, optimizer, batch_size, discount_factor, target_model, replay_buffer_length, learning_rate, \
loss_fn, optimizer, no_episodes, no_episodes_to_reach_epsilon, min_epsilon, no_episodes_before_training, no_episodes_before_updating_target, use_double_dqn, snapshot_game_every_n_episodes, no_episodes_to_fill_up_existing_model_replay_buffer

pp = pprint.PrettyPrinter(indent=4)
job_name = input("What is the job name: ")

if job_name:
    experiment = Experiment(
        python_file_name=os.path.basename(__file__),
        folder_name=job_name,
        model=model,
    )
else:
    experiment = Experiment(
        python_file_name=os.path.basename(__file__),
        model=model,
    )

experiment.add_hyperparameter(HYPERPARAMS)
pp.pprint(experiment.hyperparameters)

model_path = ""

if model_path:
    model = torch.load(model_path)
    model.eval()

def main():

    training_loop(
        replay_buffer_length,
        no_episodes,
        no_episodes_to_reach_epsilon,
        no_episodes_to_fill_up_existing_model_replay_buffer,
        min_epsilon,
        model,
        reward_func_merge_score,
        board_as_4d_tensor,
        device,
        experiment,
        snapshot_game_every_n_episodes,
        no_episodes_before_training,
        batch_size, discount_factor,
        target_model,
        loss_fn,
        optimizer,
        use_double_dqn,
        no_episodes_before_updating_target,
        extract_samples_conv,
        replay_buffer_override=generate_replay_buffer_using_A_star(100, replay_buffer_length)
        )

if __name__ == "__main__":
    main()
