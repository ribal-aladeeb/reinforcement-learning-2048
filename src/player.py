import experiments
from board import Board2048
import torch
import logging
import sys
from tqdm import tqdm


def default_reward_func(b: Board2048, next: Board2048, action, done) -> int:
    return next.merge_score() - b.merge_score()


class Player:

    def __init__(self, experiment_folder: str, reward_func=default_reward_func):

        self.experiment = experiments.Experiment(resumed=True,
                                                 folder_name=experiment_folder,
                                                 )
        if not torch.cuda.is_available():
            logging.warning("No GPU: Cuda is not utilized")
            device = "cpu"
        else:
            device = "cuda:0"

        self.model = self.experiment.model
        self.games_history = []
        self.device = device
        self.reward_func = reward_func

    def play_n_games(self, n=1, random_policy=False):
        print(f'Agent will proceed to play {n} games')
        for i in tqdm(range(n)):
            self.play_game(random_policy=random_policy)
        self.experiment.save_games_played(self.games_history)

    def play_game(self, random_policy=False):
        board = Board2048()
        done = False
        single_game_history = []

        while not done:
            available_moves = board.available_moves_as_torch_unit_vector(device=self.device)
            done = torch.max(available_moves) == 0

            state = board.normalized().state_as_4d_tensor().to(self.device)
            if not random_policy:
                Q_values = self.model(state)
            else:
                Q_values = torch.rand((4,))
            available_Q_values = available_moves * Q_values

            next_action = torch.argmax(available_Q_values)
            next_board = board.peek_action(next_action)
            reward = self.reward_func(board, next_board, next_action, done)

            single_game_history.append((board.state, ['u', 'd', 'l', 'r'][int(next_action)], reward))
            board = next_board

        self.games_history.append(single_game_history)
        return single_game_history


def main():

    if len(sys.argv) > 1:
        exp_folder = sys.argv[1]
    else:
        print("Please provide the name of the experiment folder (only provide the name that is found in the experiments folder, not the path).")
        exit()

    player = Player(experiment_folder=exp_folder)

    game_history = player.play_n_games(100)


if __name__ == "__main__":
    main()
