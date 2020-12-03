import experiments
from board import Board2048
import torch
import logging
import sys
from tqdm import tqdm
from dqn_lib import reward_func_merge_score


class Player:

    def __init__(self, experiment_folder: str, resumed: bool =True, reward_func=reward_func_merge_score):

        self.experiment = experiments.Experiment(resumed=resumed,
                                                 folder_name=experiment_folder,
                                                 )
        if not torch.cuda.is_available():
            logging.warning("No GPU: Cuda is not utilized")
            self.device = "cpu"
        else:
            self.device = "cuda:0"

        if resumed:
            self.model = self.experiment.model
        else:
            self.model = None

        self.games_history = []
        self.reward_func = reward_func

    def play_n_games(self, n=1, random_policy=False, upleft=False):
        print(f'Agent will proceed to play {n} games')
        for i in tqdm(range(n)):
            if upleft:
                self.basic_upleft_algorithm(k=4)
            else:
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
                Q_values = torch.rand((4,), device=self.device)
            available_Q_values = available_moves * Q_values

            next_action = torch.argmax(available_Q_values)
            next_board = board.peek_action(next_action)
            reward = self.reward_func(board, next_board, next_action, done)

            single_game_history.append((board.state, ['u', 'd', 'l', 'r'][int(next_action)], reward))
            board = next_board

        self.games_history.append(single_game_history)
        return single_game_history

    def basic_upleft_algorithm(self, k=4):
        board = Board2048(k=k)
        simple_score = board.simple_score()
        single_game_history = []
        while True:
            board = board.peek_action("up")
            single_game_history.append((board.state, 'up', board.simple_score()))
            board = board.peek_action("left")
            single_game_history.append((board.state, 'left', board.simple_score()))
            if simple_score == board.simple_score():
                board = board.peek_action('down')
                single_game_history.append((board.state, 'down', board.simple_score()))
                board = board.peek_action('right')
                single_game_history.append((board.state, 'r', board.simple_score()))
                if simple_score == board.simple_score():
                    break
            simple_score = board.simple_score()
        self.games_history.append(single_game_history)
        return board


    def play_state_space_search_game(self, sss_algo=None):
        pass


def main():

    # if len(sys.argv) > 1:
    #     exp_folder = sys.argv[1]
    # else:
    #     print("Please provide the name of the experiment folder (only provide the name that is found in the experiments folder, not the path).")
    #     exit()


    # print("Random Games")
    # random_player = Player(experiment_folder="random_baseline", resumed=False)
    # random_player.play_n_games(15000, random_policy=True)

    print("Upleft games")
    upleft_player = Player(experiment_folder="upleft_baseline", resumed=False)
    upleft_player.play_n_games(15000, upleft=True)


if __name__ == "__main__":
    main()
