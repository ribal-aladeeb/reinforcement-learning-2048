'''
This player uses informed search (specifically A*) to decide what move it should do at each step (i.e each board state)
'''
from board import Board2048
from state_space_search import Node, A_star


class InformedPlayer():

    def __init__(self, b: Board2048):
        self.board = b

    def play(self):
        game = self.board
        i = 0
        while 2048 not in game.state:
            i += 1
            print(f'\nMOVE {i}\n')
            possible_moves = game.available_moves()
            if len(possible_moves) == 0:
                break
            path_lengths = {}
            for move in possible_moves:
                path_lengths[move] = A_star(possible_moves[move])['path_length']

            sorted_actions = sorted(path_lengths.items(), key=lambda x: x[1])
            game = game.peek_action(sorted_actions[0][0])
            print(game)

        return game


def main():
    b = Board2048()
    print(b)
    print()
    p = InformedPlayer(b)
    game = p.play()
    print(game)

if __name__ == '__main__':
    main()


