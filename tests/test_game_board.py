from board import Board2048
import grappa
import numpy as np 

def test_game_board():
    board = Board2048()
    examples = [
        ([0, 0, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 2], [2, 0, 0, 0]),
        ([0, 0, 2, 2], [4, 0, 0, 0]),
        ([2, 0, 0, 0], [2, 0, 0, 0]),
        ([2, 0, 2, 0], [4, 0, 0, 0]),
        ([2, 2, 2, 2], [4, 4, 0, 0]),
        ([2, 2, 4, 4], [4, 8, 0, 0]),
        ([2, 2, 0, 0], [4, 0, 0, 0]),
        ([2, 0, 0, 2], [4, 0, 0, 0]),
        ([0, 0, 2, 2], [4, 0, 0, 0]),
        ([2, 4, 2, 4], [2, 4, 2, 4]),
        ([2, 2, 4, 2], [4, 4, 2, 0]),
        ([2, 4, 4, 2], [2, 8, 2, 0]),
        ([2, 4, 4, 4], [2, 8, 4, 0]),
        ([4, 8, 16, 32], [4, 8, 16, 32])
    ]
    for example in examples:
        result = board._apply_action_to_vector(np.array(example[0]))
        print(example[0], result, example[1])
        np.array_equal(result, example[1]) | grappa.should.be.true