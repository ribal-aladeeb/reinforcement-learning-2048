import torch

from src.board import Board2048
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

def test_tensor_flip():
    original = [2,4,8,16]
    reversed = Board2048()._reverse_vector_tensor(torch.tensor(original))
    np.array_equal(reversed, torch.tensor([16,8,4,2])) | grappa.should.be.true

def test_game_board_tensors():
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
        tensor = torch.tensor(example[0])
        result = board._apply_action_to_tensor(tensor)
        print(example[0], result, example[1])
        np.array_equal(result, example[1]) | grappa.should.be.true

def test_available_moves():
    board = Board2048(k=4, populate_empty_cells=False)

    pairs = [
        (np.array([
            [2,4,8,0],
            [0,0,0,0],
            [2,4,16,32],
            [0,0,0,0],
        ]), {'up','down','right'}),
        (np.array([
            [2,4,2,4],
            [2,4,2,4],
            [2,4,2,4],
            [2,4,2,4],
        ]), {'up','down',}),
        (np.array([
            [2,4,2,4],
            [4,2,4,2],
            [2,4,2,4],
            [4,2,4,2],
        ]), set({}) ),
    ]

    for configuration, possible_moves in pairs:
        board.state = configuration
        print(f'for configuration:\n')
        print(configuration)
        print(f'\nthe func returns {board.available_moves()} instead of {possible_moves}')
        (set(board.available_moves().keys()) == possible_moves) | grappa.should.be.true
