import numpy as np


class RandomPlayer:
    def __init__(self, piece):
        self.piece = piece

    def act(self, board):
        return np.random.choice(board.legal_moves)
