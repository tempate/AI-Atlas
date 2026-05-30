import numpy as np


class RandomPlayer:

    def act(self, board):
        return np.random.choice(board.legal_moves)
