import numpy as np


class RandomPlayer:

    def act(self, board):
        move = np.random.choice(board.legal_moves)
        return move, {}
