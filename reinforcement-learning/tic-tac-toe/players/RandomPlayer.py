import numpy as np


class RandomPlayer:
    def __init__(self, player):
        self.player = player

    def choose_move(self, board):
        possible_moves = board.get_possible_moves()
        move = np.random.choice(possible_moves)
        return move
