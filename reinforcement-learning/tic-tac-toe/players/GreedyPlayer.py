import numpy as np


EPSILON_START = 1.0
EPSILON_MIN = 0.2
EPSILON_DECAY = 1e-4

DISCOUNT = 0.99
LEARNING_RATE = 0.1


class GreedyPlayer:

    def __init__(self, piece, q_table):
        self.piece = piece
        self.q_table = q_table
        self.epsilon = EPSILON_START

    def act(self, board):
        moves = np.asarray(board.legal_moves)

        if np.random.random() < self.epsilon:
            # Exploration. Choose a random move.
            move = np.random.choice(moves)
        else:
            # Exploitation. Pick the best move, and break ties randomly.
            board_code = board.to_tensor()
            q_row = self.q_table[board_code]
            q_legal = q_row[moves]
            best_moves = moves[q_legal == q_legal.max()]
            move = np.random.choice(best_moves)

        return int(move)

    def update(self, trajectory, reward):
        for d, (board,move) in enumerate(reversed(trajectory)):
            code = board.to_tensor()
            target = (-DISCOUNT)**d * reward
            self.q_table[code][move] += LEARNING_RATE * (target - self.q_table[code][move])

    def update_epsilon(self, game):
        self.epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(
            -EPSILON_DECAY * game
        )
