import numpy as np


EPSILON_START = 1.0
EPSILON_MIN = 0.2
EPSILON_DECAY = 1e-4

DISCOUNT = 0.99
LEARNING_RATE = 0.1


class GreedyPlayer:

    def __init__(self, player, q_table):
        self.player = player
        self.q_table = q_table
        self.epsilon = EPSILON_START

    def choose_move(self, board):
        possible_moves = np.asarray(board.get_possible_moves())

        if np.random.random() < self.epsilon:
            # Exploration. Choose a random move.
            move = np.random.choice(possible_moves)
        else:
            # Exploitation. Pick the best move, and break ties randomly.
            board_code = board.encode_board()
            q_row = self.q_table[board_code]
            q_legal = q_row[possible_moves]
            best_moves = possible_moves[q_legal == q_legal.max()]
            move = np.random.choice(best_moves)

        return int(move)

    def learn(self, trajectory, reward):
        for d, (board,move) in enumerate(reversed(trajectory)):
            code = board.encode_board()
            target = (-DISCOUNT)**d * reward
            self.q_table[code][move] += LEARNING_RATE * (target - self.q_table[code][move])

    def update_epsilon(self, game):
        self.epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(
            -EPSILON_DECAY * game
        )
