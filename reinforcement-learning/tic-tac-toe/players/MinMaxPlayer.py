import numpy as np


class MinMaxPlayer:
    def __init__(self, player):
        self.player = player
        self._cache = {}

    def choose_move(self, board):
        scores = {}

        for move in board.get_possible_moves():
            scores[move] = self.eval_move(board, move, self.player)

            # If the move wins, choose it
            if scores[move] == 1:
                return move

        # Choose the move with the highest score
        return max(scores, key=scores.get)

    def eval_move(self, board, move, player):
        # Play the move
        next_board = board.copy()
        next_board.board[move] = player

        # Evaluate the move based on the opponent's best move
        opponent = "O" if player == "X" else "X"
        return self.minimax(next_board, opponent)

    def minimax(self, board, player):
        code = board.encode_board()

        # We have already seen this position, return the score
        if code in self._cache:
            return self._cache[code]

        winner = board.get_winner()
        if winner is None:
            moves = board.get_possible_moves()

            if len(moves) == 0:
                score = 0
            else:
                scores = [self.eval_move(board, m, player) for m in moves]
                score = max(scores) if player == self.player else min(scores)
        else:
            score = 1 if winner == self.player else -1

        # Save the positions score
        self._cache[code] = score

        return score
