
class MinMaxPlayer:
    def __init__(self, piece):
        self.piece = piece
        self._cache = {}

    def act(self, board):
        scores = {}

        for move in board.legal_moves:
            scores[move] = self._eval_move(board, move, self.piece)

            # If the move wins, choose it
            if scores[move] == 1:
                return move

        # Choose the move with the highest score
        return max(scores, key=scores.get)

    def _eval_move(self, board, move, piece):
        # Play the move
        next_board = board.copy()
        next_board.play(move, piece)

        # Evaluate the move based on the opponent's best move
        opponent = "O" if piece == "X" else "X"
        return self._minimax(next_board, opponent)

    def _minimax(self, board, piece):
        code = board.to_tensor()

        # We have already seen this position, return the score
        if code in self._cache:
            return self._cache[code]

        winner = board.winner
        if winner is None:
            moves = board.legal_moves

            if len(moves) == 0:
                score = 0
            else:
                scores = [self._eval_move(board, m, piece) for m in moves]
                score = max(scores) if piece == self.piece else min(scores)
        else:
            score = 1 if winner == self.piece else -1

        # Save the positions score
        self._cache[code] = score

        return score
