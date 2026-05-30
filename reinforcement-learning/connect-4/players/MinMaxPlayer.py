MAX_DEPTH = 4

# Discount per ply so the engine prefers to win sooner and lose later, instead of
# treating every loss as equally bad and giving up (and ignoring immediate threats).
DISCOUNT = 0.95


class MinMaxPlayer:

    def act(self, board):
        self._cache = {}
        scores = {}

        for move in board.legal_moves:
            # Play the move
            next_board = board.copy()
            next_board.play(move)

            # Evaluate the move
            eval = -DISCOUNT * self._minimax(next_board, depth=1)
            scores[move] = eval

        # Choose the move with the highest score
        return max(scores, key=scores.get)

    def _minimax(self, board, depth):
        key = (tuple(board._cells), depth)

        # We have already seen this position, return the score
        if key in self._cache:
            return self._cache[key]

        if board.winner is not None:
            # The current player lost
            score = -1
        elif board.is_full or depth > MAX_DEPTH:
            score = 0
        else:
            scores = []

            for move in board.legal_moves:
                # Play the move
                next_board = board.copy()
                next_board.play(move)

                # Evaluate the new position, discounting deeper outcomes
                eval = -DISCOUNT * self._minimax(next_board, depth+1)
                scores.append(eval)

            score = max(scores)

        # Save the position's score
        self._cache[key] = score

        return score
