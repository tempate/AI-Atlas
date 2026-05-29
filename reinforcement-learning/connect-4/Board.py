import numpy as np
import torch


N_ROWS = 6
N_COLS = 7
N_CONNECT = 4


class Board():
    def __init__(self):
        self._cells = [""] * (N_ROWS * N_COLS)
        self.move_count = 0

    def __getitem__(self, pos):
        row, col = pos
        return self._cells[row * N_COLS + col]

    def __setitem__(self, pos, piece):
        row, col = pos
        self._cells[row * N_COLS + col] = piece

    def copy(self):
        clone = Board()
        clone._cells = self._cells.copy()
        clone.move_count = self.move_count
        return clone

    @property
    def is_full(self):
        return self.move_count == N_ROWS * N_COLS

    @property
    def side_to_move(self):
        return "X" if self.move_count % 2 == 0 else "O"

    @property
    def legal_moves(self):
        possible_moves = []

        for col in range(N_COLS):
            if self[0, col] == "":
                possible_moves.append(col)

        return possible_moves

    @property
    def winner(self):
        """Return the winner if the game is over and none otherwise."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(N_ROWS):
            for col in range(N_COLS):
                cell = self[row, col]
                if cell == "":
                    continue

                for d_row, d_col in directions:
                    end_row = row + d_row * (N_CONNECT - 1)
                    end_col = col + d_col * (N_CONNECT - 1)
                    if not (0 <= end_row < N_ROWS and 0 <= end_col < N_COLS):
                        continue

                    if all(self[row + i * d_row, col + i * d_col] == cell for i in range(N_CONNECT)):
                        return cell

        return None

    def play(self, col):
        for row in range(N_ROWS - 1, -1, -1):
            if self[row, col] == "":
                self[row, col] = self.side_to_move
                self.move_count += 1
                return row

        raise ValueError(f"Column {col} is full")

    def to_tensor(self):
        bitboards = np.zeros((2, 6, 7), dtype=np.float32)

        for row in range(N_ROWS):
            for col in range(N_COLS):
                piece = self[row, col]
                if piece == self.side_to_move:
                    bitboards[0, row, col] = 1
                elif piece != "":
                    bitboards[1, row, col] = 1

        return torch.from_numpy(bitboards)
