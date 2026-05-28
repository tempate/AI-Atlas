

class Board():
    def __init__(self):
        self._cells = [""] * 9

    def __getitem__(self, pos):
        row, col = pos
        return self._cells[row * 3 + col]

    def __setitem__(self, pos, piece):
        row, col = pos
        self._cells[row * 3 + col] = piece

    def copy(self):
        clone = Board()
        clone._cells = self._cells.copy()
        return clone

    @property
    def legal_moves(self):
        possible_moves = []

        for i in range(9):
            if self._cells[i] == "":
                possible_moves.append(i)

        return possible_moves

    @property
    def winner(self):
        """Return the winner if the game is over and none otherwise."""
        # Check rows
        for row in range(3):
            if self[row,0] == self[row,1] == self[row,2] != "":
                return self[row,0]

        # Check columns
        for col in range(3):
            if self[0,col] == self[1,col] == self[2,col] != "":
                return self[0,col]

        # Check diagonal
        if self[0,2] == self[1,1] == self[2,0] != "":
            return self[0,2]

        # Check anti-diagonal
        if self[0,0] == self[1,1] == self[2,2] != "":
            return self[0,0]

        return None

    def play(self, pos, piece):
        self._cells[pos] = piece

    def to_tensor(self):
        encoding = 1

        for i in range(9):
            encoding <<= 2

            if self._cells[i] == "X":
                encoding += 1
            elif self._cells[i] == "O":
                encoding += 2

        return encoding
