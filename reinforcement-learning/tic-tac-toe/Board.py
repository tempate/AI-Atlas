

class Board():
    def __init__(self):
        self.board = [""] * 9

    def copy(self):
        new_board = Board()
        new_board.board = self.board.copy()
        return new_board

    def get_possible_moves(self):
        possible_moves = []

        for i in range(9):
            if self.board[i] == "":
                possible_moves.append(i)

        return possible_moves

    def get_winner(self):
        """Return the winner if the game is over and none otherwise."""
        # Check rows
        for row in range(3):
            if self.board[3*row] == self.board[3*row+1] == self.board[3*row+2] != "":
                return self.board[3*row]

        # Check columns
        for col in range(3):
            if self.board[col] == self.board[col+3] == self.board[col+6] != "":
                return self.board[col]

        # Check diagonal
        if self.board[2] == self.board[4] == self.board[6] != "":
            return self.board[2]

        # Check anti-diagonal
        if self.board[0] == self.board[4] == self.board[8] != "":
            return self.board[0]

        return None

    def encode_board(self):
        encoding = 1

        for i in range(9):
            encoding <<= 2

            if self.board[i] == "X":
                encoding += 1
            elif self.board[i] == "O":
                encoding += 2

        return encoding
