
class HumanPlayer:

    def act(self, board, renderer):
        moves = board.legal_moves

        while True:
            move = renderer.wait_for_click()
            if move in moves:
                return move
