import pygame

from Board import N_ROWS, N_COLS


CELL = 80
PAD = 20
GRID_W = N_COLS * CELL + 2 * PAD
GRID_H = N_ROWS * CELL + 2 * PAD
TEXT_H = 50
WIN_W = GRID_W
WIN_H = GRID_H + TEXT_H

BG = (245, 245, 240)
BOARD = (30, 70, 180)
EMPTY = (245, 245, 240)
X_COLOR = (210, 50, 50)
O_COLOR = (230, 200, 50)
TEXT = (40, 40, 40)
RADIUS = CELL // 2 - 6


class BoardRenderer:
    """Minimal pygame window that draws a Connect 4 board on demand."""

    def __init__(self, title="Connect 4"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SCALED)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 28)

    def render(self, board, status=""):
        self._pump_events()
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, BOARD,
                         (PAD, PAD, N_COLS * CELL, N_ROWS * CELL))

        for row in range(N_ROWS):
            for col in range(N_COLS):
                cx = PAD + col * CELL + CELL // 2
                cy = PAD + row * CELL + CELL // 2
                mark = board[row, col]
                if mark == "X":
                    color = X_COLOR
                elif mark == "O":
                    color = O_COLOR
                else:
                    color = EMPTY
                pygame.draw.circle(self.screen, color, (cx, cy), RADIUS)

        text = self.font.render(status, True, TEXT)
        self.screen.blit(text, (PAD, GRID_H + 12))
        pygame.display.flip()

    def pause(self, ms):
        """Wait `ms` milliseconds while staying responsive to window events."""
        steps = max(1, ms // 30)
        for _ in range(steps):
            self._pump_events()
            pygame.time.wait(30)

    def wait_for_click(self):
        """Block until the user left-clicks a column; return its index (0..N_COLS-1)."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    col = self._col_at(event.pos)
                    if col is not None:
                        return col
            pygame.time.wait(20)

    def close(self):
        pygame.quit()

    # --- internals ---------------------------------------------------------

    def _col_at(self, pos):
        x, y = pos
        if not (PAD <= x < PAD + N_COLS * CELL and PAD <= y < PAD + N_ROWS * CELL):
            return None
        return int((x - PAD) // CELL)

    def _pump_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
