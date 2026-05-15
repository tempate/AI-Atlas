import pygame


CELL = 150
PAD = 20
GRID_PX = 3 * CELL + 2 * PAD          # 490
TEXT_H = 50
WIN_W = GRID_PX
WIN_H = GRID_PX + TEXT_H

BG = (245, 245, 240)
GRID = (40, 40, 40)
X_COLOR = (40, 80, 200)
O_COLOR = (200, 60, 60)
LINE = 4
GLYPH = 8
RADIUS = 48


class BoardRenderer:
    """Minimal pygame window that draws a 3x3 board on demand."""

    def __init__(self, title="Tic-Tac-Toe"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 28)

    def render(self, board, status=""):
        self._pump_events()
        self.screen.fill(BG)
        self._draw_grid()
        for i, mark in enumerate(board.board):
            cx = PAD + (i % 3) * CELL + CELL // 2
            cy = PAD + (i // 3) * CELL + CELL // 2
            if mark == "X":
                self._draw_x(cx, cy)
            elif mark == "O":
                self._draw_o(cx, cy)
        text = self.font.render(status, True, GRID)
        self.screen.blit(text, (PAD, GRID_PX + 12))
        pygame.display.flip()

    def pause(self, ms):
        """Wait `ms` milliseconds while staying responsive to window events."""
        steps = max(1, ms // 30)
        for _ in range(steps):
            self._pump_events()
            pygame.time.wait(30)

    def close(self):
        pygame.quit()

    # --- internals ---------------------------------------------------------

    def _draw_grid(self):
        for i in range(1, 3):
            xy = PAD + i * CELL
            pygame.draw.line(self.screen, GRID, (xy, PAD), (xy, PAD + 3 * CELL), LINE)
            pygame.draw.line(self.screen, GRID, (PAD, xy), (PAD + 3 * CELL, xy), LINE)

    def _draw_x(self, cx, cy):
        pygame.draw.line(self.screen, X_COLOR, (cx - RADIUS, cy - RADIUS),
                         (cx + RADIUS, cy + RADIUS), GLYPH)
        pygame.draw.line(self.screen, X_COLOR, (cx - RADIUS, cy + RADIUS),
                         (cx + RADIUS, cy - RADIUS), GLYPH)

    def _draw_o(self, cx, cy):
        pygame.draw.circle(self.screen, O_COLOR, (cx, cy), RADIUS, GLYPH)

    def _pump_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
