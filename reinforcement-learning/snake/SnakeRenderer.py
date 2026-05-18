"""Pygame renderer for SnakeEnv."""

import pygame


CELL_PX = 32
RENDER_FPS = 5
BG = (24, 28, 32)
GRID_LINE = (40, 46, 52)
SNAKE_BODY = (80, 200, 120)
SNAKE_HEAD = (200, 240, 160)
FOOD_COLOR = (220, 90, 90)


class SnakeRenderer:
    def __init__(self, width, height, fps=RENDER_FPS):
        self.width = width
        self.height = height
        self.fps = fps

        pygame.init()
        self.window = pygame.display.set_mode(
            (width * CELL_PX, height * CELL_PX)
        )
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

    def render(self, snake, food):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        self.window.fill(BG)

        for x in range(1, self.width):
            pygame.draw.line(
                self.window, GRID_LINE,
                (x * CELL_PX, 0), (x * CELL_PX, self.height * CELL_PX),
            )
        for y in range(1, self.height):
            pygame.draw.line(
                self.window, GRID_LINE,
                (0, y * CELL_PX), (self.width * CELL_PX, y * CELL_PX),
            )

        if food is not None:
            fx, fy = food
            pygame.draw.rect(
                self.window, FOOD_COLOR,
                pygame.Rect(fx * CELL_PX + 4, fy * CELL_PX + 4,
                            CELL_PX - 8, CELL_PX - 8),
                border_radius=CELL_PX // 2,
            )

        for x, y in list(snake)[1:]:
            pygame.draw.rect(
                self.window, SNAKE_BODY,
                pygame.Rect(x * CELL_PX + 2, y * CELL_PX + 2,
                            CELL_PX - 4, CELL_PX - 4),
                border_radius=6,
            )
        hx, hy = snake[0]
        pygame.draw.rect(
            self.window, SNAKE_HEAD,
            pygame.Rect(hx * CELL_PX + 2, hy * CELL_PX + 2,
                        CELL_PX - 4, CELL_PX - 4),
            border_radius=6,
        )

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.display.quit()
        pygame.quit()
