"""A minimal Snake environment with a gymnasium-compatible API.

Pure game logic — no rendering. Observations are a 2D grid of cell codes;
actions are the four cardinal directions.
"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Action codes
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # (dx, dy) per action

# Cell codes used in the observation grid
EMPTY, BODY, HEAD, FOOD = 0, 1, 2, 3


class SnakeEnv(gym.Env):
    def __init__(self, width=12, height=12, max_steps_without_food=None):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps_without_food = (
            max_steps_without_food if max_steps_without_food is not None
            else 4 * width * height
        )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(height, width), dtype=np.int8
        )

        self.snake: deque[tuple[int, int]] = deque()
        self.direction = RIGHT
        self.food: tuple[int, int] | None = None
        self.steps_since_food = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        cx, cy = self.width // 2, self.height // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = RIGHT
        self.food = self._spawn_food()
        self.steps_since_food = 0

        return self._get_obs(), {}

    def valid_actions(self):
        actions = [UP, RIGHT, DOWN, LEFT]
        reverse = (self.direction + 2) % 4
        actions.remove(reverse)
        return actions

    def step(self, action):
        # Reject 180° reversals — keep heading instead
        if (action + 2) % 4 != self.direction:
            self.direction = int(action)

        dx, dy = DIRS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            return self._get_obs(), -1.0, True, False, {"cause": "wall"}

        # Self collision. The tail will move out of the way unless we eat.
        eats = (new_head == self.food)
        occupied = set(self.snake) if eats else set(list(self.snake)[:-1])
        if new_head in occupied:
            return self._get_obs(), -1.0, True, False, {"cause": "self"}

        self.snake.appendleft(new_head)
        if eats:
            reward = 1.0
            self.steps_since_food = 0
            self.food = self._spawn_food()
        else:
            self.snake.pop()
            reward = 0.0
            self.steps_since_food += 1

        terminated = self.food is None  # board filled — game won
        truncated = self.steps_since_food >= self.max_steps_without_food

        return self._get_obs(), reward, terminated, truncated, {}

    def _spawn_food(self):
        snake_set = set(self.snake)
        free = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in snake_set
        ]
        if not free:
            return None
        idx = int(self.np_random.integers(0, len(free)))
        return free[idx]

    def _get_obs(self):
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        for x, y in self.snake:
            grid[y, x] = BODY
        head_x, head_y = self.snake[0]
        grid[head_y, head_x] = HEAD
        if self.food is not None:
            fx, fy = self.food
            grid[fy, fx] = FOOD
        return grid
