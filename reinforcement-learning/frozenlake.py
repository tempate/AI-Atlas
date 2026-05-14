"""Reinforcement learning on FrozenLake-v1.

Scaffold: the environment, hyperparameters, training/eval loops, and reporting
are wired up. The actual RL logic is left as TODOs for you to implement:
  1. epsilon-greedy action selection      -> select_action()
  2. the value-update rule (e.g. Q-learning) -> train() inner loop

Run:  python frozenlake.py
"""

import time

import numpy as np
import gymnasium as gym


# --- Hyperparameters --------------------------------------------------------
ENV_NAME = "FrozenLake-v1"
IS_SLIPPERY = False         # deterministic variant
MAP_NAME = "4x4"            # "4x4" or "8x8"

N_EPISODES = 5_000          # training episodes
MAX_STEPS = 100             # step cap per episode

LEARNING_RATE = 0.1         # alpha
DISCOUNT = 0.99             # gamma

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.0005      # exponential decay rate applied per episode

SEED = 42


def make_env(render_mode=None):
    return gym.make(
        ENV_NAME,
        is_slippery=IS_SLIPPERY,
        map_name=MAP_NAME,
        render_mode=render_mode,
    )


def select_action(q_table, state, epsilon, action_space):
    """Return an action for `state` using an epsilon-greedy policy."""
    if np.random.random() < epsilon:
        # Explore. Pick a random action.
        return action_space.sample()

    # Exploit. Pick a greedy action, breaking ties randomly.
    best_actions = np.flatnonzero(q_table[state] == q_table[state].max())
    return int(np.random.choice(best_actions))


def train(env):
    """Run the training loop and return the learned Q-table + reward history."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    epsilon = EPSILON_START
    rewards_per_episode = []

    for episode in range(N_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = select_action(q_table, state, epsilon, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)

            target = reward + DISCOUNT * np.max(q_table[next_state])
            q_table[state, action] += LEARNING_RATE * (target - q_table[state, action])

            state = next_state
            total_reward += reward
            if terminated or truncated:
                break

        # decay epsilon toward EPSILON_MIN
        epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(
            -EPSILON_DECAY * episode
        )
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 1000 == 0:
            recent = np.mean(rewards_per_episode[-1000:])
            print(f"episode {episode + 1:>6}  avg reward (last 1k): {recent:.3f}  "
                  f"epsilon: {epsilon:.3f}")

    return q_table, rewards_per_episode


def evaluate(q_table, n_episodes=1000):
    """Greedy rollout of the learned policy; prints the success rate."""
    env = make_env()
    wins = 0

    for episode in range(n_episodes):
        state, _ = env.reset(seed=10_000 + episode)
        for _ in range(MAX_STEPS):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                wins += reward
                break

    env.close()
    print(f"\nevaluation: {wins / n_episodes:.1%} success over {n_episodes} episodes")


def watch(q_table, n_episodes=5, pause=0.4):
    """Replay the greedy policy in a window so you can watch the agent move."""
    env = make_env(render_mode="human")

    for episode in range(n_episodes):
        state, _ = env.reset(seed=10_000 + episode)
        for _ in range(MAX_STEPS):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            time.sleep(pause)
            if terminated or truncated:
                if terminated and reward:
                    outcome = "reached the goal"
                elif terminated:
                    outcome = "fell in a hole"
                else:
                    outcome = "ran out of steps (policy is stuck/untrained?)"
                print(f"episode {episode + 1}: {outcome}")
                break

    env.close()


def main():
    env = make_env()
    q_table, _ = train(env)
    env.close()

    print("\nlearned Q-table:")
    print(np.round(q_table, 3))

    evaluate(q_table)
    watch(q_table)


if __name__ == "__main__":
    main()
