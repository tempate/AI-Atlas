# from players.RandomPlayer import RandomPlayer
from players.GreedyPlayer import GreedyPlayer
from SnakeRenderer import SnakeRenderer
from SnakeEnv import SnakeEnv
from collections import deque

import numpy as np
import random
import torch


N_TRAINING_EPS = 10_000
N_EVALUATION_EPS = 100
TARGET_SYNC = 1_000
WARMUP = 1_000
SEED = 0


def train(env, player):
    lengths = deque(maxlen=100)
    returns = deque(maxlen=100)

    replay_buffer = deque(maxlen=10_000)
    steps = 0

    for episode in range(N_TRAINING_EPS):
        length = 0
        total_reward = 0

        state, _ = env.reset(seed=SEED + episode)

        while True:
            # Play an action
            action = player.choose_action(env)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Add a small negative reward to prevent stalling
            if reward == 0:
                reward = -0.01

            # Update stats
            length += 1
            total_reward += reward
            steps += 1

            # Save the transition
            replay_buffer.append((state, action, reward, next_state, terminated))

            # Teach the player based on its action
            if len(replay_buffer) >= WARMUP and length % 8 == 0:
                batch = random.sample(list(replay_buffer), 16)
                states, actions, rewards, next_states, dones = zip(*batch)

                states      = player.dqn.preprocess_batch(states)
                next_states = player.dqn.preprocess_batch(next_states)
                actions     = torch.tensor(actions, dtype=torch.long)
                rewards     = torch.tensor(rewards, dtype=torch.float32)
                dones       = torch.tensor(dones,   dtype=torch.float32)

                player.learn(states, actions, rewards, next_states, dones)

            if steps % TARGET_SYNC == 0:
                player.sync_target()

            if terminated or truncated:
                break

            state = next_state

        # Update stats
        lengths.append(length)
        returns.append(total_reward)

        # Print stats
        if episode % 100 == 0:
           print(f"ep {episode:5d} | "
            f"len {np.mean(lengths):5.1f} | "
            f"ret {np.mean(returns):+5.2f}")

        player.update_epsilon(episode)


def evaluate(env, player):
    env.reset(seed=SEED+N_TRAINING_EPS)

    score_stats = []
    steps_stats = []

    player.epsilon = 0.0

    for episode in range(N_EVALUATION_EPS):
        # Play a full game
        score, steps = play(env, player)

        # Update stats
        score_stats.append(score)
        steps_stats.append(steps)

        env.reset()

    # Print stats
    print(f"evaluation | "
    f"steps {np.mean(steps_stats):5.1f} ± {np.std(steps_stats):.2f} | "
    f"score {np.mean(score_stats):+5.2f} ± {np.std(score_stats):.2f}")


def watch(env, player, renderer):
    env.reset(seed=SEED)

    try:
        while True:
            score, steps = play(env, player, renderer)

            print(f"episode end — score={score:.0f} "
                    f"steps={steps} "
                    f"snake length={len(env.snake)} ")
            env.reset()
    except SystemExit:
        pass
    finally:
        renderer.close()


def play(env, player, renderer=None):
    total_reward = 0.0
    steps = 0

    while True:
        if renderer:
            renderer.render(env.snake, env.food)

        action = player.choose_action(env)
        _, reward, terminated, truncated, _ = env.step(action)

        # Update stats
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return total_reward, steps


def main():
    env = SnakeEnv()
    player = GreedyPlayer()

    # Train the RL player
    train(env, player)

    # Evaluate the RL player
    evaluate(env, player)

    # Watch how the RL player plays
    renderer = SnakeRenderer(env.width, env.height)
    watch(env, player, renderer)


if __name__ == "__main__":
    main()
