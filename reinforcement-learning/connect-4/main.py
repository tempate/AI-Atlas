from collections import deque
import random

from Board import Board, N_ROWS, N_COLS
from players.AlphaZero import AlphaZero
from players.RandomPlayer import RandomPlayer
from renderer import BoardRenderer


N_GAMES = 1_000
MOVE_DELAY_MS = 600

BUFFER_SIZE = 20_000
BATCH_SIZE = 64

N_ITS = 50
N_GAMES_PER_IT = 64
N_TRAIN_STEPS_PER_IT = 400
N_EVAL_GAMES_PER_IT = 20


def train(alpha_zero):
    replay_buffer = deque(maxlen=10_000)

    for it in range(N_ITS):
        alpha_zero_old = alpha_zero.copy()

        # 1. Self-play: generate games with the current network
        for _ in range(N_GAMES_PER_IT):
            _, game_history = alpha_zero.self_play(Board())
            replay_buffer.extend(game_history)

        # 2. Train: do K gradient steps on samples from the buffer
        avg_value_loss = 0
        avg_policy_loss = 0

        for _ in range(N_TRAIN_STEPS_PER_IT):
            batch = random.sample(list(replay_buffer), BATCH_SIZE)
            value_loss, policy_loss = alpha_zero.update(batch)
            avg_value_loss += value_loss
            avg_policy_loss += policy_loss

        # 3. Evaluate.
        win_rate = evaluate(alpha_zero, alpha_zero_old)
        avg_value_loss /= N_TRAIN_STEPS_PER_IT
        avg_policy_loss /= N_TRAIN_STEPS_PER_IT

        print(f"{it}. Win-rate: {win_rate:.2f}. "
              f"Value loss: {avg_value_loss:.2f}. "
              f"Policy loss: {avg_policy_loss:.2f}.")


def evaluate(alpha_zero, alpha_zero_old):
    score = 0

    for _ in range(N_EVAL_GAMES_PER_IT):
        # Set up the players
        if random.random() < 0.5:
            players = [alpha_zero, alpha_zero_old]
            alpha_zeros_piece = "X"
        else:
            players = [alpha_zero_old, alpha_zero]
            alpha_zeros_piece = "O"

        # Calculate the score
        winner = play_game(players)
        if winner == alpha_zeros_piece:
            score += 1
        elif winner != "DRAW":
            score -= 1

    # Noramlize the score
    score /= N_EVAL_GAMES_PER_IT

    return score


def play_game(players):
    board = Board()

    for n_move in range(N_ROWS * N_COLS):
        turn = n_move % 2
        player = players[turn]

        move, _ = player.act(board, greedy=True)
        board.play(move)

        winner = board.winner
        if winner:
            return winner

    return "DRAW"


def play(alpha_zero, move_delay=MOVE_DELAY_MS):
    """Play interactively against the trained AlphaZero agent. Alternates sides each game."""
    alpha_zero.network.eval()

    renderer = BoardRenderer()
    try:
        game_idx = 0
        while True:
            human_role = "X" if game_idx % 2 == 0 else "O"

            board = Board()
            renderer.render(board, f"Game {game_idx + 1} — you are {human_role}")

            winner = None
            while True:
                if board.side_to_move == human_role:
                    while True:
                        col = renderer.wait_for_click()
                        if board[0, col] == "":
                            break
                    board.play(col)
                    renderer.render(board, f"Game {game_idx + 1}: you played column {col}")
                else:
                    move, _ = alpha_zero.act(board, greedy=True)
                    board.play(move)
                    renderer.render(board, f"Game {game_idx + 1}: agent plays column {move}")
                    renderer.pause(move_delay)

                winner = board.winner
                if winner or board.is_full:
                    break

            if winner == human_role:
                outcome = "you win!"
            elif winner is None:
                outcome = "draw"
            else:
                outcome = "agent wins"
            renderer.render(board, f"Game {game_idx + 1}: {outcome} — click to play again")
            renderer.wait_for_click()
            game_idx += 1
    finally:
        renderer.close()


def main():
    alpha_zero = AlphaZero()
    train(alpha_zero)
    play(alpha_zero)


if __name__ == "__main__":
    main()
