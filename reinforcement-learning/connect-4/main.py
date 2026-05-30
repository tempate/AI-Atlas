from collections import deque
import random

from Board import Board, N_ROWS, N_COLS
from players.AlphaZero import AlphaZero
from players.HumanPlayer import HumanPlayer
from players.MinMaxPlayer import MinMaxPlayer
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

N_TEST_GAMES = 50


def train(alpha_zero):
    replay_buffer = deque(maxlen=BUFFER_SIZE)

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


def get_move(player, board, renderer=None):
    """Get a move from any player, normalizing the differing act() signatures."""
    if isinstance(player, AlphaZero):
        return player.act(board, greedy=True)[0]

    if isinstance(player, HumanPlayer):
        return player.act(board, renderer)

    return player.act(board)


def play_game(players):
    board = Board()

    for n_move in range(N_ROWS * N_COLS):
        player = players[n_move % 2]

        move = get_move(player, board)
        board.play(move)

        winner = board.winner
        if winner:
            return winner

    return "DRAW"


def play_match(alpha_zero, opponent, n_games):
    """Play n_games between AlphaZero and opponent, alternating who starts."""

    alpha_zero.network.eval()
    wins = draws = losses = 0

    for game in range(n_games):
        # Alternate sides so the first-move advantage is shared evenly.
        if game % 2 == 0:
            players = [alpha_zero, opponent]
            alpha_zeros_piece = "X"
        else:
            players = [opponent, alpha_zero]
            alpha_zeros_piece = "O"

        winner = play_game(players)
        if winner == "DRAW":
            draws += 1
        elif winner == alpha_zeros_piece:
            wins += 1
        else:
            losses += 1

    return wins, draws, losses


def play_human(alpha_zero):
    """Play interactively against the trained AlphaZero agent, alternating sides."""

    alpha_zero.network.eval()
    human = HumanPlayer()
    renderer = BoardRenderer()

    try:
        game = 0
        while True:
            # Alternate who moves first each game.
            if game % 2 == 0:
                players = [human, alpha_zero]
            else:
                players = [alpha_zero, human]

            board = Board()
            renderer.render(board)

            winner = None
            for n_move in range(N_ROWS * N_COLS):
                player = players[n_move % 2]

                move = get_move(player, board, renderer)
                board.play(move)
                renderer.render(board)
                if not isinstance(player, HumanPlayer):
                    renderer.pause(MOVE_DELAY_MS)

                winner = board.winner
                if winner:
                    break

            outcome = "Draw" if winner is None else f"{winner} wins"
            renderer.render(board, f"{outcome} — click to play again")
            renderer.wait_for_click()
            game += 1
    finally:
        renderer.close()


def main():
    alpha_zero = AlphaZero()
    train(alpha_zero)

    rivals = [
        ("Random", RandomPlayer()),
        ("MinMax", MinMaxPlayer())
    ]

    for name, opponent in rivals:
        wins, draws, losses = play_match(alpha_zero, opponent, N_TEST_GAMES)
        print(f"AlphaZero vs {name}: {wins} wins, {draws} draws, {losses} losses")

    play_human(alpha_zero)

if __name__ == "__main__":
    main()
