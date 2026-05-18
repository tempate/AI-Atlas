from Board import Board
from players.GreedyPlayer import GreedyPlayer
from players.MinMaxPlayer import MinMaxPlayer
from players.RandomPlayer import RandomPlayer
from renderer import BoardRenderer

from collections import defaultdict

import numpy as np


N_TRAINING_GAMES = 100_000
N_EVALUATION_GAMES = 2_000
N_WATCH_GAMES = 4
MOVE_DELAY_MS = 600
GAME_PAUSE_MS = 1500


def train(q_table):
    player1 = GreedyPlayer("X", q_table)
    player2 = GreedyPlayer("O", q_table)
    players = [player1, player2]

    for game in range(N_TRAINING_GAMES):
        board = Board()
        trajectory = []

        for n_move in range(9):
            # Find the player whose turn it is to play
            turn = n_move % 2
            player = players[turn]

            # Let the player make a move
            move = player.choose_move(board)
            trajectory.append((board,move))

            next_board = play_move(board, player, move)
            winner = next_board.get_winner()

            if winner or n_move == 8:
                # Give a small negative reward to draws
                reward = -0.1

                # Reward the player if his move wins
                if winner == player.player:
                    reward = 1 - 0.05 * n_move

                player.learn(trajectory, reward)
                break

            board = next_board

        player1.update_epsilon(game)
        player2.update_epsilon(game)


def evaluate(q_table, player):
    # Set up the players
    if player == "X":
        greedy = GreedyPlayer("X", q_table)
        random = MinMaxPlayer("O")
        players = [greedy, random]
    else:
        greedy = GreedyPlayer("O", q_table)
        random = MinMaxPlayer("X")
        players = [random, greedy]

    greedy.epsilon = 0.0

    # Calculate the score
    score = {"wins": 0, "losses": 0, "draws": 0}

    for _ in range(N_EVALUATION_GAMES):
        winner = play_game(players)
        if winner == player:
            score["wins"] += 1
        elif winner == "DRAW":
            score["draws"] += 1
        else:
            score["losses"] += 1

    # Noramlize the score
    for key in score:
        score[key] /= N_EVALUATION_GAMES

    return score


def play_game(players):
    board = Board()

    for n_move in range(9):
        # Find the player whose turn it is to play
        turn = n_move % 2
        player = players[turn]

        # Allow the player to make a move
        move = player.choose_move(board)
        board = play_move(board, player, move)

        # Check if there is a winner
        winner = board.get_winner()
        if winner:
            return winner

    return "DRAW"


def play_move(board, player, move):
    next_board = board.copy()
    next_board.board[move] = player.player
    return next_board


def watch(q_table, n_games=N_WATCH_GAMES,
          move_delay=MOVE_DELAY_MS, game_pause=GAME_PAUSE_MS):
    """Replay greedy-vs-random games in a window so you can watch the agent."""
    renderer = BoardRenderer()
    try:
        for game_idx in range(n_games):
            greedy_role = "X" if game_idx % 2 == 0 else "O"
            other_role = "O" if greedy_role == "X" else "X"

            greedy = GreedyPlayer(greedy_role, q_table)
            greedy.epsilon = 0.0
            rand = MinMaxPlayer(other_role)
            players = [greedy, rand] if greedy_role == "X" else [rand, greedy]

            board = Board()
            renderer.render(
                board, f"Game {game_idx + 1}/{n_games} — greedy plays {greedy_role}"
            )
            renderer.pause(game_pause // 2)

            winner = None
            for n_move in range(9):
                player = players[n_move % 2]
                move = player.choose_move(board)
                board = play_move(board, player, move)
                renderer.render(
                    board, f"Game {game_idx + 1}: {player.player} plays cell {move}"
                )
                renderer.pause(move_delay)
                winner = board.get_winner()
                if winner:
                    break

            if winner == greedy_role:
                outcome = "greedy wins"
            elif winner is None:
                outcome = "draw"
            else:
                outcome = "greedy loses"
            renderer.render(board, f"Game {game_idx + 1}: {outcome}")
            renderer.pause(game_pause)
    finally:
        renderer.close()


def main():
    q_table = defaultdict(lambda: np.zeros(9))
    train(q_table)

    scoreX = evaluate(q_table, "X")
    scoreO = evaluate(q_table, "O")
    print("Score as X:", scoreX)
    print("Score as O:", scoreO)

    watch(q_table)


if __name__ == "__main__":
    main()
