from Board import Board
from players.GreedyPlayer import GreedyPlayer
from players.MinMaxPlayer import MinMaxPlayer
from players.RandomPlayer import RandomPlayer
from renderer import BoardRenderer

from collections import defaultdict

import numpy as np


N_TRAINING_GAMES = 100_000
N_EVALUATION_GAMES = 2_000
MOVE_DELAY_MS = 600


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


def play(q_table, move_delay=MOVE_DELAY_MS):
    """Play interactively against the trained agent. Alternates sides each game."""
    renderer = BoardRenderer()
    try:
        game_idx = 0
        while True:
            human_role = "X" if game_idx % 2 == 0 else "O"
            agent_role = "O" if human_role == "X" else "X"

            agent = GreedyPlayer(agent_role, q_table)
            agent.epsilon = 0.0

            board = Board()
            renderer.render(board, f"Game {game_idx + 1} — you are {human_role}")

            winner = None
            for n_move in range(9):
                turn = "X" if n_move % 2 == 0 else "O"
                if turn == human_role:
                    while True:
                        cell = renderer.wait_for_click()
                        if board.board[cell] == "":
                            break
                    board = board.copy()
                    board.board[cell] = human_role
                    renderer.render(board, f"Game {game_idx + 1}: you played cell {cell}")
                else:
                    move = agent.choose_move(board)
                    board = play_move(board, agent, move)
                    renderer.render(board, f"Game {game_idx + 1}: agent plays cell {move}")
                    renderer.pause(move_delay)

                winner = board.get_winner()
                if winner:
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
    q_table = defaultdict(lambda: np.zeros(9))
    train(q_table)

    scoreX = evaluate(q_table, "X")
    scoreO = evaluate(q_table, "O")
    print("Score as X:", scoreX)
    print("Score as O:", scoreO)

    play(q_table)


if __name__ == "__main__":
    main()
