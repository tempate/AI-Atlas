from players.Network import Network
from dataclasses import dataclass, field

import numpy as np
import torch.nn.functional as F
import torch


DISCOUNT = 0.99
LEARNING_RATE = 5e-4

N_SIMULATIONS = 128
C_PUCT = 1.5
TEMPERATURE_THRESHOLD = 10

@dataclass
class Node:
    moves: dict = field(default_factory=dict)
    score: float = 0
    value: float = 0
    visits: int = 0


class AlphaZero:

    def __init__(self, n_moves=7):
        self.n_moves = n_moves

        # Create the network
        self.network = Network(n_moves)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

    def copy(self):
        clone = AlphaZero(self.n_moves)
        clone.network.load_state_dict(self.network.state_dict())
        return clone

    def act(self, board, greedy=False):
        # TODO: Reuse the same tree among moves of the same game
        root = Node()

        self.network.eval()
        for _ in range(N_SIMULATIONS):
            self._simulate(board, root)

        # Normalize each child's number of visits
        visits = []
        for move, child in root.moves.items():
            visits.append(child.visits / (root.visits - 1))

        if not greedy and board.move_count < TEMPERATURE_THRESHOLD:
            # Choose a move randomly based on how many times we have played each
            visits = np.array(visits, dtype=float)
            moves = list(root.moves.keys())
            move = np.random.choice(moves, p=visits)
        else:
            # Choose randomly among the most visited moves
            most_visited = max(visits)
            best_moves = []
            for move, visit in zip(root.moves.keys(), visits):
                if visit == most_visited:
                    best_moves.append(move)
            move = np.random.choice(best_moves)

        # Add illegal moves with 0 visits to simplify training
        visits = [0] * self.n_moves
        for move_, child in root.moves.items():
            visits[move_] = child.visits / (root.visits - 1)

        return move, visits

    def self_play(self, board):
        if board.winner is not None:
            return -1, []

        if board.is_full:
            return 0, []

        # Pick a move
        move, visits = self.act(board)

        # Play the move
        code = board.to_tensor()
        board.play(move)

        # Play the game from the new position
        score, history = self.self_play(board)

        # Save the position in the history
        history.append((code, visits, -score))

        return -score, history

    def update(self, batch):
        # Transform the batch into tensors
        boards, visits, scores = zip(*batch)
        boards = torch.stack(boards)
        visits = torch.tensor(visits, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)

        # Ensure the BatchNorm2d layers use batch stats
        self.network.train()

        # Compute the policy and value of the boards
        policy, value = self.network.forward(boards)

        # Compute how wrong is the network's value estimate
        value_loss = F.mse_loss(value, scores)

        # Compute how wrong is the network's policy estimate
        log_probs = F.log_softmax(policy, dim=1)
        policy_loss = -(visits * log_probs).sum(dim=1).mean()

        loss = value_loss + policy_loss

        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return value_loss, policy_loss

    def _simulate(self, board, node):
        node.visits += 1

        if board.winner is not None:
            # Since someone has won, the side to move must have lost.
            return -1

        if board.is_full:
            # There are no more possible moves, so the game is a draw.
            return 0

        if len(node.moves) == 0:
            # The game is not over, but the current node is a leaf.
            # Evaluate the position with the network.
            with torch.no_grad():
                x = board.to_tensor()
                policy, value = self.network.forward(x)
                policy = policy.squeeze(0)

            # Expand the node with all its legal moves.
            moves = board.legal_moves
            moves_logits = policy[moves]
            moves_probs = F.softmax(moves_logits, dim=0)

            for i, move in enumerate(moves):
                child = Node()
                child.score = moves_probs[i].item()
                node.moves[move] = child

            return value.item()

        # Compute the puct score of each move
        scores = {}
        for move, child in node.moves.items():
            scores[move] = self._puct(node, child)

        # Pick randomly among the best scoring legal moves
        best_score = max(scores.values())
        best_moves = [m for m in scores if scores[m] == best_score]
        best_move = np.random.choice(best_moves)

        # Update the board
        new_board = board.copy()
        new_board.play(best_move)

        # Compute the value of the child
        child = node.moves[best_move]
        value = -self._simulate(new_board, child)
        child.value += value

        return value

    def _puct(self, parent, node):
        # Compute the Q-score as the mean return value of the node.
        exploitation = 0
        if node.visits > 0:
            exploitation = node.value / node.visits

        # Compute how often we visit the node.
        n_visits_score = np.sqrt(parent.visits) / (1 + node.visits)

        # Compute the U-score.
        exploration = C_PUCT * node.score * n_visits_score

        return exploitation + exploration
