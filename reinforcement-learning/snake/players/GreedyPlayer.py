from players.DQN import DQN

import numpy as np
import torch.nn.functional as F
import torch
import copy


EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 5e-4

DISCOUNT = 0.99
LEARNING_RATE = 5e-4


class GreedyPlayer:

    def __init__(self, grid_size=12, n_actions=4):
        self.epsilon = EPSILON_START
        self.dqn = DQN(grid_size, n_actions)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)

        # Create a copy of the DQN to compute targets
        self.target_dqn = copy.deepcopy(self.dqn)
        self.target_dqn.eval()
        for p in self.target_dqn.parameters():
            p.requires_grad = False

    def choose_action(self, env):
        valid_actions = np.asarray(env.valid_actions())

        if np.random.random() < self.epsilon:
            # Exploration. Pick a random action.
            return int(np.random.choice(valid_actions))

        # Exploitation. Pick the best action, breaking ties randomly.
        # Use the DQN to compute how good each action is.
        with torch.no_grad():
            obs = env._get_obs()
            x = self.dqn.preprocess(obs).unsqueeze(0)
            q_row = self.dqn.forward(x).squeeze(0).numpy()

        # Filter actions to only allow legal actions
        q_legal = q_row[valid_actions]

        # Find the actions with the highest score
        best_actions = valid_actions[q_legal == q_legal.max()]

        # Pick an action randomly among the best legal actions
        action = int(np.random.choice(best_actions))

        return action

    def learn(self, states, actions, rewards, next_states, dones):
        # Use the target DQN to approximate how good the next_state is.
        with torch.no_grad():
            q_rows = self.target_dqn.forward(next_states)
            next_states_eval = q_rows.max(dim=1).values

        # Compute the target based on the reward and our approximation.
        # Use (1 - dones) to avoid approximating the evaluation of a finished game.
        target = rewards + DISCOUNT * next_states_eval * (1 - dones)

        # Compute how good we thought the actions we took were.
        q_rows = self.dqn.forward(states)
        pred = q_rows.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the Huber loss
        loss = F.smooth_l1_loss(pred, target)

        # Update the DQN based on the loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=10.0)
        self.optimizer.step()

    def update_epsilon(self, episode):
        self.epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(
            -EPSILON_DECAY * episode
        )

    def sync_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
