# TicTacToe

A RL agent that plays TicTacToe.

### Design

The agent has a Q-table predicting how good each move is. Given a position, if the agent is exploring, it chooses a random move; but if the agent is exploiting, it encodes the position into a number, then finds its corresponding Q-row in the Q-table, and then chooses the legal move with the highest Q-score. The agent starts by always exploring, but the probability of exploiting increases after each training episode. To train the agent, we keep track of all the moves he made during a game, and when the game is over, we update the Q-table based on the game's result.

### Evaluation

Over two thousand games, the agent's win-rate against a random player is:
| Side | Wins | Losses | Draws |
|---|---:|---:|---:|
| X | 0.988 | 0.000 | 0.012 |
| O | 0.901 | 0.000 | 0.099 |
