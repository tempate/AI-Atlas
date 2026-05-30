# Connect-4

An AlphaZero-like RL agent that plays Connect-4.

### Design

The agent has a single network with two heads: one to evaluate the position and another to evaluate the moves. To pick a move, the agent creates a tree with the possible moves and visits nodes based on a scoring function. During exploration, the agent picks a move randomly based on its visit count, and during exploitation, the agent picks the most visited move. The agent trains by playing games against itself, storing the different positions, trees, and results, and then updating the networks to better predict the game's result and the winning moves.

### Evaluation

Over fifty games games, the agent's win-rate is:
| Player | Wins | Losses | Draws |
|:---:|:---:|:---:|:---:|
| Random | 50 | 0 | 0 |
| MinMax4 | 50 | 0 | 0 |
| Human | 3 | 0 | 0 |
