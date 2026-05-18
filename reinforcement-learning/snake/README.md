# Snake

A RL agent that plays Snake.

### Design

The implementation follows the paper [Human-level control through deep RL](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf). In short, the agent has two DQNs predicting how good each action in a given position is. The agent uses one of the DQNs to pick actions and the other to evaluate them. We update the weights of the first DQN every 8 steps in batches of 16, drawing the batches randomly from a queue with experiences, which are just tuples with the state, the action the agent took, the reward he obtained, and the next state.

### Evaluation

On average, over a hundred games, the agent survives for 383.8 ± 189.85 steps and obtains a reward of 22.22 ± 7.40. The file [recording.mp4](recording.mp4) is a recording of how the agent plays.
