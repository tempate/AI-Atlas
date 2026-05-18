import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


N_CELL_CODES = 4   # EMPTY, BODY, HEAD, FOOD


class DQN(nn.Module):
    def __init__(self, grid_size, n_actions, in_channels=N_CELL_CODES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_size, grid_size)
            flat_dim = self.features(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def preprocess(self, obs):
        t = torch.as_tensor(obs, dtype=torch.long)
        one_hot = F.one_hot(t, num_classes=N_CELL_CODES)
        return one_hot.permute(2, 0, 1).float()

    def preprocess_batch(self, obs_list):
        obs_np = np.stack(obs_list)                                   # (B, H, W)
        t = torch.as_tensor(obs_np, dtype=torch.long)                 # (B, H, W)
        one_hot = F.one_hot(t, num_classes=N_CELL_CODES)              # (B, H, W, 4)
        return one_hot.permute(0, 3, 1, 2).float()

    def forward(self, x):
        return self.head(self.features(x))
