import torch.nn.functional as F
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, n_moves, in_channels=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(24, n_moves),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(12*32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        if x.dim() == 3:
            # The input is unbatched.
            # Add an extra dimension.
            x = x.unsqueeze(0)

        features = self.features(x)
        policy = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy, value
