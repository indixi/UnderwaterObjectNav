"""Encode normalized visible/bearing/distance target cues."""

from torch import nn


class TargetFeatureEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, target_cue):
        return self.layers(target_cue.float())

