"""Spatially preserving CNN map encoder base class."""

from torch import nn


class SpatialMapEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim, pool_size):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(tuple(pool_size)),
        )
        flattened = 128 * int(pool_size[0]) * int(pool_size[1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, semantic_map):
        return self.projection(self.convs(semantic_map.float()))
