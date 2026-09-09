"""Six-channel world-map encoder."""

from .map_encoder import SpatialMapEncoder


class GlobalMapEncoder(SpatialMapEncoder):
    def __init__(self, in_channels=6, feature_dim=256, pool_size=(5, 9)):
        super().__init__(in_channels, feature_dim, pool_size)

