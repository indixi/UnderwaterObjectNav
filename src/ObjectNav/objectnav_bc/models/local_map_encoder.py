"""Four-channel robot-centred local-map encoder."""

from .map_encoder import SpatialMapEncoder


class LocalMapEncoder(SpatialMapEncoder):
    def __init__(self, in_channels=4, feature_dim=128, pool_size=(4, 4)):
        super().__init__(in_channels, feature_dim, pool_size)

