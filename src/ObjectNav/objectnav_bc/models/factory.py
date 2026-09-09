"""Construct policy components from serialized model configuration."""

from .global_map_encoder import GlobalMapEncoder
from .local_map_encoder import LocalMapEncoder
from .policy_mlp import BCPolicy


def build_policy_components(config: dict, num_goals: int):
    global_encoder = GlobalMapEncoder(
        in_channels=int(config["global_map_channels"]),
        feature_dim=int(config["global_feature_dim"]),
        pool_size=tuple(config["global_pool_size"]),
    )
    local_encoder = LocalMapEncoder(
        in_channels=int(config["local_map_channels"]),
        feature_dim=int(config["local_feature_dim"]),
        pool_size=tuple(config["local_pool_size"]),
    )
    policy = BCPolicy(
        global_dim=int(config["global_feature_dim"]),
        local_dim=int(config["local_feature_dim"]),
        num_goals=num_goals,
        goal_dim=int(config["goal_embedding_dim"]),
        target_dim=int(config["target_feature_dim"]),
        num_actions=int(config["num_actions"]),
    )
    return global_encoder, local_encoder, policy
