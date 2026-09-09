"""Fuse Global/Local/Goal/Target/Yaw features into action logits."""

import torch
from torch import nn

from .target_encoder import TargetFeatureEncoder


class BCPolicy(nn.Module):
    def __init__(
        self,
        global_dim=256,
        local_dim=128,
        num_goals=1,
        goal_dim=8,
        target_dim=16,
        num_actions=4,
    ):
        super().__init__()
        self.goal_embedding = nn.Embedding(num_goals, goal_dim)
        self.target_encoder = TargetFeatureEncoder(3, target_dim)
        fused_dim = global_dim + local_dim + goal_dim + target_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(
        self,
        global_feature,
        local_feature,
        goal_id,
        target_cue,
        yaw,
    ):
        orientation = torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)
        fused = torch.cat(
            (
                global_feature,
                local_feature,
                self.goal_embedding(goal_id.long()),
                self.target_encoder(target_cue),
                orientation,
            ),
            dim=-1,
        )
        return self.mlp(fused)
