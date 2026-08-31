"""将地图、目标类别和朝向融合为离散动作 logits。"""

import torch
from torch import nn


class BCPolicy(nn.Module):
    """第一阶段 BC 策略：256 + 8 + 2 维输入，输出 4 个动作分数。"""

    def __init__(self, map_dim=256, num_goals=1, goal_dim=8, num_actions=4):
        super().__init__()
        # Embedding 是可学习的目标类别表示，而不是目标的空间坐标。
        self.goal_embedding = nn.Embedding(num_goals, goal_dim) # 创建目标类别嵌入层
        self.mlp = nn.Sequential(nn.Linear(map_dim + goal_dim + 2, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_actions))   # 创建多层感知机（MLP），输入维度为 map_dim + goal_dim + 2，输出维度为 num_actions

    def forward(self, map_feature, goal_id, yaw):
        # yaw 是周期变量，使用 sin/cos 后 0 和 2*pi 会得到相同表示，
        # 避免角度在边界处出现数值跳变。
        orientation = torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1) # 将角度 yaw 转换为 sin 和 cos 表示，避免角度在边界处出现数值跳变
        return self.mlp(torch.cat((map_feature, self.goal_embedding(goal_id.long()), orientation), dim=-1)) # 将地图特征、目标类别嵌入和方向表示拼接在一起，作为 MLP 的输入，输出动作 logits
