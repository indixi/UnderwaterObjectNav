# MapEncoder 只处理已经生成的语义地图，不直接处理 RGB 或 Depth。
# 因此感知模型可以冻结并与策略训练解耦。
import torch
from torch import nn


class MapEncoder(nn.Module):
    """将 [B, 7, H, W] 语义地图压缩为 [B, 256] 特征。"""

    def __init__(self, in_channels=7, feature_dim=256):
        super().__init__()
        # 三个 stride=2 的卷积逐步降低空间尺寸，同时扩大通道数。
        # AdaptiveAvgPool 保证地图尺寸变化时仍能得到固定长度向量。
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(128, feature_dim), nn.ReLU(inplace=True))

    def forward(self, semantic_map):
        """执行地图编码；输入会转换为 float32 以匹配网络权重。"""
        return self.projection(self.convs(semantic_map.float()))
