"""水下 ObjectNav Behavior Cloning 软件包的公共入口。

这里只导出跨模块共享的常量，避免导入本包时立即加载 PyTorch 或
MMDetection；这样仅做数据预处理时也不要求 GPU 依赖已经可用。
"""

from .constants import ACTION_NAMES, GOAL_NAMES, NUM_MAP_CHANNELS

__all__ = ["ACTION_NAMES", "GOAL_NAMES", "NUM_MAP_CHANNELS"]
