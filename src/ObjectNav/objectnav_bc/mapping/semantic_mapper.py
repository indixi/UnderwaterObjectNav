"""Accumulative 2-D semantic map in the world XY plane."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..constants import NUM_MAP_CHANNELS
from ..perception.depth_projection import CameraIntrinsics, depth_to_world


@dataclass(frozen=True)
class MapConfig:
    """二维世界地图参数。

    ``origin_x/y`` 是地图左下角在世界坐标中的位置；默认地图覆盖
    [-5, 5] x [-5, 5]，分辨率 0.5 m 时得到 20 x 20 个网格。
    """
    width_m: float = 10.0
    height_m: float = 10.0
    resolution_m: float = 0.5
    origin_x: float = -5.0
    origin_y: float = -5.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MapConfig":
        """Load map settings from the ``dataset`` section of bc.yaml."""
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("读取 bc.yaml 需要 PyYAML，请安装 requirements.txt") from exc
        with Path(path).open(encoding="utf-8") as stream:
            dataset = yaml.safe_load(stream).get("dataset", {})
        return cls(width_m=float(dataset.get("map_width_m", cls.width_m)),
                   height_m=float(dataset.get("map_height_m", cls.height_m)),
                   resolution_m=float(dataset.get("map_resolution_m", cls.resolution_m)),
                   origin_x=float(dataset.get("origin_x", cls.origin_x)),
                   origin_y=float(dataset.get("origin_y", cls.origin_y)))


class SemanticMapper:
    """在一个 Episode 内持续累积观测的世界坐标语义地图。"""

    def __init__(self, config: MapConfig = MapConfig()):
        self.config = config
        self.reset()

    @property
    def shape(self):
        """返回 ``[C, H, W]`` 地图形状。"""
        return (NUM_MAP_CHANNELS, round(self.config.height_m / self.config.resolution_m),
                round(self.config.width_m / self.config.resolution_m))

    def reset(self):
        """开始新 Episode，清空历史探索和访问记录。"""
        self.map = np.zeros(self.shape, dtype=np.float32)

    def _grid(self, xy):
        """将世界 XY 坐标转换为网格索引，并丢弃越界点。"""
        gx = np.floor((xy[:, 0] - self.config.origin_x) / self.config.resolution_m).astype(int)
        gy = np.floor((xy[:, 1] - self.config.origin_y) / self.config.resolution_m).astype(int)
        valid = (gx >= 0) & (gx < self.shape[2]) & (gy >= 0) & (gy < self.shape[1])
        return gx[valid], gy[valid]

    def update(self, depth, intrinsics: CameraIntrinsics, T_base_camera, T_world_base,
               detections=(), robot_xy=(0.0, 0.0), visited_radius_m=0.12):
        """融合一帧 RGB-D、位姿和检测结果，返回更新后的地图副本。

        ``robot`` 通道只表示当前时刻的位置，因此每帧更新前清空；
        ``visited`` 通道表示历史访问区域，因此在整个 Episode 内持续累积。
        """
        # Channel 2 是当前机器人位置，不能保留上一帧的位置。
        # Channel 3 是历史访问区域，不能在这里清空。
        self.map[2].fill(0.0)
        points, pixels = depth_to_world(depth, intrinsics, T_base_camera, T_world_base)
        gx, gy = self._grid(points[:, :2])
        self.map[1, gy, gx] = 1.0   #标记障碍物通道，1表示有障碍物
        # A projected depth return is conservative obstacle evidence. Semantic
        # detector boxes below overwrite it with class-specific evidence.
        self.map[0, gy, gx] = 1.0   #标记探索通道，1表示已探索
        self._mark_robot(robot_xy, visited_radius_m)
        for det in detections:  
            self._mark_detection(det, pixels, points)
        return self.map.copy()  #返回地图副本，避免外部修改原始地图

    def _mark_robot(self, xy, radius):
        """标记当前位置，并将当前网格加入历史 visited 通道。

        当前地图分辨率为 0.5 m，而访问半径小于一个地图网格，因此
        采用离散地图中最小的表示：只标记机器人所在的一个网格，
        不再把周围区域膨胀成 3x3 网格。``radius`` 参数保留是为了
        兼容旧调用接口，但在单网格方案中不参与计算。
        """
        # 将机器人当前位置转换为地图网格索引。
        gx, gy = self._grid(np.asarray([[xy[0], xy[1]]], dtype=np.float32))
        if len(gx):
            # Channel 2：当前位置；update() 已在调用本函数前清空旧位置。
            self.map[2, gy[0], gx[0]] = 1.0
            # Channel 3：历史访问区域；不清空，在 Episode 内持续累积。
            self.map[3, gy[0], gx[0]] = 1.0

    def _mark_detection(self, det, pixels, points):
        """将检测框内的深度点写入对应语义通道。"""
        x1, y1, x2, y2 = det.bbox
        keep = (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
        if not keep.any():
            return
        gx, gy = self._grid(points[keep, :2])
        channel = {"echinus": 4, "holothurian": 4, "rock": 5, "sand": 6}.get(det.class_name)
        if channel is not None:
            self.map[channel, gy, gx] = np.maximum(self.map[channel, gy, gx], det.score)    #标记语义通道，使用最大置信度更新
            self.map[0, gy, gx] = np.maximum(self.map[0, gy, gx], det.score)
