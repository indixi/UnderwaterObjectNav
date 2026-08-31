"""RGB-D 投影工具。

所有变换遵循训练计划的约定：
``P_world = T_world_base @ T_base_camera @ P_camera``。
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参，单位通常为像素。"""
    fx: float
    fy: float
    cx: float
    cy: float


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """使用 4x4 齐次变换矩阵批量变换 N 个三维点。"""
    points = np.asarray(points, dtype=np.float32)   #确保为Numpy数组，类型为float32
    if points.size == 0:        #如果没有点，则返回空数组
        return points.reshape(-1, 3)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1) #将三维点转换为齐次坐标
    return (np.asarray(transform, dtype=np.float32) @ homogeneous.T).T[:, :3]   #将变换应用于齐次坐标，并返回三维点，这里有转置是因为之前矩阵每个点是行向量，但是真正计算的时候需要时列向量


def depth_to_world(depth: np.ndarray, intrinsics: CameraIntrinsics,
                   T_base_camera: np.ndarray, T_world_base: np.ndarray,
                   stride: int = 4, min_depth: float = 0.05,
                   max_depth: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """把深度图转换为世界点，并保留每个点对应的原图像素坐标。

    返回值：
      points: [N, 3] 世界坐标点；
      pixels: [N, 2] 对应的 ``(u, v)``，供检测框筛选使用。
    ``stride`` 用于降采样，避免每个像素都参与地图更新。
    """
    depth = np.asarray(depth)   #统一成Numpy数组
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]   #v是行索引，u是列索引，生成一个网格，stride表示采样间隔，每隔stride个像素取一个点，横纵向都要间隔采样
    d = depth[::stride, ::stride].astype(np.float32)    #depth是二维数组，取出采样点的深度值，并转换为float32类型，所以u，v要添加depth.shape[0]、depth.shape[1]来提取最大值，但是depth不用；
    valid = np.isfinite(d) & (d >= min_depth) & (d <= max_depth)    #筛选出有效的深度值，要求深度值是有限的，并且在指定的最小和最大深度范围内
    u, v, d = u[valid].astype(np.float32), v[valid].astype(np.float32), d[valid]
    points = np.column_stack(((u - intrinsics.cx) * d / intrinsics.fx,
                              (v - intrinsics.cy) * d / intrinsics.fy, d))
    return transform_points(points, np.asarray(T_world_base) @ np.asarray(T_base_camera)), np.column_stack([u, v])
