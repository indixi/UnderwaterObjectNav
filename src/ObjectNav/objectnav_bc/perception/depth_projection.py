"""Vectorized pinhole RGB-D projection utilities."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_mapping(cls, config: dict) -> "CameraIntrinsics":
        return cls(
            fx=float(config["fx"]),
            fy=float(config["fy"]),
            cx=float(config["cx"]),
            cy=float(config["cy"]),
            width=int(config["width"]),
            height=int(config["height"]),
        )

    def validate_image(self, image: np.ndarray, name: str = "image") -> None:
        if image.ndim < 2:
            raise ValueError(f"{name} must have at least two dimensions")
        actual = tuple(image.shape[:2])
        expected = (self.height, self.width)
        if None not in expected and actual != expected:
            raise ValueError(f"{name} shape {actual} != configured {expected}")


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a ZYX yaw-pitch-roll rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def transform_from_translation_rpy(translation, rpy) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_matrix_from_rpy(*map(float, rpy))
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def camera_transform_from_mapping(config: dict) -> np.ndarray:
    transform = config["T_base_camera"]
    return transform_from_translation_rpy(
        transform["translation_m"], transform["rpy_rad"])


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1, 3)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (np.asarray(transform, dtype=np.float32) @ homogeneous.T).T[:, :3]


def valid_depth_mask(depths, min_depth: float, max_depth: float) -> np.ndarray:
    depths = np.asarray(depths)
    return (
        np.isfinite(depths)
        & (depths >= float(min_depth))
        & (depths <= float(max_depth))
    )


def pixels_depth_to_camera(
    pixels: np.ndarray,
    depths: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if len(pixels) != len(depths):
        raise ValueError("pixels and depths must have the same length")
    u, v = pixels[:, 0], pixels[:, 1]
    return np.column_stack(
        (
            (u - intrinsics.cx) * depths / intrinsics.fx,
            (v - intrinsics.cy) * depths / intrinsics.fy,
            depths,
        )
    ).astype(np.float32)


def pixels_depth_to_world(
    pixels: np.ndarray,
    depths: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_base_camera: np.ndarray,
    T_world_base: np.ndarray,
) -> np.ndarray:
    camera_points = pixels_depth_to_camera(pixels, depths, intrinsics)
    return transform_points(
        camera_points,
        np.asarray(T_world_base, dtype=np.float32)
        @ np.asarray(T_base_camera, dtype=np.float32),
    )


def sample_depth(
    depth: np.ndarray,
    stride: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid sampled ``(u,v)`` pixels and their depth values."""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    if stride < 1:
        raise ValueError("stride must be at least one")
    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    values = depth[::stride, ::stride].astype(np.float32)
    valid = valid_depth_mask(values, min_depth, max_depth)
    pixels = np.column_stack((cols[valid], rows[valid])).astype(np.float32)
    return pixels, values[valid]


def depth_to_world(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_base_camera: np.ndarray,
    T_world_base: np.ndarray,
    stride: int = 4,
    min_depth: float = 0.05,
    max_depth: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper returning sampled world points and pixels."""
    pixels, depths = sample_depth(depth, stride, min_depth, max_depth)
    points = pixels_depth_to_world(
        pixels, depths, intrinsics, T_base_camera, T_world_base)
    return points, pixels
