"""Extract sparse collision-height geometry and explored-ray endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .depth_projection import (
    CameraIntrinsics,
    pixels_depth_to_world,
    sample_depth,
)


@dataclass(frozen=True)
class DepthGeometry:
    obstacle_points: np.ndarray
    explored_endpoints: np.ndarray
    explored_pixels: np.ndarray
    sensor_world_xyz: np.ndarray


class DepthObstacleExtractor:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        camera_config: dict,
        depth_config: dict,
    ):
        self.intrinsics = intrinsics
        self.camera_config = camera_config
        self.depth_config = depth_config
        self._height_warning_emitted = False

    def extract(self, depth, T_base_camera, T_world_base) -> DepthGeometry:
        depth = np.asarray(depth)
        self.intrinsics.validate_image(depth, "depth")
        pixels, depths = sample_depth(
            depth,
            int(self.depth_config["obstacle_depth_stride"]),
            float(self.depth_config["min_depth_m"]),
            float(self.depth_config["max_depth_m"]),
        )
        transform = (
            np.asarray(T_world_base, dtype=np.float32)
            @ np.asarray(T_base_camera, dtype=np.float32)
        )
        sensor_xyz = transform[:3, 3].copy()
        nominal = float(self.camera_config["nominal_sensor_world_z_m"])
        tolerance = float(self.camera_config["sensor_height_tolerance_m"])
        if (
            not self._height_warning_emitted
            and abs(float(sensor_xyz[2]) - nominal) > tolerance
        ):
            warnings.warn(
                f"sensor world z={sensor_xyz[2]:.3f} m differs from nominal "
                f"{nominal:.3f} m by more than {tolerance:.3f} m",
                RuntimeWarning,
            )
            self._height_warning_emitted = True

        points = pixels_depth_to_world(
            pixels,
            depths,
            self.intrinsics,
            T_base_camera,
            T_world_base,
        )
        min_z = sensor_xyz[2] + float(
            self.depth_config["obstacle_z_min_offset_from_sensor_m"])
        max_z = sensor_xyz[2] + float(
            self.depth_config["obstacle_z_max_offset_from_sensor_m"])
        collision_height = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
        return DepthGeometry(
            obstacle_points=points[collision_height],
            explored_endpoints=points,
            explored_pixels=pixels,
            sensor_world_xyz=sensor_xyz,
        )
