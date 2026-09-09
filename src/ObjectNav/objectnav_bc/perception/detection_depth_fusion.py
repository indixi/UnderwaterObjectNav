"""Fuse filtered echinus/rock boxes with aligned depth."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .depth_projection import (
    CameraIntrinsics,
    pixels_depth_to_world,
    valid_depth_mask,
)
from .semantic_detector import Detection


@dataclass(frozen=True)
class TargetCue:
    visible: float = 0.0
    bearing_rad: float = 0.0
    distance_m: float = 0.0

    def normalized(self, bearing_scale_rad: float, distance_scale_m: float):
        if not self.visible:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(
            [
                1.0,
                np.clip(self.bearing_rad / bearing_scale_rad, -1.0, 1.0),
                np.clip(self.distance_m / distance_scale_m, 0.0, 1.0),
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class FusionResult:
    echinus_points: np.ndarray
    echinus_scores: np.ndarray
    rock_points: np.ndarray
    rock_scores: np.ndarray
    target: TargetCue


def _empty_points():
    return np.empty((0, 3), dtype=np.float32)


def _bbox_bounds(bbox, width: int, height: int):
    x1, y1, x2, y2 = map(float, bbox)
    left = max(0, min(width, int(math.floor(min(x1, x2)))))
    right = max(0, min(width, int(math.ceil(max(x1, x2)))))
    top = max(0, min(height, int(math.floor(min(y1, y2)))))
    bottom = max(0, min(height, int(math.ceil(max(y1, y2)))))
    return left, top, right, bottom


def _center_roi_bounds(bbox, width, height, ratio, minimum_size):
    x1, y1, x2, y2 = map(float, bbox)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    roi_width = max(float(minimum_size), abs(x2 - x1) * float(ratio))
    roi_height = max(float(minimum_size), abs(y2 - y1) * float(ratio))
    return _bbox_bounds(
        (
            center_x - roi_width / 2.0,
            center_y - roi_height / 2.0,
            center_x + roi_width / 2.0,
            center_y + roi_height / 2.0,
        ),
        width,
        height,
    )


class DetectionDepthFusion:
    def __init__(self, intrinsics: CameraIntrinsics, depth_config: dict):
        self.intrinsics = intrinsics
        self.config = depth_config
        self.min_depth = float(depth_config["min_depth_m"])
        self.max_depth = float(depth_config["max_depth_m"])

    def _roi_median(self, depth, bounds, minimum_valid):
        left, top, right, bottom = bounds
        values = np.asarray(depth[top:bottom, left:right], dtype=np.float32)
        valid = values[
            valid_depth_mask(values, self.min_depth, self.max_depth)]
        if valid.size < int(minimum_valid):
            return None
        return float(np.median(valid))

    def _echinus(self, depth, detection, T_base_camera, T_world_base):
        height, width = depth.shape
        bounds = _center_roi_bounds(
            detection.bbox,
            width,
            height,
            self.config["echinus_center_roi_ratio"],
            self.config["echinus_min_roi_size_px"],
        )
        distance = self._roi_median(
            depth, bounds, self.config["echinus_min_valid_depth_pixels"])
        if distance is None:
            return None
        x1, y1, x2, y2 = detection.bbox
        u = float(np.clip((x1 + x2) / 2.0, 0, width - 1))
        v = float(np.clip((y1 + y2) / 2.0, 0, height - 1))
        point = pixels_depth_to_world(
            np.asarray([[u, v]], dtype=np.float32),
            np.asarray([distance], dtype=np.float32),
            self.intrinsics,
            T_base_camera,
            T_world_base,
        )[0]
        bearing = math.atan((u - self.intrinsics.cx) / self.intrinsics.fx)
        return point, distance, bearing

    def _rock(self, depth, detection, T_base_camera, T_world_base):
        height, width = depth.shape
        center_bounds = _center_roi_bounds(
            detection.bbox,
            width,
            height,
            self.config["rock_center_roi_ratio"],
            self.config.get("echinus_min_roi_size_px", 5),
        )
        reference = self._roi_median(
            depth, center_bounds, self.config["rock_min_valid_depth_pixels"])
        if reference is None:
            return _empty_points()

        left, top, right, bottom = _bbox_bounds(
            detection.bbox, width, height)
        stride = int(self.config["rock_depth_stride"])
        rows, cols = np.mgrid[top:bottom:stride, left:right:stride]
        if rows.size == 0:
            return _empty_points()
        values = depth[top:bottom:stride, left:right:stride].astype(np.float32)
        valid = valid_depth_mask(values, self.min_depth, self.max_depth)
        valid &= np.abs(values - reference) < float(
            self.config["rock_depth_gate_m"])
        if not valid.any():
            return _empty_points()
        pixels = np.column_stack((cols[valid], rows[valid])).astype(np.float32)
        return pixels_depth_to_world(
            pixels,
            values[valid],
            self.intrinsics,
            T_base_camera,
            T_world_base,
        )

    def process(
        self,
        depth: np.ndarray,
        detections: list[Detection],
        T_base_camera: np.ndarray,
        T_world_base: np.ndarray,
    ) -> FusionResult:
        depth = np.asarray(depth)
        self.intrinsics.validate_image(depth, "depth")
        echinus_points, echinus_scores, targets = [], [], []
        rock_points, rock_scores = [], []

        for detection in detections:
            if detection.class_name == "echinus":
                estimate = self._echinus(
                    depth, detection, T_base_camera, T_world_base)
                if estimate is None:
                    continue
                point, distance, bearing = estimate
                echinus_points.append(point)
                echinus_scores.append(detection.score)
                targets.append((distance, bearing))
            elif detection.class_name == "rock":
                points = self._rock(
                    depth, detection, T_base_camera, T_world_base)
                if len(points):
                    rock_points.append(points)
                    rock_scores.append(
                        np.full(len(points), detection.score, dtype=np.float32))

        if targets:
            distance, bearing = min(targets, key=lambda item: item[0])
            target = TargetCue(1.0, bearing, distance)
        else:
            target = TargetCue()
        return FusionResult(
            np.asarray(echinus_points, dtype=np.float32).reshape(-1, 3),
            np.asarray(echinus_scores, dtype=np.float32),
            np.concatenate(rock_points, axis=0) if rock_points else _empty_points(),
            np.concatenate(rock_scores) if rock_scores else np.empty(0, np.float32),
            target,
        )

