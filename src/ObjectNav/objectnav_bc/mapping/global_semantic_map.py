"""Persistent world-coordinate semantic/geometric map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .explored_visibility import explored_ray_points


GLOBAL_CHANNELS = (
    "obstacle",
    "explored",
    "visited",
    "robot",
    "echinus",
    "rock",
)
GLOBAL_CHANNEL_INDEX = {name: index for index, name in enumerate(GLOBAL_CHANNELS)}


def _world_xy(points):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"world points must be Nx2 or Nx3, got {points.shape}")
    return points[:, :2]


@dataclass(frozen=True)
class WorldMapSpec:
    origin_x: float
    origin_y: float
    width_m: float
    height_m: float
    resolution_m: float

    @classmethod
    def from_mapping(cls, mapping: dict) -> "WorldMapSpec":
        return cls(
            origin_x=float(mapping["origin_x"]),
            origin_y=float(mapping["origin_y"]),
            width_m=float(mapping["width_m"]),
            height_m=float(mapping["height_m"]),
            resolution_m=float(mapping["resolution_m"]),
        )

    @property
    def height_cells(self):
        return round(self.height_m / self.resolution_m)

    @property
    def width_cells(self):
        return round(self.width_m / self.resolution_m)


class GridMap:
    def __init__(self, channels: tuple[str, ...], spec: WorldMapSpec):
        self.channels = channels
        self.channel_index = {
            name: index for index, name in enumerate(channels)}
        self.spec = spec
        self.reset()

    @property
    def shape(self):
        return (len(self.channels), self.spec.height_cells, self.spec.width_cells)

    def reset(self):
        self.map = np.zeros(self.shape, dtype=np.float32)

    def grid_indices(self, xy):
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        gx = np.floor(
            (xy[:, 0] - self.spec.origin_x) / self.spec.resolution_m
        ).astype(np.int32)
        gy = np.floor(
            (xy[:, 1] - self.spec.origin_y) / self.spec.resolution_m
        ).astype(np.int32)
        valid = (
            (gx >= 0)
            & (gx < self.spec.width_cells)
            & (gy >= 0)
            & (gy < self.spec.height_cells)
        )
        return gx, gy, valid

    def mark_points(self, channel: str, points, values=1.0):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if not len(points):
            return
        gx, gy, valid = self.grid_indices(points)
        if np.isscalar(values):
            selected_values = np.full(valid.sum(), float(values), np.float32)
        else:
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if len(values) != len(points):
                raise ValueError("one map value is required for every point")
            selected_values = values[valid]
        np.maximum.at(
            self.map[self.channel_index[channel]],
            (gy[valid], gx[valid]),
            selected_values,
        )

    def mark_disks(self, channel: str, centers, radius_m: float, values=1.0):
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        if not len(centers):
            return
        values = (
            np.full(len(centers), float(values), np.float32)
            if np.isscalar(values)
            else np.asarray(values, dtype=np.float32).reshape(-1)
        )
        radius_cells = int(np.ceil(float(radius_m) / self.spec.resolution_m))
        offsets = []
        for row in range(-radius_cells, radius_cells + 1):
            for col in range(-radius_cells, radius_cells + 1):
                if np.hypot(row, col) * self.spec.resolution_m <= radius_m + 1e-6:
                    offsets.append((col, row))
        gx, gy, valid_centers = self.grid_indices(centers)
        channel_map = self.map[self.channel_index[channel]]
        for col_offset, row_offset in offsets or [(0, 0)]:
            x = gx + col_offset
            y = gy + row_offset
            valid = valid_centers & (x >= 0) & (x < self.spec.width_cells)
            valid &= (y >= 0) & (y < self.spec.height_cells)
            np.maximum.at(channel_map, (y[valid], x[valid]), values[valid])


class GlobalSemanticMap(GridMap):
    def __init__(self, spec: WorldMapSpec):
        super().__init__(GLOBAL_CHANNELS, spec)

    def update(
        self,
        obstacle_points,
        explored_endpoints,
        explored_pixels,
        sensor_xy,
        robot_xy,
        echinus_points,
        echinus_scores,
        rock_points,
        rock_scores,
        robot_radius_m,
        visited_radius_m,
        echinus_radius_m,
    ):
        self.map[GLOBAL_CHANNEL_INDEX["robot"]].fill(0.0)
        rays = explored_ray_points(
            sensor_xy,
            explored_endpoints,
            explored_pixels,
            self.spec.resolution_m,
        )
        self.mark_points("explored", rays)
        self.mark_points("obstacle", _world_xy(obstacle_points))
        self.mark_disks("robot", [robot_xy], robot_radius_m)
        self.mark_disks("visited", [robot_xy], visited_radius_m)
        self.mark_disks(
            "echinus",
            _world_xy(echinus_points),
            echinus_radius_m,
            echinus_scores,
        )
        self.mark_points(
            "rock", _world_xy(rock_points), rock_scores)
        return self.map.copy()
