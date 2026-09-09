"""Persistent fine world evidence and robot-centred egocentric sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .global_semantic_map import GridMap, WorldMapSpec, _world_xy


LOCAL_CHANNELS = ("obstacle", "rock", "echinus", "visited")


@dataclass(frozen=True)
class LocalMapSpec:
    width_m: float
    height_m: float
    resolution_m: float

    @classmethod
    def from_mapping(cls, mapping: dict) -> "LocalMapSpec":
        return cls(
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


class LocalEvidenceMap(GridMap):
    """Fine-resolution episode memory covering the configured world bounds."""

    def __init__(self, world_spec: WorldMapSpec, resolution_m: float):
        fine_spec = WorldMapSpec(
            world_spec.origin_x,
            world_spec.origin_y,
            world_spec.width_m,
            world_spec.height_m,
            float(resolution_m),
        )
        super().__init__(LOCAL_CHANNELS, fine_spec)

    def update(
        self,
        obstacle_points,
        robot_xy,
        echinus_points,
        echinus_scores,
        rock_points,
        rock_scores,
        visited_radius_m,
        echinus_radius_m,
    ):
        self.mark_points("obstacle", _world_xy(obstacle_points))
        self.mark_points(
            "rock", _world_xy(rock_points), rock_scores)
        self.mark_disks(
            "echinus",
            _world_xy(echinus_points),
            echinus_radius_m,
            echinus_scores,
        )
        self.mark_disks("visited", [robot_xy], visited_radius_m)

    def render(self, robot_xy, yaw: float, local_spec: LocalMapSpec):
        """Nearest-neighbour sample with forward at image top and right at right."""
        rows, cols = np.mgrid[
            0:local_spec.height_cells, 0:local_spec.width_cells]
        forward = (
            local_spec.height_cells / 2.0 - (rows + 0.5)
        ) * local_spec.resolution_m
        right = (
            (cols + 0.5) - local_spec.width_cells / 2.0
        ) * local_spec.resolution_m
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        world_x = float(robot_xy[0]) + cosine * forward - sine * right
        world_y = float(robot_xy[1]) + sine * forward + cosine * right
        points = np.column_stack((world_x.ravel(), world_y.ravel()))
        gx, gy, valid = self.grid_indices(points)
        output = np.zeros(
            (
                len(LOCAL_CHANNELS),
                local_spec.height_cells,
                local_spec.width_cells,
            ),
            dtype=np.float32,
        )
        flat = output.reshape(len(LOCAL_CHANNELS), -1)
        flat[:, valid] = self.map[:, gy[valid], gx[valid]]
        return output
