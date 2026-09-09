"""Shared offline map-update pipeline."""

from dataclasses import dataclass

import numpy as np

from ..perception.depth_obstacle_extractor import DepthObstacleExtractor
from ..perception.detection_depth_fusion import DetectionDepthFusion, TargetCue
from .global_semantic_map import GlobalSemanticMap, WorldMapSpec
from .local_egocentric_map import LocalEvidenceMap, LocalMapSpec


@dataclass(frozen=True)
class MappingOutput:
    global_map: np.ndarray
    local_map: np.ndarray
    target: TargetCue
    sensor_world_z_m: float


class SemanticMappingPipeline:
    def __init__(self, runtime, intrinsics, T_base_camera):
        self.runtime = runtime
        self.intrinsics = intrinsics
        self.T_base_camera = np.asarray(T_base_camera, dtype=np.float32)
        self.global_spec = WorldMapSpec.from_mapping(runtime.map["global"])
        self.local_spec = LocalMapSpec.from_mapping(runtime.map["local"])
        self.global_map = GlobalSemanticMap(self.global_spec)
        self.local_evidence = LocalEvidenceMap(
            self.global_spec, self.local_spec.resolution_m)
        self.fusion = DetectionDepthFusion(intrinsics, runtime.depth)
        self.geometry = DepthObstacleExtractor(
            intrinsics, runtime.camera, runtime.depth)

    def reset(self):
        self.global_map.reset()
        self.local_evidence.reset()

    def update(self, depth, T_world_base, detections, robot_xy, yaw):
        geometry = self.geometry.extract(
            depth, self.T_base_camera, T_world_base)
        semantic = self.fusion.process(
            depth, detections, self.T_base_camera, T_world_base)
        map_config = self.runtime.map
        global_map = self.global_map.update(
            geometry.obstacle_points,
            geometry.explored_endpoints,
            geometry.explored_pixels,
            geometry.sensor_world_xyz[:2],
            robot_xy,
            semantic.echinus_points,
            semantic.echinus_scores,
            semantic.rock_points,
            semantic.rock_scores,
            float(map_config["robot_marker_radius_m"]),
            float(map_config["visited_radius_m"]),
            float(map_config["echinus_marker_radius_m"]),
        )
        self.local_evidence.update(
            geometry.obstacle_points,
            robot_xy,
            semantic.echinus_points,
            semantic.echinus_scores,
            semantic.rock_points,
            semantic.rock_scores,
            float(map_config["visited_radius_m"]),
            float(map_config["echinus_marker_radius_m"]),
        )
        local_map = self.local_evidence.render(robot_xy, yaw, self.local_spec)
        return MappingOutput(
            global_map,
            local_map,
            semantic.target,
            float(geometry.sensor_world_xyz[2]),
        )
