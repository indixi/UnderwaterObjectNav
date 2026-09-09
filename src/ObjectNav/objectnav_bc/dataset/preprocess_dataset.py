"""Build cached Global/Local maps and BC manifests from expert episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config_loader import load_bc_config
from ..constants import (
    ACTION_TO_ID,
    GLOBAL_MAP_CHANNELS,
    GOAL_TO_ID,
    LOCAL_MAP_CHANNELS,
)
from ..mapping import SemanticMappingPipeline
from ..perception.depth_projection import (
    CameraIntrinsics,
    camera_transform_from_mapping,
)
from ..perception.detection_cache import (
    deserialize_detections,
    load_detection_cache,
    validate_detection_cache_metadata,
)
from ..perception.semantic_detector import filter_detections
from .episode_io import (
    episode_directories,
    pose_matrix_from_row,
    trajectory_rows,
)
from .splits import split_by_episode


def _serializable_runtime(runtime, thresholds):
    return {
        "camera": runtime.camera,
        "depth": runtime.depth,
        "map": runtime.map,
        "target": runtime.target,
        "model": runtime.model,
        "dataset": runtime.dataset,
        "detector": {
            "classes": list(runtime.detector.classes),
            "thresholds": thresholds,
            "checkpoint": (
                runtime.detector.checkpoint_path.name
                if runtime.detector.checkpoint_path
                else None
            ),
        },
    }


def build(dataset_root, output_root, detection_cache, runtime):
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validate_detection_cache_metadata(detection_cache, runtime.detector)
    cache = load_detection_cache(detection_cache)
    thresholds = runtime.detector.resolved_score_thresholds()
    intrinsics = CameraIntrinsics.from_mapping(runtime.camera)
    T_base_camera = camera_transform_from_mapping(runtime.camera)
    mapper = SemanticMappingPipeline(runtime, intrinsics, T_base_camera)
    storage_dtype = np.dtype(runtime.dataset["map_storage_dtype"])
    include_failed = bool(runtime.dataset["include_failed_episodes"])
    episodes = episode_directories(dataset_root, include_failed)
    all_records = []

    for episode_dir in episodes:
        mapper.reset()
        global_dir = output_root / episode_dir.name / "global_map"
        local_dir = output_root / episode_dir.name / "local_map"
        global_dir.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)

        for row in trajectory_rows(episode_dir):
            step = int(row["step_id"])
            key = (episode_dir.name, step)
            if key not in cache:
                raise KeyError(f"missing detection cache record: {key}")
            candidates = deserialize_detections(
                cache[key], runtime.detector.classes)
            detections = filter_detections(candidates, thresholds)
            depth = np.load(episode_dir / row["depth_path"])
            T_world_base = pose_matrix_from_row(row)
            robot_xy = (float(row["x"]), float(row["y"]))
            yaw = float(row["yaw"])
            output = mapper.update(
                depth, T_world_base, detections, robot_xy, yaw)

            global_file = global_dir / f"{step:06d}.npy"
            local_file = local_dir / f"{step:06d}.npy"
            np.save(global_file, output.global_map.astype(storage_dtype))
            np.save(local_file, output.local_map.astype(storage_dtype))

            goal_name = row["goal_category"].strip()
            if goal_name not in GOAL_TO_ID:
                raise ValueError(
                    f"unsupported goal_category {goal_name!r} in "
                    f"{episode_dir.name} step {step}"
                )
            action_name = row["expert_action"].strip().upper()
            if action_name not in ACTION_TO_ID:
                raise ValueError(
                    f"unsupported expert_action {action_name!r} in "
                    f"{episode_dir.name} step {step}"
                )
            all_records.append(
                {
                    "global_map": str(
                        global_file.relative_to(output_root)).replace("\\", "/"),
                    "local_map": str(
                        local_file.relative_to(output_root)).replace("\\", "/"),
                    "episode_id": episode_dir.name,
                    "step_id": step,
                    "timestamp": float(row["timestamp"]),
                    "goal_category": goal_name,
                    "goal_id": GOAL_TO_ID[goal_name],
                    "target_visible": output.target.visible,
                    "target_bearing_rad": output.target.bearing_rad,
                    "target_distance_m": output.target.distance_m,
                    "sensor_world_z_m": output.sensor_world_z_m,
                    "yaw": yaw,
                    "action": ACTION_TO_ID[action_name],
                }
            )

    groups = split_by_episode(
        all_records,
        seed=int(runtime.dataset["split_seed"]),
        ratios=(
            float(runtime.dataset["train_ratio"]),
            float(runtime.dataset["validation_ratio"]),
            float(runtime.dataset["test_ratio"]),
        ),
    )
    for name, records in zip(("train", "val", "test"), groups):
        with (output_root / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "records": len(all_records),
        "episodes": len(episodes),
        "episode_ids": [path.name for path in episodes],
        "global_map_channels": list(GLOBAL_MAP_CHANNELS),
        "local_map_channels": list(LOCAL_MAP_CHANNELS),
        "global_map_shape": list(mapper.global_map.shape),
        "local_map_shape": [
            len(LOCAL_MAP_CHANNELS),
            mapper.local_spec.height_cells,
            mapper.local_spec.width_cells,
        ],
        "map_storage_dtype": storage_dtype.name,
        "runtime": _serializable_runtime(runtime, thresholds),
        "split_counts": {
            name: len(records)
            for name, records in zip(("train", "val", "test"), groups)
        },
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"processed {len(all_records)} steps from {len(episodes)} successful "
        f"episodes into {output_root}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--detection-cache", required=True)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "bc.yaml"),
    )
    args = parser.parse_args()
    runtime = load_bc_config(args.config, validate_detector=False)
    build(
        args.dataset_root,
        args.output_root,
        args.detection_cache,
        runtime,
    )


if __name__ == "__main__":
    main()
