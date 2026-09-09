"""Read the episode layout produced by the ROS data_generate package."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from ..perception.depth_projection import rotation_matrix_from_rpy


REQUIRED_TRAJECTORY_FIELDS = {
    "step_id",
    "action_id",
    "action_start_time",
    "timestamp",
    "rgb_path",
    "depth_path",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "goal_category",
    "expert_action",
}


def episode_metadata(episode_dir: Path) -> dict:
    path = episode_dir / "episode.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"episode metadata missing: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"episode metadata must be a mapping: {path}")
    return document


def episode_directories(dataset_root: Path, include_failed: bool) -> list[Path]:
    directories = sorted(
        path
        for path in Path(dataset_root).glob("episode_*")
        if path.is_dir() and not path.name.startswith("episode_false")
    )
    selected = []
    for directory in directories:
        metadata = episode_metadata(directory)
        if include_failed or metadata.get("success") is True:
            selected.append(directory)
    return selected


def trajectory_rows(episode_dir: Path) -> list[dict[str, str]]:
    path = episode_dir / "trajectory.csv"
    if not path.is_file():
        raise FileNotFoundError(f"trajectory missing: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_TRAJECTORY_FIELDS - fields
        if missing:
            raise ValueError(
                f"trajectory missing fields {sorted(missing)}: {path}")
        rows = list(reader)
    expected_steps = list(range(len(rows)))
    actual_steps = [int(row["step_id"]) for row in rows]
    if actual_steps != expected_steps:
        raise ValueError(f"non-contiguous step_id in {path}")
    return rows


def pose_matrix_from_row(row: dict[str, str]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_matrix_from_rpy(
        float(row["roll"]), float(row["pitch"]), float(row["yaw"])
    )
    transform[:3, 3] = [
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
    ]
    return transform
