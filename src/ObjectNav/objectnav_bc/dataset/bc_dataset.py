"""Episode-level splitting and processed Behavior Cloning dataset."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import ACTION_TO_ID, GOAL_TO_ID
from .splits import split_by_episode


class BehaviorCloningDataset(Dataset):
    def __init__(self, manifest, root=None):
        manifest = Path(manifest).resolve()
        self.root = Path(root or manifest.parent).resolve()
        with manifest.open(encoding="utf-8") as stream:
            self.records = [
                json.loads(line) for line in stream if line.strip()]
        if not self.records:
            raise ValueError(f"No records in {manifest}")

        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"processed metadata missing: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target = self.metadata["runtime"]["target"]
        self.bearing_scale = float(target["bearing_scale_rad"])
        self.distance_scale = float(target["distance_scale_m"])
        self.global_shape = tuple(self.metadata["global_map_shape"])
        self.local_shape = tuple(self.metadata["local_map_shape"])

    def __len__(self):
        return len(self.records)

    def _load_map(self, record, key, expected_shape):
        path = Path(record[key])
        if not path.is_absolute():
            path = self.root / path
        value = np.load(path).astype(np.float32)
        if value.shape != expected_shape:
            raise ValueError(
                f"{key} shape {value.shape} != {expected_shape}: {path}")
        return torch.from_numpy(value)

    def __getitem__(self, index):
        record = self.records[index]
        action = record["action"]
        if isinstance(action, str):
            action = ACTION_TO_ID[action]
        goal = record.get("goal_id")
        if goal is None:
            goal_name = record.get("goal_category", "echinus")
            if goal_name not in GOAL_TO_ID:
                raise ValueError(f"unsupported goal_category: {goal_name!r}")
            goal = GOAL_TO_ID[goal_name]
        visible = float(record["target_visible"])
        target_cue = np.zeros(3, dtype=np.float32)
        if visible:
            target_cue[:] = (
                1.0,
                np.clip(
                    float(record["target_bearing_rad"]) / self.bearing_scale,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    float(record["target_distance_m"]) / self.distance_scale,
                    0.0,
                    1.0,
                ),
            )
        return {
            "global_map": self._load_map(
                record, "global_map", self.global_shape),
            "local_map": self._load_map(
                record, "local_map", self.local_shape),
            "goal_id": torch.tensor(goal, dtype=torch.long),
            "target_cue": torch.from_numpy(target_cue),
            "yaw": torch.tensor(float(record["yaw"]), dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.long),
            "episode_id": str(record["episode_id"]),
            "step_id": int(record["step_id"]),
        }


def class_counts(dataset):
    return Counter(
        int(record["action"])
        if isinstance(record["action"], int)
        else ACTION_TO_ID[record["action"]]
        for record in dataset.records
    )
