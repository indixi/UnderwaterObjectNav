"""Asynchronous recorder for policy-generated online inspection runs."""

import csv
import json
import os
from pathlib import Path
import queue
import shutil
import threading

import cv2
import numpy as np
import yaml


TRAJECTORY_FIELDS = (
    "step_id",
    "action_id",
    "observation_time",
    "inference_start_time",
    "inference_finish_time",
    "action_request_time",
    "action_start_time",
    "action_finish_time",
    "action_state",
    "rgb_path",
    "depth_path",
    "global_map_path",
    "local_map_path",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "goal_category",
    "policy_action",
)


class OnlineEpisodeRecorder:
    def __init__(self, root, scene_id, checkpoint, bc_config, online_config):
        self.root = Path(root).expanduser().resolve()
        self.scene_id = str(scene_id)
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.bc_config = str(Path(bc_config).expanduser().resolve())
        self.online_config = str(Path(online_config).expanduser().resolve())
        self.work_queue = queue.Queue()
        self.errors = []
        self.lock = threading.RLock()
        self.worker = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker.start()
        self.reset_state()

    def reset_state(self):
        self.episode_id = None
        self.episode_dir = None
        self.goal_category = None
        self.start_time = None
        self.start_pose = None
        self.rows = []
        self.policy_records = []
        self.detection_records = []
        self.row_by_action_id = {}

    @property
    def active(self):
        return self.episode_dir is not None

    def _next_episode_id(self):
        self.root.mkdir(parents=True, exist_ok=True)
        numbers = []
        for path in self.root.glob("episode_*"):
            suffix = path.name.split("_", 1)[-1]
            if path.is_dir() and suffix.isdigit():
                numbers.append(int(suffix))
        return max(numbers, default=0) + 1

    def start(self, goal_category, start_time, start_pose):
        with self.lock:
            if self.episode_dir is not None:
                raise RuntimeError("an online recording episode is already active")
            self.errors = []
            try:
                self.episode_id = self._next_episode_id()
                self.episode_dir = self.root / ("episode_%04d" % self.episode_id)
                for name in ("rgb", "depth", "global_map", "local_map"):
                    (self.episode_dir / name).mkdir(parents=True, exist_ok=False)
                self.goal_category = str(goal_category)
                self.start_time = float(start_time)
                self.start_pose = dict(start_pose)
                for source, target in (
                    (self.bc_config, "bc.yaml"),
                    (self.online_config, "online.yaml"),
                ):
                    if os.path.isfile(source):
                        shutil.copy2(source, self.episode_dir / target)
                self._write_episode_yaml(None, None, None)
                return self.episode_id, str(self.episode_dir)
            except Exception:
                # Keep a partially-created directory as diagnostics, but allow
                # navigation to continue without an active recorder.
                self.reset_state()
                raise

    def record_decision(
        self,
        step_id,
        action_id,
        observation_time,
        inference_start_time,
        inference_finish_time,
        action_request_time,
        pose,
        action,
        result,
        rgb,
        depth,
        global_map,
        local_map,
    ):
        with self.lock:
            if self.episode_dir is None:
                raise RuntimeError("no active online recording episode")
            index = "%06d" % int(step_id)
            paths = {
                "rgb_path": "rgb/%s.png" % index,
                "depth_path": "depth/%s.npy" % index,
                "global_map_path": "global_map/%s.npy" % index,
                "local_map_path": "local_map/%s.npy" % index,
            }
            row = {
                "step_id": int(step_id),
                "action_id": int(action_id),
                "observation_time": float(observation_time),
                "inference_start_time": float(inference_start_time),
                "inference_finish_time": float(inference_finish_time),
                "action_request_time": float(action_request_time),
                "action_start_time": "",
                "action_finish_time": "",
                "action_state": "REQUESTED",
                **paths,
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "z": float(pose["z"]),
                "roll": float(pose["roll"]),
                "pitch": float(pose["pitch"]),
                "yaw": float(pose["yaw"]),
                "goal_category": self.goal_category,
                "policy_action": str(action),
            }
            self.rows.append(row)
            self.row_by_action_id[int(action_id)] = row
            self.policy_records.append(
                {
                    "step_id": int(step_id),
                    "action_id": int(action_id),
                    "probabilities": result["probabilities"],
                    "action": str(action),
                    "target": result["target"],
                    "sensor_world_z_m": result["sensor_world_z_m"],
                    "inference_time_s": result["inference_time_s"],
                }
            )
            self.detection_records.append(
                {
                    "step_id": int(step_id),
                    "action_id": int(action_id),
                    "detections": result["detections"],
                }
            )
            self.work_queue.put(
                (
                    self.episode_dir,
                    paths,
                    np.array(rgb, copy=True),
                    np.array(depth, dtype=np.float32, copy=True),
                    np.array(global_map, dtype=np.float16, copy=True),
                    np.array(local_map, dtype=np.float16, copy=True),
                )
            )

    def update_action_status(self, action_id, state, start_time, finish_time):
        with self.lock:
            row = self.row_by_action_id.get(int(action_id))
            if row is None:
                return
            row["action_state"] = str(state)
            if start_time:
                row["action_start_time"] = float(start_time)
            if finish_time:
                row["action_finish_time"] = float(finish_time)

    def _writer_loop(self):
        while True:
            item = self.work_queue.get()
            try:
                if item is None:
                    return
                episode_dir, paths, rgb, depth, global_map, local_map = item
                if not cv2.imwrite(str(episode_dir / paths["rgb_path"]), rgb):
                    raise IOError("OpenCV could not save RGB image")
                np.save(episode_dir / paths["depth_path"], depth)
                np.save(episode_dir / paths["global_map_path"], global_map)
                np.save(episode_dir / paths["local_map_path"], local_map)
            except Exception as exc:
                with self.lock:
                    self.errors.append("%s: %s" % (type(exc).__name__, exc))
            finally:
                self.work_queue.task_done()

    def _write_jsonl(self, name, records):
        with (self.episode_dir / name).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_episode_yaml(self, success, failure_reason, finish_time):
        document = {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "goal_category": self.goal_category,
            "source": "policy",
            "success": success,
            "failure_reason": failure_reason,
            "start_time": self.start_time,
            "finish_time": finish_time,
            "start_pose": self.start_pose,
            "step_count": len(self.rows),
            "checkpoint": self.checkpoint,
            "bc_config_snapshot": "bc.yaml",
            "online_config_snapshot": "online.yaml",
            "recording_errors": list(self.errors),
        }
        (self.episode_dir / "episode.yaml").write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def finish(self, success, failure_reason, finish_time):
        self.work_queue.join()
        with self.lock:
            if self.episode_dir is None:
                return []
            try:
                try:
                    with (self.episode_dir / "trajectory.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as stream:
                        writer = csv.DictWriter(
                            stream, fieldnames=TRAJECTORY_FIELDS)
                        writer.writeheader()
                        writer.writerows(self.rows)
                except Exception as exc:
                    self.errors.append(
                        "trajectory.csv: %s: %s" %
                        (type(exc).__name__, exc))
                for filename, records in (
                    ("policy.jsonl", self.policy_records),
                    ("detections.jsonl", self.detection_records),
                ):
                    try:
                        self._write_jsonl(filename, records)
                    except Exception as exc:
                        self.errors.append(
                            "%s: %s: %s" %
                            (filename, type(exc).__name__, exc))
                try:
                    self._write_episode_yaml(
                        bool(success), failure_reason or None,
                        float(finish_time))
                except Exception as exc:
                    self.errors.append(
                        "episode.yaml: %s: %s" %
                        (type(exc).__name__, exc))
                return list(self.errors)
            finally:
                self.reset_state()

    def shutdown(self):
        self.work_queue.join()
        self.work_queue.put(None)
        self.worker.join(timeout=2.0)
