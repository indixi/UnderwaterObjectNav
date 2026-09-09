"""Conda-side GFL + mapping + BC inference worker without ROS imports."""

import argparse
import socket
import time
import traceback

import numpy as np

from ..config_loader import load_bc_config
from ..constants import ACTION_NAMES, GLOBAL_MAP_CHANNELS, LOCAL_MAP_CHANNELS
from ..eval.eval_closed_loop import ClosedLoopPolicy
from ..mapping import SemanticMappingPipeline
from ..perception.depth_projection import (
    CameraIntrinsics,
    camera_transform_from_mapping,
    rotation_matrix_from_rpy,
)
from ..perception.semantic_detector import build_semantic_detector
from .ipc_protocol import receive_message, send_message


class OnlineInferenceEngine:
    def __init__(self, bc_config, checkpoint, device):
        self.runtime = load_bc_config(bc_config, validate_detector=True)
        self.detector = build_semantic_detector(
            self.runtime.detector, device=device)
        self.policy = ClosedLoopPolicy(checkpoint, device=device)
        self.mapper = SemanticMappingPipeline(
            self.runtime,
            CameraIntrinsics.from_mapping(self.runtime.camera),
            camera_transform_from_mapping(self.runtime.camera),
        )
        self.goal_category = None
        self._validate_contracts()

    def _validate_contracts(self):
        state = self.policy.state
        if tuple(self.runtime.detector.classes) != ("echinus", "rock"):
            raise ValueError(
                "GFL class order must be 0=echinus, 1=rock")
        if tuple(state["action_names"]) != ACTION_NAMES:
            raise ValueError("BC checkpoint action contract is incompatible")
        if tuple(state["global_map_channels"]) != GLOBAL_MAP_CHANNELS:
            raise ValueError("BC checkpoint Global Map channels are incompatible")
        if tuple(state["local_map_channels"]) != LOCAL_MAP_CHANNELS:
            raise ValueError("BC checkpoint Local Map channels are incompatible")
        if tuple(state["goal_names"]) != ("echinus",):
            raise ValueError("BC checkpoint goal contract is incompatible")
        model = state["model_config"]
        if int(model["global_map_channels"]) != len(GLOBAL_MAP_CHANNELS):
            raise ValueError("BC checkpoint Global Map channel count is invalid")
        if int(model["local_map_channels"]) != len(LOCAL_MAP_CHANNELS):
            raise ValueError("BC checkpoint Local Map channel count is invalid")

    def start_episode(self, goal_category):
        if goal_category != "echinus":
            raise ValueError("unsupported goal_category %r" % goal_category)
        self.mapper.reset()
        self.goal_category = goal_category

    def reset(self):
        self.mapper.reset()
        self.goal_category = None

    @staticmethod
    def _pose_matrix(pose):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation_matrix_from_rpy(
            float(pose["roll"]),
            float(pose["pitch"]),
            float(pose["yaw"]),
        )
        transform[:3, 3] = [
            float(pose["x"]),
            float(pose["y"]),
            float(pose["z"]),
        ]
        return transform

    def infer(self, payload, arrays):
        if self.goal_category is None:
            raise RuntimeError("START_EPISODE is required before INFER")
        rgb = np.asarray(arrays["rgb"])
        depth = np.asarray(arrays["depth"], dtype=np.float32)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be HxWx3 uint8")
        expected = (int(self.runtime.camera["height"]),
                    int(self.runtime.camera["width"]))
        if rgb.shape[:2] != expected or depth.shape != expected:
            raise ValueError(
                "RGB-D shape mismatch: rgb=%r depth=%r expected=%r" %
                (rgb.shape, depth.shape, expected))

        start = time.perf_counter()
        detections = self.detector.detect(rgb)
        pose = payload["pose"]
        yaw = float(pose["yaw"])
        mapping = self.mapper.update(
            depth,
            self._pose_matrix(pose),
            detections,
            (float(pose["x"]), float(pose["y"])),
            yaw,
        )
        prediction = self.policy.predict(
            mapping.global_map,
            mapping.local_map,
            mapping.target.visible,
            mapping.target.bearing_rad,
            mapping.target.distance_m,
            goal_category=self.goal_category,
            yaw=yaw,
        )
        elapsed = time.perf_counter() - start
        response = {
            "type": "INFER_RESULT",
            "request_id": int(payload["request_id"]),
            "ok": True,
            "action_id": int(prediction["action_id"]),
            "action": prediction["action"],
            "probabilities": [float(value) for value in prediction["probabilities"]],
            "detections": [
                {
                    "bbox": [float(value) for value in item.bbox],
                    "score": float(item.score),
                    "class_name": item.class_name,
                }
                for item in detections
            ],
            "target": {
                "visible": float(mapping.target.visible),
                "bearing_rad": float(mapping.target.bearing_rad),
                "distance_m": float(mapping.target.distance_m),
            },
            "sensor_world_z_m": float(mapping.sensor_world_z_m),
            "inference_time_s": float(elapsed),
            "robot_pose": {
                name: float(pose[name])
                for name in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            "global_map_spec": {
                name: float(self.runtime.map["global"][name])
                for name in (
                    "origin_x", "origin_y", "width_m", "height_m",
                    "resolution_m",
                )
            },
        }
        return response, {
            "global_map": mapping.global_map.astype(np.float16),
            "local_map": mapping.local_map.astype(np.float16),
        }


def _serve_connection(connection, engine):
    while True:
        payload, arrays = receive_message(connection)
        message_type = payload.get("type")
        try:
            if message_type == "HEALTH":
                send_message(connection, {"type": "READY", "ok": True})
            elif message_type == "START_EPISODE":
                engine.start_episode(str(payload["goal_category"]))
                send_message(connection, {"type": "STARTED", "ok": True})
            elif message_type == "RESET":
                engine.reset()
                send_message(connection, {"type": "RESET_DONE", "ok": True})
            elif message_type == "INFER":
                response, result_arrays = engine.infer(payload, arrays)
                send_message(connection, response, result_arrays)
            elif message_type == "SHUTDOWN":
                send_message(connection, {"type": "SHUTDOWN_DONE", "ok": True})
                return False
            else:
                raise ValueError("unknown message type %r" % message_type)
        except Exception as exc:
            send_message(
                connection,
                {
                    "type": "ERROR",
                    "ok": False,
                    "request_id": payload.get("request_id"),
                    "error": "%s: %s" % (type(exc).__name__, exc),
                },
            )
            traceback.print_exc()
    return True


def run(args):
    print("Loading GFL and BC models...", flush=True)
    engine = OnlineInferenceEngine(args.bc_config, args.checkpoint, args.device)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(
        "ObjectNav inference worker READY on %s:%d" % (args.host, args.port),
        flush=True,
    )
    try:
        connection, address = server.accept()
        print("ROS client connected from %s:%d" % address, flush=True)
        with connection:
            _serve_connection(connection, engine)
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29500)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
