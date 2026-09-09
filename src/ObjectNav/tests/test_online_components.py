import csv
import json
import socket

import numpy as np
import yaml

from objectnav_bc.online.ipc_protocol import receive_message, send_message
from objectnav_bc.online.online_recorder import OnlineEpisodeRecorder


def test_ipc_protocol_round_trip_preserves_arrays():
    sender, receiver = socket.socketpair()
    try:
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        depth = np.arange(12, dtype=np.float32).reshape(3, 4)
        send_message(
            sender,
            {"type": "INFER", "request_id": 7},
            {"rgb": rgb, "depth": depth},
        )
        payload, arrays = receive_message(receiver)
    finally:
        sender.close()
        receiver.close()

    assert payload["type"] == "INFER"
    assert payload["request_id"] == 7
    np.testing.assert_array_equal(arrays["rgb"], rgb)
    np.testing.assert_array_equal(arrays["depth"], depth)


def test_online_recorder_writes_complete_policy_episode(tmp_path):
    bc_config = tmp_path / "bc.yaml"
    online_config = tmp_path / "online.yaml"
    bc_config.write_text("camera: {}\n", encoding="utf-8")
    online_config.write_text("online: {}\n", encoding="utf-8")
    recorder = OnlineEpisodeRecorder(
        tmp_path / "runs",
        "rock_seaurchin",
        tmp_path / "best_policy.pt",
        bc_config,
        online_config,
    )
    try:
        episode_id, episode_path = recorder.start(
            "echinus",
            10.0,
            {"x": 0.0, "y": 0.0, "z": 3.32, "yaw": 0.0},
        )
        result = {
            "probabilities": [0.1, 0.2, 0.3, 0.4],
            "target": {
                "visible": 1.0,
                "bearing_rad": 0.1,
                "distance_m": 0.8,
            },
            "sensor_world_z_m": 3.3,
            "inference_time_s": 0.05,
            "detections": [
                {
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.9,
                    "class_name": "echinus",
                }
            ],
        }
        recorder.record_decision(
            0,
            1,
            11.0,
            11.01,
            11.06,
            11.07,
            {
                "x": 1.0,
                "y": 2.0,
                "z": 3.32,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.5,
            },
            "STOP",
            result,
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.ones((4, 5), dtype=np.float32),
            np.zeros((6, 8, 14), dtype=np.float16),
            np.zeros((4, 6, 6), dtype=np.float16),
        )
        recorder.update_action_status(1, "STOPPED", 11.08, 11.09)
        errors = recorder.finish(True, None, 11.1)
    finally:
        recorder.shutdown()

    assert episode_id == 1
    episode_path = tmp_path / "runs" / "episode_0001"
    assert not errors
    assert (episode_path / "rgb" / "000000.png").is_file()
    assert (episode_path / "depth" / "000000.npy").is_file()
    assert (episode_path / "global_map" / "000000.npy").is_file()
    assert (episode_path / "local_map" / "000000.npy").is_file()
    with (episode_path / "trajectory.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["policy_action"] == "STOP"
    assert rows[0]["action_state"] == "STOPPED"
    policy = json.loads(
        (episode_path / "policy.jsonl").read_text(encoding="utf-8"))
    assert policy["probabilities"] == [0.1, 0.2, 0.3, 0.4]
    metadata = yaml.safe_load(
        (episode_path / "episode.yaml").read_text(encoding="utf-8"))
    assert metadata["success"] is True
    assert metadata["source"] == "policy"
