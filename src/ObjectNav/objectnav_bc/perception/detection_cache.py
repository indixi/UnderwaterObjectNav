"""Serializable low-threshold GFL detection cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .semantic_detector import Detection


CACHE_SCHEMA_VERSION = 1


def _file_signature(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "sha256": digest,
    }


def detector_signature(config) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "config": _file_signature(config.config_path),
        "checkpoint": _file_signature(config.checkpoint_path),
        "classes": list(config.classes),
        "candidate_score_threshold": config.candidate_score_threshold,
    }


def serialize_detections(
    episode_name: str,
    step_id: int,
    rgb_path: str,
    detections: list[Detection],
    class_names: tuple[str, ...] = ("echinus", "rock"),
) -> dict:
    class_to_id = {name: index for index, name in enumerate(class_names)}
    unknown = sorted(
        {item.class_name for item in detections} - set(class_to_id))
    if unknown:
        raise ValueError(f"Unknown detector class names: {unknown}")
    return {
        "episode_id": episode_name,
        "step_id": int(step_id),
        "rgb_path": rgb_path.replace("\\", "/"),
        "boxes_xyxy": [list(item.bbox) for item in detections],
        "scores": [item.score for item in detections],
        "labels": [class_to_id[item.class_name] for item in detections],
        "class_names": [item.class_name for item in detections],
    }


def deserialize_detections(
    record: dict,
    class_names: tuple[str, ...] | None = None,
) -> list[Detection]:
    boxes = record.get("boxes_xyxy", ())
    scores = record.get("scores", ())
    names = record.get("class_names", ())
    if not (len(boxes) == len(scores) == len(names)):
        raise ValueError("detection cache arrays have different lengths")
    labels = record.get("labels")
    if labels is not None and len(labels) != len(names):
        raise ValueError("detection cache labels have a different length")
    if labels is not None and class_names is not None:
        for label, name in zip(labels, names):
            label = int(label)
            if label < 0 or label >= len(class_names):
                raise ValueError(f"Detection cache label is out of range: {label}")
            if class_names[label] != name:
                raise ValueError(
                    "Detection cache label/name mismatch: "
                    f"{label} maps to {class_names[label]!r}, not {name!r}"
                )
    return [
        Detection(tuple(map(float, box)), float(score), str(name))
        for box, score, name in zip(boxes, scores, names)
    ]


def load_detection_cache(path: str | Path) -> dict[tuple[str, int], dict]:
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["episode_id"]), int(record["step_id"]))
            if key in records:
                raise ValueError(f"duplicate detection cache key: {key}")
            records[key] = record
    return records


def validate_detection_cache_metadata(path: str | Path, detector_config) -> dict:
    """Ensure a raw cache uses the configured class order and score floor."""
    records_path = Path(path).resolve()
    metadata_path = records_path.parent / "detection_cache_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Detection cache metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    signature = metadata.get("detector_signature", {})
    cached_classes = tuple(signature.get("classes", ()))
    if cached_classes != tuple(detector_config.classes):
        raise ValueError(
            "Detection cache class order mismatch: "
            f"cache={cached_classes}, configured={detector_config.classes}"
        )
    for artifact_name, configured_path in (
        ("config", detector_config.config_path),
        ("checkpoint", detector_config.checkpoint_path),
    ):
        cached_artifact = signature.get(artifact_name, {})
        if configured_path is None:
            raise ValueError(f"Configured detector {artifact_name} path is missing")
        if cached_artifact.get("name") != configured_path.name:
            raise ValueError(
                f"Detection cache {artifact_name} mismatch: "
                f"cache={cached_artifact.get('name')!r}, "
                f"configured={configured_path.name!r}"
            )
        if configured_path.is_file():
            current = _file_signature(configured_path)
            if current != cached_artifact:
                raise ValueError(
                    f"Detection cache was built with a different {artifact_name}"
                )
    cached_floor = float(signature.get("candidate_score_threshold", 1.0))
    thresholds = detector_config.resolved_score_thresholds()
    if cached_floor > min(thresholds.values()):
        raise ValueError(
            "Detection cache candidate threshold is above a deployment "
            "threshold and may have discarded required detections"
        )
    return metadata
