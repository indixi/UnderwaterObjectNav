"""Load and validate the single offline ObjectNav YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectorConfig:
    enabled: bool
    config_path: Path | None
    checkpoint_path: Path | None
    score_threshold: float
    threshold_file: Path | None
    candidate_score_threshold: float = 0.001
    classes: tuple[str, ...] = ("echinus", "rock")

    def resolved_score_thresholds(self) -> dict[str, float]:
        """Return per-class thresholds, applying echinus calibration if set."""
        thresholds = {name: self.score_threshold for name in self.classes}
        if self.threshold_file is None:
            return thresholds
        if not self.threshold_file.is_file():
            raise FileNotFoundError(
                f"Detector threshold file not found: {self.threshold_file}")

        document = json.loads(self.threshold_file.read_text(encoding="utf-8"))
        target_class = str(document.get("target_class", "echinus"))
        if target_class not in thresholds:
            raise ValueError(
                f"threshold target_class {target_class!r} is not in "
                f"detector classes {self.classes}"
            )
        if "recommended_threshold" not in document:
            raise ValueError(
                "threshold file has no recommended_threshold: "
                f"{self.threshold_file}"
            )
        calibrated_checkpoint = document.get("checkpoint")
        if (
            calibrated_checkpoint
            and self.checkpoint_path is not None
            and Path(str(calibrated_checkpoint)).name != self.checkpoint_path.name
        ):
            raise ValueError(
                "threshold/checkpoint mismatch: "
                f"{Path(str(calibrated_checkpoint)).name!r} != "
                f"{self.checkpoint_path.name!r}"
            )
        thresholds[target_class] = float(document["recommended_threshold"])
        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"score threshold for {name!r} must be in [0, 1]")
        return thresholds

    def validate(self, require_threshold: bool = True) -> None:
        if not self.enabled:
            return
        if self.config_path is None or not self.config_path.is_file():
            raise FileNotFoundError(
                f"Detector config not found: {self.config_path}")
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Detector checkpoint not found: {self.checkpoint_path}")
        if require_threshold:
            self.resolved_score_thresholds()
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(
                "perception.detector.score_threshold must be in [0, 1]")
        if not 0.0 <= self.candidate_score_threshold <= 1.0:
            raise ValueError(
                "perception.detector.candidate_score_threshold must be in "
                "[0, 1]"
            )
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError(
                "perception.detector.classes must be non-empty and unique")


@dataclass(frozen=True)
class BCConfig:
    source_path: Path
    dataset: dict[str, Any]
    camera: dict[str, Any]
    depth: dict[str, Any]
    map: dict[str, Any]
    target: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    detector: DetectorConfig


def _resolve_path(yaml_dir: Path, value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (yaml_dir / path).resolve()


def _positive(section: dict, key: str) -> float:
    value = float(section[key])
    if value <= 0:
        raise ValueError(f"{key} must be greater than 0")
    return value


def _validate(document: dict[str, Any]) -> None:
    for name in ("dataset", "camera", "depth", "map", "target", "model"):
        if not isinstance(document.get(name), dict):
            raise ValueError(f"Configuration section missing: {name}")

    camera = document["camera"]
    for key in ("width", "height", "fx", "fy"):
        _positive(camera, key)
    _positive(camera, "nominal_sensor_world_z_m")
    _positive(camera, "sensor_height_tolerance_m")
    transform = camera.get("T_base_camera", {})
    if len(transform.get("translation_m", ())) != 3:
        raise ValueError("camera.T_base_camera.translation_m must have 3 values")
    if len(transform.get("rpy_rad", ())) != 3:
        raise ValueError("camera.T_base_camera.rpy_rad must have 3 values")

    depth = document["depth"]
    if not 0 < float(depth["min_depth_m"]) < float(depth["max_depth_m"]):
        raise ValueError("depth min/max range is invalid")
    for key in ("rock_depth_stride", "obstacle_depth_stride"):
        if int(depth[key]) < 1:
            raise ValueError(f"depth.{key} must be at least 1")
    if not (
        float(depth["obstacle_z_min_offset_from_sensor_m"])
        < float(depth["obstacle_z_max_offset_from_sensor_m"])
    ):
        raise ValueError("Obstacle Z-height range is invalid")
    for key in (
        "echinus_center_roi_ratio",
        "echinus_min_roi_size_px",
        "echinus_min_valid_depth_pixels",
        "rock_center_roi_ratio",
        "rock_depth_gate_m",
        "rock_min_valid_depth_pixels",
    ):
        _positive(depth, key)

    map_doc = document["map"]
    float(map_doc["global"]["origin_x"])
    float(map_doc["global"]["origin_y"])
    for map_name in ("global", "local"):
        section = map_doc[map_name]
        for key in ("width_m", "height_m", "resolution_m"):
            _positive(section, key)
        height_cells = float(section["height_m"]) / float(
            section["resolution_m"])
        width_cells = float(section["width_m"]) / float(
            section["resolution_m"])
        if abs(height_cells - round(height_cells)) > 1e-6 or abs(
            width_cells - round(width_cells)
        ) > 1e-6:
            raise ValueError(
                f"map.{map_name} dimensions must be divisible by resolution_m")
    for key in (
        "robot_marker_radius_m",
        "visited_radius_m",
        "echinus_marker_radius_m",
    ):
        _positive(map_doc, key)

    target = document["target"]
    _positive(target, "bearing_scale_rad")
    _positive(target, "distance_scale_m")

    model = document["model"]
    if int(model["global_map_channels"]) != 6:
        raise ValueError("model.global_map_channels must match the 6 map channels")
    if int(model["local_map_channels"]) != 4:
        raise ValueError("model.local_map_channels must match the 4 map channels")
    for key in (
        "global_feature_dim",
        "local_feature_dim",
        "goal_embedding_dim",
        "target_feature_dim",
    ):
        _positive(model, key)
    if int(model["num_actions"]) != 4:
        raise ValueError("model.num_actions must match the 4-action contract")
    for key in ("global_pool_size", "local_pool_size"):
        values = model[key]
        if len(values) != 2 or any(int(value) <= 0 for value in values):
            raise ValueError(f"model.{key} must contain two positive integers")

    ratios = [
        float(document["dataset"][key])
        for key in ("train_ratio", "validation_ratio", "test_ratio")
    ]
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(
            "dataset train/validation/test ratios must be non-negative and "
            "sum to 1"
        )
    if document["dataset"].get("map_storage_dtype") not in (
        "float16",
        "float32",
    ):
        raise ValueError(
            "dataset.map_storage_dtype supports only float16/float32")


def load_bc_config(
    path: str | Path,
    validate_detector: bool = True,
    require_threshold: bool = True,
) -> BCConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read the configuration") from exc

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"BC configuration file not found: {source}")
    with source.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"BC configuration root must be a mapping: {source}")
    _validate(document)

    detector_doc = document.get("perception", {}).get("detector", {})
    detector = DetectorConfig(
        enabled=bool(detector_doc.get("enabled", False)),
        config_path=_resolve_path(source.parent, detector_doc.get("config_path")),
        checkpoint_path=_resolve_path(
            source.parent, detector_doc.get("checkpoint_path")),
        score_threshold=float(detector_doc.get("score_threshold", 0.50)),
        threshold_file=_resolve_path(
            source.parent, detector_doc.get("threshold_file")),
        candidate_score_threshold=float(
            detector_doc.get("candidate_score_threshold", 0.001)),
        classes=tuple(detector_doc.get("classes", ("echinus", "rock"))),
    )
    if validate_detector:
        detector.validate(require_threshold=require_threshold)

    return BCConfig(
        source_path=source,
        dataset=dict(document["dataset"]),
        camera=dict(document["camera"]),
        depth=dict(document["depth"]),
        map=dict(document["map"]),
        target=dict(document["target"]),
        model=dict(document["model"]),
        training=dict(document.get("training", {})),
        detector=detector,
    )
