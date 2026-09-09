"""Frozen object detector adapters used by the ObjectNav semantic mapper."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    """One object detection in source-image pixel coordinates."""

    bbox: tuple[float, float, float, float]
    score: float
    class_name: str


def filter_detections(
    detections: list[Detection], thresholds: dict[str, float]
) -> list[Detection]:
    """Apply per-class thresholds without requiring MMDetection."""
    return [
        detection
        for detection in detections
        if detection.class_name in thresholds
        and detection.score >= float(thresholds[detection.class_name])
    ]


class MMDetSemanticDetector:
    """Frozen MMDetection adapter shared by offline and online ObjectNav.

    MMDetection labels are contiguous model indices. For this checkpoint the
    only valid order is ``0=echinus, 1=rock``. Thresholds are applied per class
    because the calibrated echinus operating point must not be reused for rock.
    """

    def __init__(
        self,
        config: str,
        checkpoint: str,
        class_names: tuple[str, ...] = ("echinus", "rock"),
        class_thresholds: dict[str, float] | None = None,
        candidate_score_threshold: float = 0.001,
        device: str | None = None,
    ):
        # Keep policy-only workflows importable when MMDetection is absent.
        from mmdet.apis import DetInferencer

        inferencer_args = {"model": config, "weights": checkpoint}
        if device is not None:
            inferencer_args["device"] = device
        self.inferencer = DetInferencer(**inferencer_args)
        self.class_names = tuple(class_names)
        if not self.class_names:
            raise ValueError("class_names cannot be empty")

        model_classes = tuple(
            self.inferencer.model.dataset_meta.get("classes", ()))
        if model_classes and model_classes != self.class_names:
            raise ValueError(
                "detector class order mismatch: "
                f"model={model_classes}, configured={self.class_names}"
            )

        thresholds = class_thresholds or {
            name: 0.5 for name in self.class_names
        }
        unknown = set(thresholds) - set(self.class_names)
        missing = set(self.class_names) - set(thresholds)
        if unknown or missing:
            raise ValueError(
                f"class thresholds mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        self.class_thresholds = {
            name: float(thresholds[name]) for name in self.class_names
        }
        self.candidate_score_threshold = float(candidate_score_threshold)
        if any(
            not 0.0 <= value <= 1.0
            for value in self.class_thresholds.values()
        ):
            raise ValueError("all class thresholds must be in [0, 1]")
        if not 0.0 <= self.candidate_score_threshold <= 1.0:
            raise ValueError("candidate_score_threshold must be in [0, 1]")

        # Stage-one ObjectNav freezes perception. These flags make that
        # boundary explicit even when embedded in a larger PyTorch process.
        self.inferencer.model.eval()
        for parameter in self.inferencer.model.parameters():
            parameter.requires_grad_(False)

    def detect(self, image: str | Path | np.ndarray) -> list[Detection]:
        """Detect one RGB frame and return mapper-compatible detections."""
        return self.filter_detections(self.detect_candidates(image))

    def detect_candidates(
        self, image: str | Path | np.ndarray
    ) -> list[Detection]:
        """Return low-threshold post-NMS candidates for reusable caching."""
        inferencer_input = str(image) if isinstance(image, Path) else image
        result = self.inferencer(
            inferencer_input,
            pred_score_thr=self.candidate_score_threshold,
            no_save_pred=True,
            return_datasamples=True,
        )
        pred = result["predictions"][0].pred_instances.cpu()
        detections = []
        for box, score, label in zip(
            pred.bboxes.numpy(), pred.scores.numpy(), pred.labels.numpy()
        ):
            label = int(label)
            if label < 0 or label >= len(self.class_names):
                raise ValueError(
                    f"prediction label {label} is outside {self.class_names}"
                )
            name = self.class_names[label]
            detections.append(
                Detection(tuple(map(float, box)), float(score), name)
            )
        return detections

    def filter_detections(
        self, detections: list[Detection]
    ) -> list[Detection]:
        """Apply the deployment threshold belonging to each class."""
        return filter_detections(detections, self.class_thresholds)


def build_semantic_detector(
    config,
    require_threshold: bool = True,
    device: str | None = None,
):
    """Build the configured detector, or return ``None`` when disabled.

    Offline preprocessing and the online inference worker both use this factory,
    keeping checkpoint details out of the navigation implementation.
    """
    if not config.enabled:
        return None
    config.validate(require_threshold=require_threshold)
    thresholds = (
        config.resolved_score_thresholds()
        if require_threshold
        else {name: config.score_threshold for name in config.classes}
    )
    return MMDetSemanticDetector(
        config=str(config.config_path),
        checkpoint=str(config.checkpoint_path),
        class_names=config.classes,
        class_thresholds=thresholds,
        candidate_score_threshold=config.candidate_score_threshold,
        device=device,
    )


class JsonDetectionDetector:
    """Read cached detections produced by image inference tooling."""

    def __init__(self, records: dict, score_threshold: float = 0.30):
        self.records = records
        self.score_threshold = score_threshold

    def detect(self, image: str | Path | np.ndarray) -> list[Detection]:
        """Look up a path-keyed record; return no detections if absent."""
        key = (
            str(Path(image).resolve())
            if not isinstance(image, np.ndarray)
            else ""
        )
        record = self.records.get(key, self.records.get(str(image), {}))
        return [
            Detection(tuple(box), float(score), name)
            for box, score, name in zip(
                record.get("boxes_xyxy", []),
                record.get("scores", []),
                record.get("class_names", []),
            )
            if score >= self.score_threshold
        ]
