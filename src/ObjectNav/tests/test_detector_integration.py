import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from objectnav_bc.config_loader import DetectorConfig
from objectnav_bc.perception.detection_cache import (
    deserialize_detections,
    serialize_detections,
)
from objectnav_bc.perception.semantic_detector import Detection
from objectnav_bc.perception.semantic_detector import MMDetSemanticDetector


class DetectorConfigTests(unittest.TestCase):
    def test_calibrated_echinus_and_fallback_rock_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            threshold_file = root / "echinus_threshold.json"
            threshold_file.write_text(
                json.dumps(
                    {
                        "target_class": "echinus",
                        "checkpoint": str(
                            root
                            / "best_echinus_recall_at_precision_95_epoch_5.pth"
                        ),
                        "recommended_threshold": 0.73,
                    }
                ),
                encoding="utf-8",
            )
            config = DetectorConfig(
                enabled=True,
                config_path=root / "model.py",
                checkpoint_path=(
                    root / "best_echinus_recall_at_precision_95_epoch_5.pth"
                ),
                score_threshold=0.5,
                threshold_file=threshold_file,
                candidate_score_threshold=0.001,
                classes=("echinus", "rock"),
            )

            self.assertEqual(
                config.resolved_score_thresholds(),
                {"echinus": 0.73, "rock": 0.5},
            )


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


class _FakeInstances:
    bboxes = _FakeTensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    scores = _FakeTensor([0.72, 0.74, 0.49])
    labels = _FakeTensor([0, 0, 1])

    def cpu(self):
        return self


class _FakeSample:
    pred_instances = _FakeInstances()


class _FakeModel:
    dataset_meta = {"classes": ("echinus", "rock")}

    def eval(self):
        return self

    def parameters(self):
        return ()


class _FakeInferencer:
    def __init__(self, model, weights):
        self.model = _FakeModel()

    def __call__(self, image, **kwargs):
        return {"predictions": [_FakeSample()]}


class DetectorFilteringTests(unittest.TestCase):
    def test_thresholds_are_applied_per_class(self):
        fake_apis = types.ModuleType("mmdet.apis")
        fake_apis.DetInferencer = _FakeInferencer
        fake_mmdet = types.ModuleType("mmdet")
        fake_mmdet.apis = fake_apis
        with patch.dict(
            sys.modules,
            {"mmdet": fake_mmdet, "mmdet.apis": fake_apis},
        ):
            detector = MMDetSemanticDetector(
                "model.py",
                "model.pth",
                class_thresholds={"echinus": 0.73, "rock": 0.5},
            )
            detections = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "echinus")
        self.assertAlmostEqual(detections[0].score, 0.74)

    def test_cache_preserves_model_labels_and_class_names(self):
        record = serialize_detections(
            "episode_0001",
            3,
            "episode_0001/rgb/000003.png",
            [
                Detection((1, 2, 3, 4), 0.8, "echinus"),
                Detection((5, 6, 7, 8), 0.6, "rock"),
            ],
            ("echinus", "rock"),
        )
        self.assertEqual(record["labels"], [0, 1])
        restored = deserialize_detections(record, ("echinus", "rock"))
        self.assertEqual([item.class_name for item in restored], ["echinus", "rock"])


if __name__ == "__main__":
    unittest.main()
