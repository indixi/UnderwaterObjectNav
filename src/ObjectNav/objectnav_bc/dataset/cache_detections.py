"""Run frozen GFL once and cache reusable low-threshold predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config_loader import load_bc_config
from ..perception.detection_cache import (
    detector_signature,
    serialize_detections,
)
from ..perception.semantic_detector import build_semantic_detector
from .episode_io import episode_directories, trajectory_rows


def build_cache(dataset_root, output_root, runtime, force=False):
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "detections.jsonl"
    metadata_path = output_root / "detection_cache_metadata.json"
    signature = detector_signature(runtime.detector)

    if records_path.is_file() and metadata_path.is_file() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("detector_signature") == signature:
            print(f"reuse detection cache: {records_path}")
            return records_path
        raise ValueError(
            "existing detection cache belongs to another model; use --force")

    detector = build_semantic_detector(
        runtime.detector, require_threshold=False)
    include_failed = bool(runtime.dataset["include_failed_episodes"])
    count = 0
    with records_path.open("w", encoding="utf-8") as stream:
        for episode_dir in episode_directories(dataset_root, include_failed):
            for row in trajectory_rows(episode_dir):
                rgb = episode_dir / row["rgb_path"]
                candidates = detector.detect_candidates(rgb)
                record = serialize_detections(
                    episode_dir.name,
                    int(row["step_id"]),
                    str(Path(episode_dir.name) / row["rgb_path"]),
                    candidates,
                    runtime.detector.classes,
                )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"cached {count} frames")

    metadata_path.write_text(
        json.dumps(
            {
                "detector_signature": signature,
                "records": count,
                "dataset_root": str(dataset_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {count} detections to {records_path}")
    return records_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "bc.yaml"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runtime = load_bc_config(
        args.config, validate_detector=True, require_threshold=False)
    build_cache(
        args.dataset_root, args.output_root, runtime, force=args.force)


if __name__ == "__main__":
    main()
