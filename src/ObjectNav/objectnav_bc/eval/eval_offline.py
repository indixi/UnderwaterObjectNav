"""Evaluate a trained BC policy on the held-out episode split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..constants import ACTION_NAMES, GLOBAL_MAP_CHANNELS, LOCAL_MAP_CHANNELS
from ..dataset.bc_dataset import BehaviorCloningDataset
from ..models import build_policy_components


def evaluate(data_root, checkpoint, device="cpu", batch_size=64):
    dataset = BehaviorCloningDataset(Path(data_root) / "test.jsonl")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    state = torch.load(checkpoint, map_location=device)
    if tuple(state["action_names"]) != ACTION_NAMES:
        raise ValueError("Checkpoint action order is incompatible")
    if tuple(state["global_map_channels"]) != GLOBAL_MAP_CHANNELS:
        raise ValueError("Checkpoint Global Map channel order is incompatible")
    if tuple(state["local_map_channels"]) != LOCAL_MAP_CHANNELS:
        raise ValueError("Checkpoint Local Map channel order is incompatible")
    if state["model_config"] != dataset.metadata["runtime"]["model"]:
        raise ValueError("Checkpoint and processed dataset model configs differ")
    components = build_policy_components(
        state["model_config"], len(state["goal_names"]))
    for component, key in zip(
        components, ("global_encoder", "local_encoder", "policy")
    ):
        component.load_state_dict(state[key])
        component.to(device).eval()

    confusion = np.zeros(
        (len(ACTION_NAMES), len(ACTION_NAMES)), dtype=np.int64)
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            global_feature = components[0](batch["global_map"].to(device))
            local_feature = components[1](batch["local_map"].to(device))
            logits = components[2](
                global_feature,
                local_feature,
                batch["goal_id"].to(device),
                batch["target_cue"].to(device),
                batch["yaw"].to(device),
            )
            truth = batch["action"].numpy()
            prediction = logits.argmax(1).cpu().numpy()
            total_loss += loss_function(
                logits, batch["action"].to(device)
            ).item() * len(truth)
            np.add.at(confusion, (truth, prediction), 1)

    per_class = {
        ACTION_NAMES[index]: (
            float(confusion[index, index] / confusion[index].sum())
            if confusion[index].sum()
            else 0.0
        )
        for index in range(len(ACTION_NAMES))
    }
    stop_index = ACTION_NAMES.index("STOP")
    stop_true_positive = confusion[stop_index, stop_index]
    return {
        "overall_accuracy": float(
            np.trace(confusion) / max(1, confusion.sum())),
        "per_class_accuracy": per_class,
        "stop_precision": float(
            stop_true_positive / max(1, confusion[:, stop_index].sum())),
        "stop_recall": per_class["STOP"],
        "test_loss": float(total_loss / len(dataset)),
        "confusion_matrix": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="offline_metrics.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = evaluate(
        args.data_root, args.checkpoint, args.device, args.batch_size)
    print(json.dumps(result, indent=2))
    Path(args.output).write_text(
        json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
