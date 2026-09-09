"""Train Global CNN + Local CNN + Target encoder + MLP with BC."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config_loader import load_bc_config
from ..constants import (
    ACTION_NAMES,
    GLOBAL_MAP_CHANNELS,
    GOAL_NAMES,
    LOCAL_MAP_CHANNELS,
)
from ..dataset.bc_dataset import BehaviorCloningDataset, class_counts
from ..models import build_policy_components


def _forward(components, batch, device):
    global_encoder, local_encoder, policy = components
    global_feature = global_encoder(batch["global_map"].to(device))
    local_feature = local_encoder(batch["local_map"].to(device))
    return policy(
        global_feature,
        local_feature,
        batch["goal_id"].to(device),
        batch["target_cue"].to(device),
        batch["yaw"].to(device),
    )


def _checkpoint(components, runtime, epoch):
    global_encoder, local_encoder, policy = components
    return {
        "global_encoder": global_encoder.state_dict(),
        "local_encoder": local_encoder.state_dict(),
        "policy": policy.state_dict(),
        "model_config": runtime.model,
        "target_config": runtime.target,
        "action_names": ACTION_NAMES,
        "goal_names": GOAL_NAMES,
        "global_map_channels": GLOBAL_MAP_CHANNELS,
        "local_map_channels": LOCAL_MAP_CHANNELS,
        "epoch": epoch,
    }


def run(args, runtime):
    seed = int(args.seed if args.seed is not None else runtime.training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    output = Path(args.work_dir)
    output.mkdir(parents=True, exist_ok=True)

    train = BehaviorCloningDataset(Path(args.data_root) / "train.jsonl")
    validation = BehaviorCloningDataset(Path(args.data_root) / "val.jsonl")
    if train.metadata != validation.metadata:
        raise ValueError("train and validation metadata differ")
    if train.metadata["runtime"]["model"] != runtime.model:
        raise ValueError(
            "Processed dataset model configuration differs from bc.yaml; "
            "reprocess the dataset or restore the matching configuration"
        )
    if tuple(train.metadata["global_map_channels"]) != GLOBAL_MAP_CHANNELS:
        raise ValueError("Processed Global Map channel order is incompatible")
    if tuple(train.metadata["local_map_channels"]) != LOCAL_MAP_CHANNELS:
        raise ValueError("Processed Local Map channel order is incompatible")
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else runtime.training["batch_size"]
    )
    workers = int(
        args.workers if args.workers is not None else runtime.training["workers"])
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=workers)
    validation_loader = DataLoader(
        validation, batch_size=batch_size, shuffle=False, num_workers=workers)

    components = build_policy_components(runtime.model, len(GOAL_NAMES))
    for component in components:
        component.to(device)
    counts = class_counts(train)
    weights = torch.tensor(
        [
            len(train) / (len(ACTION_NAMES) * max(1, counts.get(index, 0)))
            for index in range(len(ACTION_NAMES))
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    learning_rate = float(
        args.lr if args.lr is not None else runtime.training["learning_rate"])
    weight_decay = float(
        args.weight_decay
        if args.weight_decay is not None
        else runtime.training["weight_decay"]
    )
    parameters = [
        parameter
        for component in components
        for parameter in component.parameters()
    ]
    optimizer = torch.optim.Adam(
        parameters, lr=learning_rate, weight_decay=weight_decay)
    epochs = int(
        args.epochs if args.epochs is not None else runtime.training["epochs"])

    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        for component in components:
            component.train()
        train_loss = 0.0
        for batch in train_loader:
            logits = _forward(components, batch, device)
            loss = criterion(logits, batch["action"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch["action"])

        for component in components:
            component.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                logits = _forward(components, batch, device)
                validation_loss += criterion(
                    logits, batch["action"].to(device)
                ).item() * len(batch["action"])
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / len(train),
            "val_loss": validation_loss / len(validation),
        }
        history.append(metrics)
        print(metrics)
        state = _checkpoint(components, runtime, epoch)
        torch.save(state, output / "latest_policy.pt")
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            torch.save(state, output / "best_policy.pt")

    (output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", default="work_dirs/objectnav_bc")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "bc.yaml"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    runtime = load_bc_config(args.config, validate_detector=False)
    run(args, runtime)


if __name__ == "__main__":
    main()
