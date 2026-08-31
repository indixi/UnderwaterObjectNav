"""训练 MapEncoder + GoalEmbedding + MLP 策略。

损失是按动作频数计算权重的 Cross Entropy，用于缓解专家数据类别不平衡。
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..dataset.bc_dataset import BehaviorCloningDataset, class_counts
from ..models import MapEncoder, BCPolicy
from ..constants import ACTION_NAMES, GOAL_NAMES


def run(args):
    """执行完整训练循环，并保存 latest/best checkpoint。"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = BehaviorCloningDataset(Path(args.data_root) / "train.jsonl"); val = BehaviorCloningDataset(Path(args.data_root) / "val.jsonl")
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    vloader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    encoder, policy = MapEncoder(), BCPolicy(num_goals=len(GOAL_NAMES)); encoder.to(device); policy.to(device)  #创建编码器和策略网络，num_goals是目标类别的数量，这里是1，并放入设备
    # w_c=N/(K*N_c)，出现次数少的动作会得到更高的损失权重。
    counts = class_counts(train); total = len(train); weights = torch.tensor([total / (4 * max(1, counts.get(i, 0))) for i in range(4)], dtype=torch.float32, device=device)    #计算每个动作类别的权重，N是总样本数，K是类别数，N_c是类别c的样本数，出现次数少的动作会得到更高的损失权重
    criterion = torch.nn.CrossEntropyLoss(weight=weights)   #创建加权交叉熵损失函数，使用计算得到的权重
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(policy.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        encoder.train(); policy.train(); train_loss = 0.0
        for b in loader:
            # 前向顺序与训练计划一致：地图编码 -> 策略融合 -> logits。
            logits = policy(encoder(b["semantic_map"].to(device)), b["goal_id"].to(device), b["yaw"].to(device)); loss = criterion(logits, b["action"].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()    #计算梯度并更新参数
            train_loss += loss.item() * len(b["action"])
        encoder.eval(); policy.eval(); val_loss = 0.0
        with torch.no_grad(): 
            for b in vloader:
                logits = policy(encoder(b["semantic_map"].to(device)), b["goal_id"].to(device), b["yaw"].to(device)); val_loss += criterion(logits, b["action"].to(device)).item() * len(b["action"])
        metrics = {"epoch": epoch, "train_loss": train_loss / len(train), "val_loss": val_loss / max(1, len(val))}; history.append(metrics); print(metrics)
        state = {"encoder": encoder.state_dict(), "policy": policy.state_dict(), "action_names": ACTION_NAMES, "goal_names": GOAL_NAMES, "map_channels": 7, "epoch": epoch}
        torch.save(state, out / "latest_policy.pt") #保存checkpoint
        if metrics["val_loss"] < best: best = metrics["val_loss"]; torch.save(state, out / "best_policy.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def main():
    """解析训练超参数并启动训练。"""
    p = argparse.ArgumentParser(); p.add_argument("--data-root", required=True); p.add_argument("--work-dir", default="work_dirs/objectnav_bc")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())

if __name__ == "__main__": main()
