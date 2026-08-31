"""计算训练计划要求的离线 BC 指标。"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..constants import ACTION_NAMES, GOAL_NAMES
from ..dataset.bc_dataset import BehaviorCloningDataset
from ..models import MapEncoder, BCPolicy


def evaluate(data_root, checkpoint, device="cpu", batch_size=64):
    """在 test.jsonl 上计算准确率、混淆矩阵和 STOP 指标。"""
    ds = BehaviorCloningDataset(Path(data_root) / "test.jsonl"); loader = DataLoader(ds, batch_size=batch_size, shuffle=False)  #加载测试数据集
    state = torch.load(checkpoint, map_location=device) #加载模型参数
    encoder, policy = MapEncoder(), BCPolicy(num_goals=len(GOAL_NAMES)) #创建编码器和策略网络，num_goals是目标类别的数量，这里是1 
    encoder.load_state_dict(state["encoder"])   #加载模型参数
    policy.load_state_dict(state["policy"]) #加载模型参数
    encoder.to(device).eval(); policy.to(device).eval() #将模型移动到指定设备，并设置为评估模式，关闭dropout和batchnorm等训练特性
    # cm[真实动作, 预测动作]，行归一化即可得到每类准确率。
    cm = np.zeros((4, 4), dtype=np.int64); loss_fn = torch.nn.CrossEntropyLoss(); loss = 0  #计算交叉熵损失
    with torch.no_grad():   #在不计算梯度的情况下进行推理
        for b in loader:
            logits = policy(encoder(b["semantic_map"].to(device)), b["goal_id"].to(device), b["yaw"].to(device))    #计算动作 logits，先通过编码器提取特征，再通过策略网络得到动作 logits
            pred = logits.argmax(1).cpu().numpy()   #计算预测动作的索引
            true = b["action"].numpy()   #获取真实动作的索引
            loss += loss_fn(logits, b["action"].to(device)).item() * len(true)  #计算交叉熵损失，并乘以样本数量，累加到总损失中
            for t, q in zip(true, pred): cm[t, q] += 1  #统计混淆矩阵，t是真实动作，q是预测动作，cm[t, q]表示真实动作为t且预测动作为q的样本数量
    per_class = {ACTION_NAMES[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0) for i in range(4)}    #计算每类动作的准确率，cm[i, i]表示真实动作为i且预测动作为i的样本数量，cm[i].sum()表示真实动作为i的样本总数
    stop_tp = cm[3, 3]; stop_precision = float(stop_tp / max(1, cm[:, 3].sum())); stop_recall = per_class["STOP"]
    return {"overall_accuracy": float(np.trace(cm) / max(1, cm.sum())), "per_class_accuracy": per_class, "stop_precision": stop_precision, "stop_recall": stop_recall, "validation_loss": float(loss / max(1, len(ds))), "confusion_matrix": cm.tolist()}


def main():
    """命令行评估入口。"""
    p = argparse.ArgumentParser(); p.add_argument("--data-root", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--output", default="offline_metrics.json"); p.add_argument("--device", default="cpu"); a = p.parse_args(); result = evaluate(a.data_root, a.checkpoint, a.device); print(json.dumps(result, indent=2)); Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
