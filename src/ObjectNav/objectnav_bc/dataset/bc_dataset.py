"""BC 数据集和 Episode 级别数据划分。"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..constants import ACTION_TO_ID, GOAL_TO_ID


def split_by_episode(records, seed=42, ratios=(.70, .15, .15)):
    """按 Episode 划分记录，禁止同一轨迹泄漏到多个数据集。

    先随机打乱 Episode ID，再按 70/15/15 比例分组；随机种子固定后
    可以复现实验划分。
    """
    episodes = sorted({str(r["episode_id"]) for r in records})  #找出所有不同的 episode_id，并排序
    rng = np.random.default_rng(seed); rng.shuffle(episodes)    #随机打乱 episode_id 的顺序，使用指定的随机种子
    n = len(episodes); n_train = int(n * ratios[0]); n_val = int(n * ratios[1]) #计算要划分的数据集
    groups = (set(episodes[:n_train]), set(episodes[n_train:n_train+n_val]), set(episodes[n_train+n_val:])) #划分数据集
    return [[r for r in records if str(r["episode_id"]) in group] for group in groups]  #对于 groups 里的每一组 episode ID，都去 records 里把属于这一组 episode 的记录挑出来，最后得到三组记录。


class BehaviorCloningDataset(Dataset):
    """读取 jsonl 清单和对应的累计语义地图。"""

    def __init__(self, manifest, root=None):
        self.root = Path(root or Path(manifest).parent).resolve()   #确定根目录，如果没有指定，则使用清单文件所在的目录，是Path(manifest).parent是获取清单文件的父目录，Path(manifest).parent.resolve()是获取清单文件的绝对路径
        with open(manifest, encoding="utf-8") as f:    #打开文件 
            self.records = [json.loads(line) for line in f if line.strip()] #读取每一行，去掉空白行，并将每一行的 JSON 字符串解析为 Python 对象，存储在列表中
        if not self.records: raise ValueError(f"No records in {manifest}")

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        """返回训练计划要求的 semantic_map、goal_id、yaw、action 等字段。"""
        r = self.records[index]
        map_path = Path(r["semantic_map"])  #找到语义地图路径
        if not map_path.is_absolute(): map_path = self.root / map_path
        semantic_map = np.load(map_path).astype(np.float32) #读取语义地图
        if semantic_map.ndim != 3: raise ValueError(f"Expected CxHxW map: {map_path}")  #检查格式
        action = r["action"]
        if isinstance(action, str): action = ACTION_TO_ID[action]
        goal = r.get("goal_id", GOAL_TO_ID.get(r.get("goal_category", "sea_urchin"), 0))    #获取目标类别的 ID，如果没有指定，则默认为 0
        return {"semantic_map": torch.from_numpy(semantic_map), "goal_id": torch.tensor(goal, dtype=torch.long),
                "yaw": torch.tensor(float(r["yaw"]), dtype=torch.float32), "action": torch.tensor(action, dtype=torch.long),
                "episode_id": str(r["episode_id"]), "step_id": int(r["step_id"])}


def class_counts(dataset):
    """统计四种专家动作数量，用于计算加权交叉熵权重。"""
    return Counter(int(r["action"]) if isinstance(r["action"], int) else ACTION_TO_ID[r["action"]] for r in dataset.records) #统计各种专家动作分别有多少条数据
