"""将采集器 Episode 转换为累积语义地图 BC 数据。

Example:
  python preprocess_dataset.py --dataset-root DATA --output-root DATA/processed
  --intrinsics 500 500 320 240 --T-base-camera T.npy --T-world-base-mode pose
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from ..constants import ACTION_TO_ID, GOAL_TO_ID
from ..mapping.semantic_mapper import MapConfig, SemanticMapper
from ..perception.depth_projection import CameraIntrinsics
from .bc_dataset import split_by_episode


def _pose_matrix(row):
    """将 CSV 中的 xyz+rpy 位姿转换成 4x4 ``T_world_base``。

    这里采用 ZYX（yaw-pitch-roll）旋转组合，角度单位为弧度。
    """
    x, y, z = (float(row[k]) for k in ("x", "y", "z")); roll, pitch, yaw = (float(row[k]) for k in ("roll", "pitch", "yaw"))
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    R = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, :3] = R; T[:3, 3] = (x, y, z); return T


def build(dataset_root, output_root, intrinsics, T_base_camera, detector=None, map_config=MapConfig()):
    """遍历所有 Episode，逐步更新地图并生成 train/val/test jsonl。

    ``detector`` 只需要实现 ``detect(rgb_path)``；传入 None 时跳过语义
    检测，但仍会生成探索、障碍、机器人和访问通道。
    """
    dataset_root, output_root = Path(dataset_root).resolve(), Path(output_root).resolve(); output_root.mkdir(parents=True, exist_ok=True)
    all_records, episode_dirs = [], sorted(dataset_root.glob("episode_*"))  #遍历所有 episode 开头的目录
    for episode_dir in episode_dirs:
        trajectory = episode_dir / "trajectory.csv"
        if not trajectory.exists(): continue
        mapper = SemanticMapper(map_config); rows = list(csv.DictReader(trajectory.open(encoding="utf-8", newline="")))
        for row in rows:
            # 每条记录对应动作执行前的观测，保证 observation_t 与 action_t 对齐。
            rgb = episode_dir / row["rgb_path"]; depth = np.load(episode_dir / row["depth_path"])
            detections = detector.detect(rgb) if detector else []   #在这里就开始检测
            mapper.update(depth, intrinsics, T_base_camera, _pose_matrix(row), detections, (float(row["x"]), float(row["y"])))  #更新地图
            ep_out = output_root / episode_dir.name / "semantic_map"; ep_out.mkdir(parents=True, exist_ok=True) #创建输出目录
            step = int(row["step_id"]); map_file = ep_out / f"{step:06d}.npy"; np.save(map_file, mapper.map)
            all_records.append({"semantic_map": str(map_file.relative_to(output_root)), "episode_id": row.get("episode_id", episode_dir.name), "step_id": step,
                                "yaw": float(row["yaw"]), "goal_category": row.get("goal_category", "sea_urchin"),
                                "goal_id": GOAL_TO_ID.get(row.get("goal_category", "sea_urchin"), 0), "action": ACTION_TO_ID[row["expert_action"]]})
    groups = split_by_episode(all_records)
    for name, records in zip(("train", "val", "test"), groups):
        with (output_root / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    (output_root / "metadata.json").write_text(json.dumps({"records": len(all_records), "episodes": len(episode_dirs), "map_config": map_config.__dict__}, indent=2), encoding="utf-8")


def main():
    """命令行入口；地图参数从 bc.yaml 读取。"""
    p = argparse.ArgumentParser(); p.add_argument("--dataset-root", required=True); p.add_argument("--output-root", required=True)
    p.add_argument("--intrinsics", nargs=4, type=float, required=True, metavar=("FX", "FY", "CX", "CY")); p.add_argument("--T-base-camera", required=True)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "bc.yaml"), help="BC 配置文件")
    a = p.parse_args(); build(a.dataset_root, a.output_root, CameraIntrinsics(*a.intrinsics), np.load(a.T_base_camera), map_config=MapConfig.from_yaml(a.config))

if __name__ == "__main__": main()
