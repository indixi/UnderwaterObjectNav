# ObjectNav 离线 BC 使用说明

本目录实现离线处理、离线训练、离线评估，以及供 ROS 在线节点复用的纯 Python 推理 Worker。ROS 服务、动作状态机和录制入口位于上级 `scripts/`；完整在线用法见 [`../underwater_objectnav_online_usage.md`](../underwater_objectnav_online_usage.md)。图像检测器被冻结，BC 反向传播不会修改 GFL/ResNet50/FPN 权重。

完整的问题分析和设计依据另见 [`../underwater_objectnav_gfl_depth_bc_reference.md`](../underwater_objectnav_gfl_depth_bc_reference.md)。

## 1. 数据流

```text
RGB --冻结 GFL--> 后处理候选框缓存（score >= 0.001）
                              |
Depth + 世界位姿 --------------+
                              v
              按部署阈值过滤 + bbox/depth 融合
                              |
                +-------------+-------------+
                v                           v
       Global Map (6×80×140)       Local Map (4×60×60)
                +-------------+-------------+
                              |
        Goal + 当前帧 Target Cue + sin/cos(yaw)
                              v
                         4 动作 BC
```

检测候选缓存和地图预处理分成两步。以后只调整 `echinus` 或 `rock` 阈值时，可直接重建地图，不必再次运行 GFL。

## 2. 检测器文件与类别约定

把另一台主机上的文件复制到 `objectnav_bc/perception/model/`：

```text
best_echinus_recall_at_precision_95_epoch_5.pth
echinus_threshold.json
```

推理专用配置已放在同一目录：

```text
gfl_r50_fpn_underwater_objectnav_inference.py
```

三者的路径统一由 `config/bc.yaml` 配置。检测模型内部类别 ID 必须保持：

```text
0 = echinus
1 = rock
```

这里的检测类别 ID 与导航目标 ID、地图通道编号彼此独立。当前唯一导航目标为 `echinus`，导航目标 ID 是 0；`rock` 不作为终点，但会写入地图辅助导航。

- `echinus` 使用 `echinus_threshold.json` 中的 `recommended_threshold`。
- `rock` 暂时使用 YAML 中的 `score_threshold: 0.50`。
- JSON 记录的 checkpoint 文件名必须与配置的 checkpoint 文件名一致，否则立即报错。

## 3. 配置

所有可人工调整的参数都在 `config/bc.yaml`，包括：

- 相机内参、640×480 图像尺寸和 `T_base_camera`；
- 传感器标称世界 Z 高度 3.3 m 及误差告警范围；
- 深度有效范围、ROI、采样步长和障碍物高度带；
- Global Map 的世界原点、物理范围和分辨率；
- Local Map 的尺寸和分辨率；
- 语义阈值、地图标记半径、数据集划分和网络维度。

默认 Global Map 使用固定 NED 世界坐标范围：

```text
origin=(-7,-4), width=14 m, height=8 m, resolution=0.1 m
shape=[6,80,140]
channels=[obstacle, explored, visited, robot, echinus, rock]
```

默认 Local Map 是以机器人为中心、机器人前方朝图像上方的累计高分辨率地图：

```text
width=3 m, height=3 m, resolution=0.05 m
shape=[4,60,60]
channels=[obstacle, rock, echinus, visited]
```

NED 坐标系中左转会让固定环境在 Local Map 上顺时针移动。每个 episode 开始时 Global/Local 累积证据都清零；语义置信度在同一 episode 内取历史最大值。

## 4. 原始数据要求

数据读取格式与 `data_generate` 一致：

```text
underwater_objectnav_dataset/
  metadata.yaml
  episode_0001/
    episode.yaml
    trajectory.csv
    rgb/000000.png
    depth/000000.npy
```

`trajectory.csv` 必须包含：

```text
step_id,action_id,action_start_time,timestamp,rgb_path,depth_path,
x,y,z,roll,pitch,yaw,goal_category,expert_action
```

`goal_category` 使用 `echinus`。只处理 `episode.yaml` 中 `success: true` 的 episode；所有以 `episode_false` 开头的目录无条件忽略。数据按 episode 以固定随机种子划分为 70%/15%/15%，相邻帧不会跨集合泄漏。

## 5. 运行顺序

以下命令在 `UnderwaterObjectNav/src/ObjectNav` 下执行：

```powershell
$env:PYTHONPATH = (Get-Location).Path
pip install -r objectnav_bc/requirements.txt
```

MMDetection/MMEngine 的安装版本应与训练 GFL 时一致。

第一步，只运行一次冻结检测器并保存低阈值、NMS 后候选框：

```powershell
python -m objectnav_bc.dataset.cache_detections --dataset-root objectnav_bc/data/underwater_objectnav_dataset  --output-root objectnav_bc/data/underwater_objectnav_dataset/detection_cache  --config objectnav_bc/config/bc.yaml
```

第二步，根据当前部署阈值融合深度并生成地图：

```powershell
python -m objectnav_bc.dataset.preprocess_dataset  --dataset-root objectnav_bc/data/underwater_objectnav_dataset  --detection-cache objectnav_bc/data/underwater_objectnav_dataset/detection_cache/detections.jsonl  --output-root objectnav_bc/data/underwater_objectnav_dataset/processed  --config objectnav_bc/config/bc.yaml
```

地图在磁盘上保存为 `float16`，Dataset 加载时转换为 `float32`。输出包含 `train.jsonl`、`val.jsonl`、`test.jsonl`、`metadata.json` 和每一步的 Global/Local `.npy`。

第三步，训练 BC：

```powershell
python -m objectnav_bc.train.train_bc  --data-root objectnav_bc/data/underwater_objectnav_dataset/processed  --work-dir work_dirs/objectnav_bc  --device cuda
```

第四步，在测试集评估：

```powershell
python -m objectnav_bc.eval.eval_offline  --data-root objectnav_bc/data/underwater_objectnav_dataset/processed  --checkpoint work_dirs/objectnav_bc/best_policy.pt  --output work_dirs/objectnav_bc/offline_metrics.json  --device cuda
```

## 6. 网络输入

每一步送入 BC 网络的是：

- Global Map，经 CNN 和 `AdaptiveAvgPool(5,9)` 得到 256 维；
- Local Map，经 CNN 和 `AdaptiveAvgPool(4,4)` 得到 128 维；
- `goal_id` 的 8 维 embedding；
- 当前帧 Target Cue 经 `Linear(3,16)`；
- `sin(yaw), cos(yaw)` 两维方向信息。

Target Cue 原始值为 `[visible, bearing_rad, distance_m]`。只有当前帧海胆检测框内存在有效深度时 `visible=1`；多只海胆选择距离最近的一只作为 cue，但所有有效海胆都写入地图。送入网络前，bearing 除以 35°并裁剪至 `[-1,1]`，distance 除以 10 m 并裁剪至 `[0,1]`；不可见时三维全部为 0。

## 7. 测试

```powershell
python -B -m unittest discover -s tests -v
```

测试覆盖类别阈值、相机外参投影、无效深度、episode 筛选、置信度最大累积/清零和 Local Map 左转方向。安装 PyTorch 后还会执行网络形状测试。
