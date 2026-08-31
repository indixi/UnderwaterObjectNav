# underwater ObjectNav 代码框架与使用说明

本目录实现训练计划中的第一阶段：使用 RGB-D、机器人位姿和专家动作，构建累积语义地图，并训练离散动作 Behavior Cloning（BC）策略。

当前策略动作空间为：

```text
0 FORWARD       前进
1 TURN_LEFT     左转
2 TURN_RIGHT    右转
3 STOP          停止
```

## 1. 总体框架

```text
采集数据
  RGB + Depth + Robot Pose + Goal Category + Expert Action
                         |
                         v
              preprocess_dataset.py
                         |
                         +--> 深度投影到世界坐标
                         +--> DUO GFL 检测器提供语义目标框
                         +--> 更新累积 Semantic Map
                         +--> 保存每个 step 的 .npy 地图
                         +--> 按 Episode 划分 train/val/test
                         |
                         v
                 BehaviorCloningDataset
                         |
        +----------------+----------------+
        v                v                v
   MapEncoder       GoalEmbedding     sin(yaw), cos(yaw)
       256-D             8-D                2-D
        +----------------+----------------+
                         v
                    266-D state
                         |
                         v
                    Policy MLP
                         |
                    4-D logits
                         |
                         v
                 离散动作 / ROS 执行
```

第一阶段冻结 ResNet50 + FPN + GFL 感知模型，只使用它的检测结果更新语义地图；BC 训练只更新地图编码器、目标 Embedding 和策略 MLP。

## 2. 目录结构

```text
object_Nav/
├── underwater_objectnav_bc_training_plan.md  # 训练需求
├── README.md                                 # 本说明
└── objectnav_bc/
    ├── constants.py                           # 动作、目标和地图通道定义
    ├── config/bc.yaml                         # 地图和训练超参数
    ├── perception/
    │   ├── depth_projection.py                 # Depth 像素投影到世界坐标
    │   └── semantic_detector.py               # DUO GFL / 缓存 JSON 适配器
    ├── mapping/
    │   └── semantic_mapper.py                 # 累积 2-D 语义地图
    ├── dataset/
    │   ├── preprocess_dataset.py              # 原始 Episode 转 BC 数据
    │   └── bc_dataset.py                      # PyTorch Dataset 和 episode 划分
    ├── models/
    │   ├── map_encoder.py                     # 7 通道地图 -> 256-D
    │   └── policy_mlp.py                      # 266-D -> 4-D logits
    ├── train/train_bc.py                      # BC 训练入口
    └── eval/
        ├── eval_offline.py                    # 准确率、混淆矩阵、STOP 指标
        └── eval_closed_loop.py                # ROS 可调用的推理适配器
```

## 3. 各模块实现细节

### 3.1 `constants.py`

地图固定为 7 个通道：

```text
0 obstacle     障碍物/深度返回
1 explored     已探索区域
2 robot        当前机器人位置
3 visited      历史访问网格
4 sea_urchin   海胆语义证据
5 rock         岩石语义证据
6 sand         沙地语义证据
```

默认地图范围为 `10 m × 10 m`，分辨率为 `0.5 m`，因此地图张量为：

```text
[7, 20, 20]
```

目标类别当前为 `sea_urchin`。DUO 数据集中的 `echinus`、`holothurian` 等检测类别可在检测适配器中映射为海胆语义证据。

### 3.2 深度投影

`depth_projection.py` 使用相机内参：

```text
Xc = (u - cx) * d / fx
Yc = (v - cy) * d / fy
Zc = d
```

再通过：

```text
Pw = T_world_base @ T_base_camera @ Pc
```

投影到世界坐标。无效深度、NaN、过近和过远深度会被过滤。默认每隔 4 个像素采样一次，降低地图更新开销。

### 3.3 `SemanticMapper`

`SemanticMapper` 在一个 Episode 内持续保存地图，不会每帧清空：

1. 将有效深度点投影到世界 XY 平面。
2. 标记 `explored` 和 `obstacle`。
3. 根据机器人世界坐标标记 `robot` 和 `visited`。
4. 使用检测器 bbox 内的深度点更新海胆、岩石、沙地通道。
5. 保存当前累计地图副本。

检测器不是地图模块的硬依赖；没有检测结果时仍可生成探索、障碍和轨迹地图，便于先验证数据链路。

### 3.4 Dataset 与防止数据泄漏

原始数据必须按 Episode 保存。预处理后生成：

```text
processed/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── metadata.json
└── episode_0001/semantic_map/000000.npy
```

每行 JSONL 至少包含：

```json
{
  "semantic_map": "episode_0001/semantic_map/000000.npy",
  "episode_id": "episode_0001",
  "step_id": 0,
  "goal_category": "sea_urchin",
  "goal_id": 0,
  "yaw": 0.35,
  "action": 1
}
```

划分比例为 70% / 15% / 15%，先打乱 Episode，再按 Episode 划分。因此同一条轨迹的连续帧不会同时出现在训练集和验证集。

### 3.5 策略网络

`MapEncoder`：

```text
7 x H x W
 -> Conv(7, 32, stride=2) + ReLU
 -> Conv(32, 64, stride=2) + ReLU
 -> Conv(64, 128, stride=2) + ReLU
 -> AdaptiveAvgPool
 -> Linear(128, 256)
```

`BCPolicy`：

```text
goal_id -> Embedding(1, 8)
yaw     -> [sin(yaw), cos(yaw)]       # 2-D

[map_feature(256), goal_feature(8), orientation(2)]
                         -> 266-D
                         -> Linear(266, 256)
                         -> Linear(256, 128)
                         -> Linear(128, 4)
```

训练使用加权交叉熵：

```text
w_c = N / (4 * N_c)
L   = CrossEntropyLoss(weight=w)(logits, expert_action)
```

这样可以缓解 `FORWARD` 样本远多于 `STOP`、转向动作的问题。

## 4. 输入数据格式

采集器输出应符合：

```text
dataset/
├── metadata.yaml
├── episode_0001/
│   ├── rgb/000000.png
│   ├── depth/000000.npy
│   ├── trajectory.csv
│   └── episode.yaml
└── episode_0002/
    └── ...
```

`trajectory.csv` 必须包含：

```text
step_id,timestamp,rgb_path,depth_path,
x,y,z,roll,pitch,yaw,goal_category,expert_action
```

位姿角度使用弧度；`rgb_path` 和 `depth_path` 相对于当前 Episode 目录。动作值可以使用数字，也可以使用 `FORWARD`、`TURN_LEFT`、`TURN_RIGHT`、`STOP` 字符串。

## 5. 安装与运行

在 `UnderwaterObjectNav/src/object_Nav` 目录执行：

```powershell
$env:PYTHONPATH = (Get-Location).Path
pip install -r objectnav_bc/requirements.txt
```

其中 PyTorch 应根据实际 CUDA、GPU 和操作系统从 PyTorch 官方命令安装；DUO GFL 检测器依赖仍按 `duo_gfl_project/README.md` 安装。

### 5.1 生成语义地图数据

```powershell
python -m objectnav_bc.dataset.preprocess_dataset `
  --dataset-root D:/path/to/underwater_objectnav_dataset `
  --output-root D:/path/to/underwater_objectnav_dataset/processed `
  --intrinsics FX FY CX CY `
  --T-base-camera D:/path/to/T_base_camera.npy
```

如果暂时不接 DUO 检测器，上述命令仍可生成基础地图。接入检测器时，在 Python 中创建：

```python
from objectnav_bc.perception.semantic_detector import MMDetSemanticDetector
from objectnav_bc.dataset.preprocess_dataset import build
from objectnav_bc.perception.depth_projection import CameraIntrinsics
import numpy as np

detector = MMDetSemanticDetector(
    config="../../../../duo_gfl_project/configs/gfl_r50_fpn_duo_base.py",
    checkpoint="path/to/best_checkpoint.pth",
    score_threshold=0.30,
)
build(
    dataset_root="path/to/dataset",
    output_root="path/to/dataset/processed",
    intrinsics=CameraIntrinsics(fx, fy, cx, cy),
    T_base_camera=np.load("path/to/T_base_camera.npy"),
    detector=detector,
)
```

也可以使用 `duo_gfl_project/tools/infer.py` 生成的缓存 JSON，通过 `JsonDetectionDetector` 接入，避免预处理时重复跑检测器。

### 5.2 训练 BC 策略

```powershell
python -m objectnav_bc.train.train_bc `
  --data-root D:/path/to/underwater_objectnav_dataset/processed `
  --work-dir work_dirs/objectnav_bc `
  --batch-size 32 `
  --lr 0.0003 `
  --epochs 30
```

输出：

```text
work_dirs/objectnav_bc/best_policy.pt
work_dirs/objectnav_bc/latest_policy.pt
work_dirs/objectnav_bc/history.json
```

`best_policy.pt` 根据验证集 loss 保存；训练日志中会输出每个 epoch 的训练 loss 和验证 loss。

### 5.3 离线评估

```powershell
python -m objectnav_bc.eval.eval_offline `
  --data-root D:/path/to/underwater_objectnav_dataset/processed `
  --checkpoint work_dirs/objectnav_bc/best_policy.pt `
  --output work_dirs/objectnav_bc/offline_metrics.json
```

评估输出包括：

```text
overall_accuracy
per_class_accuracy
stop_precision
stop_recall
validation_loss
confusion_matrix
```

不能只看 overall accuracy，尤其要检查 `STOP` precision/recall 和四类动作混淆矩阵。

### 5.4 闭环推理

```python
from objectnav_bc.eval.eval_closed_loop import ClosedLoopPolicy

policy = ClosedLoopPolicy("work_dirs/objectnav_bc/best_policy.pt", device="cuda")
result = policy.predict(
    semantic_map=current_mapper.map,
    goal_category="sea_urchin",
    yaw=current_yaw,
)
print(result["action"], result["probabilities"])
```

ROS 节点应负责 RGB、Depth、Pose 同步和底层动作执行；本模块只负责地图更新和策略预测。推荐闭环顺序为：

```text
同步 RGB-D-Pose
  -> detector.detect(RGB)
  -> mapper.update(Depth, Pose, detections)
  -> policy.predict(mapper.map, goal, yaw)
  -> 底层控制器执行一个离散动作
  -> 等待动作完成后采集下一帧
```

## 6. 当前边界与后续扩展

当前版本明确只实现第一阶段：`Semantic Map -> CNN -> MLP -> BC`。以下功能暂未纳入策略训练：

- 直接将绝对位置 `x/y/z` 输入策略；
- ResNet50 + FPN 的 BC 反向微调；
- DAgger、GNN、Attention、ACT 或 Diffusion Policy；
- 具体 ROS 话题名称和底层推进器控制；
- 完整闭环成功率、碰撞次数和重复探索次数统计。

这些功能应在基础 BC 能稳定训练并完成离线评估后逐步加入。
