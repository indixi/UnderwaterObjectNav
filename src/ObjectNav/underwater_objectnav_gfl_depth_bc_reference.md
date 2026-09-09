# Underwater ObjectNav：GFL、深度建图与 BC 接入整理

本文整理本次 ObjectNav 改造过程中讨论的问题、最终采用的设计、代码对应位置，以及离线训练的完整使用方法。

当前范围是：

```text
冻结的 GFL 图像检测器
    + 对齐深度
    + 机器人世界位姿
    ↓
Global/Local 地图与当前帧目标提示
    ↓
Behavior Cloning 离线训练和离线评估
```

当前不包含 ROS 在线执行节点。专家动作也不重新生成，直接使用 `underwater_objectnav_dataset` 中已有的 `expert_action`。

---

## 1. 最初要解决的问题

图像检测部分已经完成：

```text
训练 GFL
  ↓
按 echinus precision≥95% 选择最佳 checkpoint
  ↓
在验证集上校准 echinus score 阈值
  ↓
在测试集上做最终检测评估
```

已有部署文件：

```text
best_echinus_recall_at_precision_95_epoch_5.pth
outputs/echinus_threshold.json
gfl_r50_fpn_underwater_objectnav.py
```

需要解决的核心问题是：如何把已经训练好的图像检测器冻结并接入 ObjectNav，同时让之后修改 ObjectNav 策略结构时不再依赖 `image_process_ResNet50` 工程，也不反复运行 GFL。

进一步需要确定：

1. GFL 输出什么数据，ObjectNav 真正使用其中哪些数据。
2. `echinus` 与 `rock` 的类别 ID 是否需要修改。
3. bbox 如何结合深度变为世界坐标语义点。
4. 如何构造固定世界坐标 Global Map 和机器人中心 Local Map。
5. 海胆不可见、深度无效、多只海胆等情况怎样处理。
6. `rock` 不是导航终点，是否仍然写入地图。
7. 如何控制计算量，满足后续每 2 s 至少输出一次动作。
8. 如何保证不同 episode 不发生地图状态污染和数据划分泄漏。

---

## 2. 最终采用的主要决策

### 2.1 类别命名

所有导航数据中的目标名统一为：

```text
echinus
```

不再使用 `sea_urchin`。`data_generate` 的默认元数据、使用示例和当前数据集根目录的 `metadata.yaml` 都已经相应修改。

### 2.2 类别 ID 不混用

系统中存在三套互相独立的编号：

| 编号空间 | echinus | rock | 说明 |
|---|---:|---:|---|
| GFL 检测标签 | 0 | 1 | 必须与训练 checkpoint 一致 |
| ObjectNav 导航目标 ID | 0 | 无 | 当前只有 echinus 是导航终点 |
| Global Map 通道 | 4 | 5 | 前四个通道是几何和轨迹信息 |
| Local Map 通道 | 2 | 1 | Local Map 使用独立通道顺序 |

因此不需要修改 GFL 的 `class_id`。GFL 内部仍保持：

```text
0 = echinus
1 = rock
```

导航目标 ID 为 0 并不意味着它必须等于某个地图通道编号。

### 2.3 rock 的作用

`rock` 不是最终导航目标，但会写入 Global/Local Map。它作为环境语义线索，能够帮助策略学习“在什么样的岩石分布附近更可能找到海胆”。

当前 rock 阈值使用：

```yaml
score_threshold: 0.50
```

以后可单独校准 rock，而不改变 echinus 阈值。

### 2.4 删除 Obstacle Distance 和 sand

没有把“最近障碍物距离”作为额外标量输入，因为单一距离不能表达是哪个方向、哪个障碍物，容易丢失空间关系。障碍物空间布局直接保存在地图中。

也没有增加 sand 通道，因为当前 GFL 不检测 sand，且增加没有可靠来源的维度会提高学习难度。

### 2.5 每个 episode 独立建图

- Global Map 和 Local Map 在一个 episode 内累积。
- 新 episode 开始前全部清零。
- `echinus` 和 `rock` 的同一地图位置使用历史最大置信度，不做求和。
- 当前机器人位置每帧清除后重新写入。
- visited 轨迹持续累积。

这样不会把上一场景的语义和轨迹带入下一场景。

---

## 3. GFL 最终输出及代码位置

### 3.1 GFL 内部输出

GFL 检测头内部预测：

- 各候选位置的类别/质量分数；
- 离散边界框距离分布；
- 由分布解码得到的边界框。

这些是检测模型内部张量，不直接送进 BC 网络。

### 3.2 ObjectNav 使用的后处理结果

经过 bbox 解码和 NMS 后，ObjectNav 对每个检测目标只保留：

```python
Detection(
    bbox=(x1, y1, x2, y2),
    score=confidence,
    class_name="echinus" or "rock",
)
```

对应代码：

```text
objectnav_bc/perception/semantic_detector.py
```

其中：

- `detect_candidates()` 获得 `score >= 0.001` 的 NMS 后候选结果；
- `filter_detections()` 再按照各类别的部署阈值过滤；
- 模型输出 label 会根据固定顺序转换成 `echinus` 或 `rock`；
- 如果 checkpoint 的类别顺序不是 `("echinus", "rock")`，程序直接报错。

因此，GFL 特征图、FPN 特征和 bbox 距离分布不会直接输入 BC。BC 使用的是检测框与深度融合后产生的地图和 Target Cue。

---

## 4. 为什么使用两级检测缓存

检测与地图预处理拆成两步：

```text
第一级：GFL + bbox 解码 + NMS + 0.001 候选阈值
                   ↓
             detections.jsonl
                   ↓
第二级：echinus 校准阈值 / rock 0.5 阈值
                   ↓
           深度融合和地图构建
```

这样做的好处：

- 修改 echinus 或 rock 部署阈值时，不需要重新运行 GFL；
- 修改地图范围、分辨率、ROI 或网络结构时，不需要重新训练检测器；
- 检测模型配置、checkpoint 哈希和类别顺序会写进缓存元数据，避免误用缓存；
- 只有更换 GFL 配置、checkpoint 或候选阈值时才必须重新生成检测缓存。

缓存中每帧保存：

```json
{
  "episode_id": "episode_0001",
  "step_id": 0,
  "rgb_path": "episode_0001/rgb/000000.png",
  "boxes_xyxy": [[x1, y1, x2, y2]],
  "scores": [0.86],
  "labels": [0],
  "class_names": ["echinus"]
}
```

---

## 5. 相机、深度和世界坐标

### 5.1 相机参数

当前 YAML 使用已确认的 Stonefish 参数：

```yaml
camera:
  width: 640
  height: 480
  fx: 457.01
  fy: 457.01
  cx: 319.5
  cy: 239.5
  nominal_sensor_world_z_m: 3.3
  sensor_height_tolerance_m: 0.10
  T_base_camera:
    translation_m: [0.20, 0.0, -0.02]
    rpy_rad: [1.5707963268, 0.0, 1.5707963268]
```

RGB 和 Depth 都必须是 640×480，并且已经对齐。

### 5.2 像素投影

深度像素首先转换到相机坐标：

```text
Xc = (u - cx) × depth / fx
Yc = (v - cy) × depth / fy
Zc = depth
```

再转换到世界坐标：

```text
P_world = T_world_base × T_base_camera × P_camera
```

代码位置：

```text
objectnav_bc/perception/depth_projection.py
```

相机中心像素沿相机光轴投影。使用当前 Stonefish 外参时，该方向对应机器人前方。

### 5.3 传感器高度

`3.3 m` 表示传感器标称世界 Z 高度，不是目标距离，也不是机器人离海底高度。

每帧实际传感器高度由下式计算：

```text
T_world_camera = T_world_base × T_base_camera
sensor_world_z = T_world_camera[2, 3]
```

标称 `3.3 m` 用于检查实际结果是否异常。超过 YAML 中 `0.10 m` 容差时发出 warning；障碍物高度带仍相对于实际计算出的传感器高度。

### 5.4 NED 坐标和障碍物高度

当前使用 NED 坐标，世界 Z 正方向向下。障碍物候选点必须满足：

```text
sensor_world_z - 0.15
    <= point_world_z <=
sensor_world_z + 0.19
```

这样只保留接近机器人碰撞高度的深度点，不把全部海底或远离机器人高度的点都写成障碍物。

深度默认每隔 4 个像素采样一次，有效范围为 0.2～10 m。

---

## 6. bbox 与深度融合

对应代码：

```text
objectnav_bc/perception/detection_depth_fusion.py
objectnav_bc/perception/depth_obstacle_extractor.py
```

### 6.1 echinus

对每个 echinus bbox：

1. 取 bbox 中心区域，宽高比例为原框的 0.4。
2. 中心区域最小尺寸为 5 px。
3. 至少需要 3 个有效深度像素。
4. 使用有效深度的中位数作为目标深度。
5. 使用 bbox 中心像素和中位深度投影一个世界坐标点。
6. 在地图上以该点为圆心、0.1 m 为半径写入检测置信度。

如果一个画面中有多只有效海胆：

- 所有海胆都写入地图；
- Target Cue 使用有效深度最近的一只。

如果海胆检测框没有足够有效深度，则这个框不写入地图，也不产生可见目标提示。

### 6.2 rock

对每个 rock bbox：

1. 使用中心 0.4 区域的深度中位数作为参考深度。
2. 在整个 bbox 内每隔 4 个像素采样。
3. 只保留与参考深度相差小于 0.2 m 的像素。
4. 将这些像素投影到世界坐标并写入 rock 通道。

这个深度门限可以减少 bbox 中背景深度被错误写成 rock 的问题。

### 6.3 当前帧 Target Cue

Target Cue 原始形式为：

```text
[visible, bearing_rad, distance_m]
```

规则：

- 只有当前帧存在带有效深度的 echinus 时 `visible=1`；
- 不累积上一帧的 visible、bearing 或 distance；
- 无有效目标时三项全部为 0；
- bearing 根据 bbox 中心像素计算；
- distance 使用中心 ROI 深度中位数。

送入网络前归一化为：

```text
visible                       -> 0 或 1
bearing / 35°                 -> 裁剪至 [-1, 1]
distance / 10 m               -> 裁剪至 [0, 1]
```

然后通过 `Linear(3,16) + ReLU` 学习一个 16 维目标提示特征。

---

## 7. Global Map

Global Map 使用固定世界坐标范围，参数允许人工配置：

```yaml
map:
  global:
    origin_x: -7.0
    origin_y: -4.0
    width_m: 14.0
    height_m: 8.0
    resolution_m: 0.10
```

默认张量形状：

```text
[6, 80, 140]
```

通道顺序：

| 通道 | 名称 | 数据含义 |
|---:|---|---|
| 0 | obstacle | 碰撞高度范围内的深度点 |
| 1 | explored | 相机到有效深度端点之间的可见区域 |
| 2 | visited | episode 内累计经过的位置 |
| 3 | robot | 当前机器人位置 |
| 4 | echinus | 海胆位置及检测置信度 |
| 5 | rock | 岩石位置及检测置信度 |

`explored` 不只标记深度端点，还标记相机到端点之间的射线区域。为控制开销，每个采样图像列保留一条最远有效射线。

对应代码：

```text
objectnav_bc/mapping/global_semantic_map.py
objectnav_bc/mapping/explored_visibility.py
```

---

## 8. Local Map

Local Map 以机器人为中心，并随着机器人 yaw 改变采样方向：

```yaml
map:
  local:
    width_m: 3.0
    height_m: 3.0
    resolution_m: 0.05
```

默认张量形状：

```text
[4, 60, 60]
```

通道顺序：

| 通道 | 名称 |
|---:|---|
| 0 | obstacle |
| 1 | rock |
| 2 | echinus |
| 3 | visited |

Local Map 图像上方始终表示机器人当前前方，右侧表示机器人当前右方。

内部先维护一张覆盖整个全局范围的 0.05 m 高分辨率 episode 证据图，再围绕机器人位置旋转采样出 3×3 m Local Map。这样可以同时实现：

- 保留 episode 内历史局部证据；
- 机器人始终位于局部地图中心；
- 空间布局不会被直接压缩成一个均值向量。

在当前 NED yaw 约定中：

- `TURN_RIGHT` 后 yaw 增大；
- `TURN_LEFT` 后 yaw 减小；
- 机器人左转时，固定世界环境在 Local Map 上顺时针移动。

对应代码：

```text
objectnav_bc/mapping/local_egocentric_map.py
```

---

## 9. 为什么 CNN 池化后仍能保留空间布局

Global/Local Map 没有直接做全图平均，而是分别使用：

```text
Global: AdaptiveAvgPool(5, 9)
Local:  AdaptiveAvgPool(4, 4)
```

例如 Global Map 被划分为 5×9 个粗略空间区域。每个区域分别保留自己的特征，随后按照固定位置展平并送入 Linear。因此网络仍能区分：

```text
目标在左上区域
目标在右下区域
障碍物在机器人前方
岩石集中在某个世界区域
```

如果使用 `AdaptiveAvgPool(1,1)`，整张地图会只剩每个特征通道的全局均值，空间布局才会大幅丢失。

---

## 10. BC 网络输入与输出

网络输入由五部分组成：

```text
Global Map -> Global CNN -> 256 维
Local Map  -> Local CNN  -> 128 维
goal_id    -> Embedding  ->   8 维
Target Cue -> Linear     ->  16 维
yaw        -> sin/cos    ->   2 维
```

拼接后：

```text
256 + 128 + 8 + 16 + 2 = 410 维
```

再经过：

```text
Linear(410,256)
ReLU
Linear(256,128)
ReLU
Linear(128,4)
```

输出四个动作 logits：

```text
0 FORWARD
1 TURN_LEFT
2 TURN_RIGHT
3 STOP
```

训练使用带类别权重的交叉熵，缓解不同专家动作样本数量不平衡。

对应代码：

```text
objectnav_bc/models/global_map_encoder.py
objectnav_bc/models/local_map_encoder.py
objectnav_bc/models/target_encoder.py
objectnav_bc/models/policy_mlp.py
objectnav_bc/models/factory.py
```

---

## 11. 数据筛选和划分

原始数据目录：

```text
catkin_ws/underwater_objectnav_dataset/
```

实际检查结果：

```text
成功 episode：59
成功 episode 中的 step：1046
goal_category：全部为 echinus
expert_action：四种动作都存在
```

筛选规则：

- 默认只读取 `episode.yaml` 中 `success: true` 的 episode；
- 所有名字以 `episode_false` 开头的目录始终忽略；
- 每个 episode 的 `step_id` 必须从 0 连续增长；
- trajectory 必须包含当前 `data_generate` 实际生成的全部字段。

需要的 CSV 字段：

```text
step_id,action_id,action_start_time,timestamp,rgb_path,depth_path,
x,y,z,roll,pitch,yaw,goal_category,expert_action
```

划分方式：

```text
按 episode 打乱
随机种子 42
train / val / test = 70% / 15% / 15%
```

不能先把所有帧混在一起再随机划分，否则同一轨迹相邻帧可能同时出现在训练集和测试集，造成数据泄漏。

---

## 12. 文件放置

在另一台主机复制以下两个文件：

```text
best_echinus_recall_at_precision_95_epoch_5.pth
echinus_threshold.json
```

放到：

```text
UnderwaterObjectNav/src/ObjectNav/objectnav_bc/perception/model/
```

最终目录应为：

```text
objectnav_bc/perception/model/
  gfl_r50_fpn_underwater_objectnav_inference.py
  best_echinus_recall_at_precision_95_epoch_5.pth
  echinus_threshold.json
```

`gfl_r50_fpn_underwater_objectnav_inference.py` 是从训练配置中提取的自包含推理配置，不含训练数据路径、自定义训练 metric 或训练 hook，因此 ObjectNav 不会导入 `image_process_ResNet50`。

所有路径都在以下 YAML 中统一配置：

```text
objectnav_bc/config/bc.yaml
```

YAML 中的相对路径以 `bc.yaml` 所在目录为基准，不以终端当前目录为基准。

---

## 13. 环境准备

在以下目录执行命令：

```text
UnderwaterObjectNav/src/ObjectNav
```

PowerShell：

```powershell
$env:PYTHONPATH = (Get-Location).Path
pip install -r objectnav_bc/requirements.txt
```

此外，需要安装与 GFL 训练主机兼容的 MMDetection、MMEngine、MMCV 和 CUDA/PyTorch 版本。PyTorch 的 CUDA 版本应根据当前 RTX 4060 环境安装，不建议仅依赖普通 `pip install torch` 自动选择。

---

## 14. 第一步：生成 GFL 候选缓存

```powershell
python -m objectnav_bc.dataset.cache_detections `
  --dataset-root ../../../underwater_objectnav_dataset `
  --output-root ../../../underwater_objectnav_dataset/detection_cache `
  --config objectnav_bc/config/bc.yaml
```

输出：

```text
underwater_objectnav_dataset/detection_cache/
  detections.jsonl
  detection_cache_metadata.json
```

程序会：

1. 加载冻结 GFL checkpoint。
2. 设置模型为 `eval()`。
3. 禁止参数梯度。
4. 对全部成功 episode 的 RGB 帧推理。
5. 保存 score≥0.001 的 NMS 后候选框。
6. 保存配置和 checkpoint 的 SHA-256 签名。

如果缓存已存在且模型签名完全一致，会直接复用。如果更换了模型，需要显式重新运行：

```powershell
python -m objectnav_bc.dataset.cache_detections `
  --dataset-root ../../../underwater_objectnav_dataset `
  --output-root ../../../underwater_objectnav_dataset/detection_cache `
  --config objectnav_bc/config/bc.yaml `
  --force
```

---

## 15. 第二步：生成 Global/Local 地图

```powershell
python -m objectnav_bc.dataset.preprocess_dataset `
  --dataset-root ../../../underwater_objectnav_dataset `
  --detection-cache ../../../underwater_objectnav_dataset/detection_cache/detections.jsonl `
  --output-root ../../../underwater_objectnav_dataset/processed `
  --config objectnav_bc/config/bc.yaml
```

输出结构：

```text
processed/
  metadata.json
  train.jsonl
  val.jsonl
  test.jsonl
  episode_0001/
    global_map/000000.npy
    local_map/000000.npy
```

磁盘上的地图为 `float16`，减少空间占用；PyTorch Dataset 加载后自动转换为 `float32`。

JSONL 中每一步包含：

```text
Global/Local 地图路径
episode_id 和 step_id
goal_category 和 goal_id
target_visible/bearing/distance
sensor_world_z
yaw
expert action ID
```

调整下列参数时可以复用 GFL 缓存，只重新运行本步骤：

- echinus/rock 部署阈值；
- 地图范围或分辨率；
- bbox 深度 ROI；
- 障碍物高度范围；
- 相机内外参；
- 地图标记半径。

---

## 16. 第三步：离线训练 BC

```powershell
python -m objectnav_bc.train.train_bc `
  --data-root ../../../underwater_objectnav_dataset/processed `
  --work-dir work_dirs/objectnav_bc `
  --device cuda
```

常用覆盖参数：

```powershell
--batch-size 32
--lr 0.0003
--epochs 30
--workers 2
--seed 42
```

不提供时读取 `bc.yaml` 的 `training` 配置。

输出：

```text
work_dirs/objectnav_bc/
  best_policy.pt
  latest_policy.pt
  history.json
```

checkpoint 中同时保存：

- Global Encoder 权重；
- Local Encoder 权重；
- Policy/Goal/Target 权重；
- 模型维度配置；
- 动作顺序、目标顺序和地图通道顺序；
- Target Cue 归一化配置。

训练器会检查 processed 数据的通道顺序和模型配置，避免把旧地图或不同结构的 checkpoint 静默混用。

---

## 17. 第四步：离线测试集评估

```powershell
python -m objectnav_bc.eval.eval_offline `
  --data-root ../../../underwater_objectnav_dataset/processed `
  --checkpoint work_dirs/objectnav_bc/best_policy.pt `
  --output work_dirs/objectnav_bc/offline_metrics.json `
  --device cuda
```

输出指标：

```text
overall_accuracy
per_class_accuracy
stop_precision
stop_recall
test_loss
confusion_matrix
```

除了整体准确率，还应重点检查：

- TURN_LEFT 与 TURN_RIGHT 是否大量混淆；
- STOP precision 是否过低，避免提前停止；
- STOP recall 是否过低，避免找到目标后仍继续移动；
- 类别加权后是否出现某个动作被过度预测。

---

## 18. 修改不同参数后需要重跑哪些步骤

| 修改内容 | 重跑 GFL 缓存 | 重建地图 | 重新训练 BC |
|---|---:|---:|---:|
| echinus/rock 部署阈值 | 否 | 是 | 是 |
| 地图范围、分辨率、通道含义 | 否 | 是 | 是 |
| 深度 ROI、障碍物高度参数 | 否 | 是 | 是 |
| BC CNN/MLP 结构 | 否 | 当前安全检查下需要更新 processed metadata | 是 |
| GFL checkpoint | 是 | 是 | 是 |
| GFL 配置或类别顺序 | 是 | 是 | 是 |
| 候选缓存阈值 0.001 | 是 | 是 | 是 |

最重要的解耦点是：修改 ObjectNav 地图或策略结构不会重新训练 GFL；只要 GFL 本身没变，低阈值候选缓存仍可复用。

---

## 19. 当前性能检查

在当前开发主机上，使用 640×480 合成深度、一个 echinus 框和一个 rock 框测试，不含 GFL 的以下处理：

```text
深度稀疏投影
障碍物高度过滤
bbox/depth 融合
Global Map 更新
Local Map 旋转采样
```

平均约为：

```text
7 ms / frame
```

因此相对于 2 s 一个原子动作的要求，地图部分有较大余量。后续在线阶段主要需要实测的是 RTX 4060 上 GFL 推理、RGB-D 同步和数据传输总耗时。

---

## 20. 已完成的测试

运行：

```powershell
python -B -m unittest discover -s tests -v
```

当前测试覆盖：

- echinus 校准阈值和 rock 0.5 阈值；
- 检测 label 与 class name 缓存一致性；
- Stonefish 相机外参投影方向；
- echinus 深度无效时 Target Cue 全零；
- 地图置信度采用最大值；
- episode reset 清空历史地图；
- `episode_false*` 和失败 episode 筛选；
- NED 下 TURN_LEFT 对应局部地图中的顺时针环境移动；
- 安装 PyTorch 后的 Global/Local/Policy 张量形状。

当前开发环境测试结果为 8 项通过、1 项因未安装 PyTorch 自动跳过；全部 36 个 Python 文件通过语法解析。

---

## 21. 常见错误和检查方法

### 21.1 找不到 checkpoint 或 threshold JSON

检查：

```text
objectnav_bc/perception/model/
```

以及 `bc.yaml` 中三条相对路径。

### 21.2 threshold/checkpoint mismatch

`echinus_threshold.json` 内记录的 checkpoint 文件名与当前配置的 checkpoint 不同。必须复制同一次校准对应的权重和 JSON，不能随意组合。

### 21.3 detector class order mismatch

checkpoint 或配置中类别顺序不是：

```text
(echinus, rock)
```

不能只交换 YAML 顺序，因为模型输出层的 label 含义由训练时决定。

### 21.4 detection cache model mismatch

当前缓存由另一个配置或 checkpoint 产生。更换模型后使用 `--force` 重新生成候选缓存。

### 21.5 depth shape 不一致

当前配置要求：

```text
depth.shape == (480, 640)
```

如果实际分辨率变化，必须同时修改 width、height、fx、fy、cx、cy，不能只改图像尺寸。

### 21.6 sensor world z warning

说明由机器人位姿和相机外参计算出的传感器世界 Z 与 3.3 m 相差超过 0.1 m。应检查：

- 位姿是否为世界坐标；
- z 是否使用 NED；
- roll/pitch/yaw 是否为弧度；
- `T_base_camera` 方向是否写反；
- 位姿时间是否与深度帧对应。

### 21.7 unsupported goal_category

trajectory 中仍然存在 `sea_urchin` 或其他未定义名称。当前必须统一为：

```text
echinus
```

### 21.8 地图边界外数据被丢弃

如果机器人或目标世界坐标超过：

```text
x ∈ [-7, 7)
y ∈ [-4, 4)
```

对应点不会写入 Global Map。场景范围改变时应调整 `origin_x`、`origin_y`、`width_m` 和 `height_m`。

---

## 22. 后续在线 ROS 接入边界

未来在线阶段可以复用完全相同的四个模块：

```text
build_semantic_detector()
SemanticMappingPipeline.reset()
SemanticMappingPipeline.update()
ClosedLoopPolicy.predict()
```

推荐在线流程：

```text
新 episode
  -> mapper.reset()

每个动作周期：
  同步 RGB + Depth + Pose
    -> detector.detect(RGB)
    -> mapper.update(Depth, Pose, detections, robot_xy, yaw)
    -> policy.predict(Global Map, Local Map, Target Cue, goal, yaw)
    -> 执行一个约 2 s 原子动作
```

ROS 层以后只负责：

- RGB、Depth、Pose 时间同步；
- episode 开始/结束通知；
- 调用冻结检测器和建图模块；
- 调用策略并下发离散动作；
- 超时、碰撞和安全停止。

ROS 层不应该重新实现类别阈值、bbox/depth 融合或地图通道逻辑，否则离线训练与在线推理会产生不一致。

---

## 23. 关键代码索引

| 功能 | 文件 |
|---|---|
| 统一 YAML 读取与检查 | `objectnav_bc/config_loader.py` |
| 所有人工参数 | `objectnav_bc/config/bc.yaml` |
| GFL 推理适配与阈值过滤 | `objectnav_bc/perception/semantic_detector.py` |
| 检测缓存格式和签名 | `objectnav_bc/perception/detection_cache.py` |
| 生成检测缓存 | `objectnav_bc/dataset/cache_detections.py` |
| 相机坐标投影 | `objectnav_bc/perception/depth_projection.py` |
| bbox 与深度融合 | `objectnav_bc/perception/detection_depth_fusion.py` |
| 障碍物点提取 | `objectnav_bc/perception/depth_obstacle_extractor.py` |
| Global Map | `objectnav_bc/mapping/global_semantic_map.py` |
| Local Map | `objectnav_bc/mapping/local_egocentric_map.py` |
| 统一建图入口 | `objectnav_bc/mapping/mapping_pipeline.py` |
| episode 数据读取 | `objectnav_bc/dataset/episode_io.py` |
| 地图数据预处理 | `objectnav_bc/dataset/preprocess_dataset.py` |
| PyTorch Dataset | `objectnav_bc/dataset/bc_dataset.py` |
| 网络构建 | `objectnav_bc/models/factory.py` |
| BC 训练 | `objectnav_bc/train/train_bc.py` |
| 离线评估 | `objectnav_bc/eval/eval_offline.py` |
| 后续在线策略适配器 | `objectnav_bc/eval/eval_closed_loop.py` |

---

## 24. 当前尚未完成的外部条件

代码已经完成，但当前工作区还没有以下两个外部文件：

```text
best_echinus_recall_at_precision_95_epoch_5.pth
echinus_threshold.json
```

当前开发环境也没有安装 PyTorch/MMDetection。因此现阶段可以验证配置、数据读取、深度投影和地图逻辑，但只有复制权重并安装匹配的推理环境后，才能真正生成 1046 帧的 GFL 缓存并开始完整 BC 训练。
