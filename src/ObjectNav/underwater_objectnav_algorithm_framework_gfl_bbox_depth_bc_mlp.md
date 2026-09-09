# 水下 ObjectNav 算法框架（GFL + BBox-Depth + BC + MLP）

## 1. 任务定义

目标类别：

```text
echinus
```

环境语义类别：

```text
rock
```

离散动作：

```text
FORWARD
TURN_LEFT
TURN_RIGHT
STOP
```

训练方式：

```text
Human Expert Demonstration
→ Behavior Cloning
→ MLP Policy
```

第一版不使用 A*、FMM、Frontier Planner、Safety Stop、GNN、Transformer、ACT、Diffusion 或 Flow Matching。

---

## 2. 总体流程

```text
RGB + Depth + Robot Pose
          ↓
         GFL
          ↓
Detection(bbox, score, class_name)
          ↓
 ┌────────┴─────────┐
 │                  │
echinus bbox       rock bbox
 │                  │
Depth ROI          Depth ROI 稀疏采样
 │                  │
目标3D位置          Rock 3D点
 │                  │
 └────────┬─────────┘
          ↓
      Semantic Layer
 echinus / rock / robot /
 explored / visited
          +
Sparse Depth Geometry
          ↓
只提取进入机器人碰撞高度包络的点
          ↓
      Obstacle Layer
          ↓
Global Semantic-Geometric Map
+
Local Egocentric Map
          ↓
Global CNN + Local CNN
          +
Goal Embedding
Target Cue
Orientation
          ↓
       MLP Policy
          ↓
FORWARD / TURN_LEFT /
TURN_RIGHT / STOP
```

核心原则：

```text
GFL Detection
→ 决定“是什么”

BBox + Depth + Pose
→ 决定“在哪里”

Sparse Depth Geometry
→ 决定“是否构成当前运动障碍”
```

不再把整幅 Depth 转换成完整语义地图。

---

## 3. GFL 输出接口

ObjectNav 使用 GFL NMS 后结果：

```python
Detection(
    bbox=(x1, y1, x2, y2),
    score=0.86,
    class_name="echinus"  # or "rock"
)
```

类别：

```text
0 = echinus
1 = rock
```

过滤规则：

```text
echinus:
使用 echinus_threshold.json 推荐阈值

rock:
score >= 0.5
```

RGB 与 Depth：

```text
分辨率一致
像素已对齐
```

因此 RGB bbox 可以直接映射到同一区域的 Depth ROI。

---

## 4. 专家行为规则

### 4.1 启动

```text
原地 TURN_LEFT / TURN_RIGHT
→ 观察周围
→ 更新地图
→ 判断 target_visible
```

### 4.2 第一优先级：echinus 可见

```text
1. 多个 echinus 同时出现时，优先选择距离最近的目标
2. 根据 target_bearing 左右旋转
3. 将目标调整到相机中心附近
4. 移动过程中根据 Obstacle 信息学习避开 rock / 其他障碍
5. 目标基本居中后 FORWARD
6. 重复观察、旋转、前进
7. target_distance <= success_distance 时 STOP
```

### 4.3 第二优先级：echinus 不可见

```text
1. 根据 Global Rock Channel 判断岩石分布
2. 优先朝 Rock-rich Direction 移动
3. 连续移动一段距离 / 若干动作
4. 再原地旋转观察
5. 如果发现 echinus，切换到第一优先级
6. 如果仍未发现，根据更新后的地图继续选择 Rock-rich Direction
```

核心：

```text
echinus 不可见
→ 朝岩石更多区域移动
→ 移动一段
→ 原地观察
→ 再决策
```

### 4.4 无明显 rock 线索

```text
echinus 不可见
+
没有明显 Rock-rich Direction
```

则：

```text
优先探索未观察方向
+
周期性原地旋转观察
```

---

## 5. 地图总体设计

采用：

```text
Global Semantic-Geometric Map
+
Local Egocentric Map
```

其中：

```text
Global Map
→ 决定“往哪里搜索”

Local Map
→ 决定“当前一步怎么移动”
```

---

## 6. Global Semantic-Geometric Map

建议：

```text
resolution = 0.05 ~ 0.10 m / cell
```

通道：

```text
Channel 0 : Obstacle
Channel 1 : Explored
Channel 2 : Visited
Channel 3 : Robot
Channel 4 : Echinus
Channel 5 : Rock
Channel 6 : Obstacle Distance
```

即：

```text
Global Map Channels = 7
```

取消 Sand Semantic Channel。

原因：

```text
当前 GFL 不识别 sand；
sand 不是目标搜索需要的语义先验；
无关地形不写入语义层。
```

重要：

```text
Semantic != Obstacle
```

同一格允许：

```text
Rock     = 1
Echinus  = 1
Obstacle = 1
```

也允许：

```text
Rock     = 1
Obstacle = 0
```

是否构成障碍由 3D 高度和机器人碰撞包络决定。

---

## 7. Local Egocentric Map

建议：

```text
范围：机器人周围 2 ~ 4 m
分辨率：0.025 ~ 0.05 m / cell
```

机器人位于 Local Map 中心。

通道：

```text
Channel 0 : Obstacle
Channel 1 : Rock
Channel 2 : Echinus
Channel 3 : Visited
Channel 4 : Obstacle Distance
```

即：

```text
Local Map Channels = 5
```

第一版不设置 Sand / Free Space Semantic Channel。

---

## 8. Echinus BBox + Depth 融合

GFL 输出：

```text
bbox=(x1,y1,x2,y2)
score
class_name="echinus"
```

处理：

```text
Echinus bbox
→ 取 bbox 中心附近小 ROI
→ 查询 Depth
→ 过滤 NaN / 0 / 过近 / 过远
→ 有效 Depth 中位数
→ target_distance
→ bbox 中心计算 target_bearing
→ 代表像素 + Depth 反投影到 Camera 3D
→ Camera → Base → World
→ 更新 Echinus Channel
```

建议：

```text
Depth ROI = bbox 中心 5×5 / 7×7
```

或者 bbox 中心 30% ~ 50% 区域。

海胆为小目标，第一版只保存代表性世界位置，不做整框点云。

---

## 9. Rock BBox + Depth 融合

GFL 输出：

```text
bbox=(x1,y1,x2,y2)
score
class_name="rock"
```

bbox 只作为候选区域，不能把整个矩形直接写入 Rock Map。

处理：

```text
Rock bbox
→ Depth ROI
→ 稀疏采样
→ 有效深度过滤
→ 深度一致性筛选
→ 保留可能属于 rock 的深度点
→ 每个点反投影到 3D
→ Camera → Base → World
→ 更新 Rock Channel
```

### 9.1 稀疏采样

```python
depth_roi = depth[y1:y2:4, x1:x2:4]
```

建议：

```text
rock_depth_stride = 4
```

### 9.2 sand / 背景深度过滤

bbox 中可能同时包含：

```text
rock
sand
background
```

第一版：

```text
bbox 中心 30% ~ 50% 区域
→ 有效 Depth Median
→ d_ref
```

再对 bbox 稀疏采样点：

```text
abs(d_i - d_ref) < rock_depth_gate
```

例如：

```yaml
rock_depth_gate_m: 0.20
```

最终：

```text
bbox = 检测候选区域
Depth 筛选后的点 = 真正写入 Rock Map 的空间点
```

因此 Rock Map 不会简单变成矩形。

---

## 10. Sparse Depth Geometry → Obstacle

除 GFL 检测框定位外，Depth 还用于构建几何障碍层。

但不做：

```text
整幅 Depth
→ 所有点
→ 完整语义地图
```

而做：

```text
Depth Image
→ 稀疏采样
→ 3D Point
→ World Coordinate
→ 机器人碰撞高度过滤
→ Obstacle Map
```

原则：

```text
sand / seabed：
如果位于机器人运动碰撞包络之外
→ Ignore

rock / 凸起 / 其他几何物：
如果进入机器人运动碰撞高度范围
→ Obstacle = 1
```

因此：

```text
Depth 不负责语义分类
Depth Geometry 只负责判断能不能通过
```

---

## 11. Obstacle 高度过滤

设：

```text
z_robot
robot_half_height
height_margin
```

只有世界点进入：

```text
[z_robot - robot_half_height - height_margin,
 z_robot + robot_half_height + height_margin]
```

才参与当前二维 Obstacle Map。

Stonefish 的 z 正负方向必须按当前场景坐标系统一实现。

---

## 12. Obstacle Depth 稀疏采样

第一版建议：

```yaml
obstacle_depth_stride: 4
```

或：

```yaml
obstacle_depth_stride: 8
```

例如：

```text
640×480
→ stride=4
→ 160×120 samples
→ 有效点
→ 高度过滤
→ Obstacle Map
```

这些点只更新：

```text
Obstacle
```

不会自动更新：

```text
Rock
Echinus
```

---

## 13. Robot / Visited / Explored

### Robot

根据 Robot Pose 更新 Robot Channel。

### Visited

机器人经过的地图格累计：

```text
Visited = 1
```

### Explored

第一版根据：

```text
Robot Pose
+
Camera FOV
+
有效 Depth
```

更新已观察区域。

Explored 仅表示：

```text
“这个区域看过”
```

不表示：

```text
“这里是什么语义”
```

---

## 14. Obstacle Distance

```text
Obstacle Map
→ Distance Transform
→ Obstacle Distance Map
```

该通道作为 Policy 输入，不单独实现 Safety Stop。

---

## 15. Target Cue

输入：

```text
target_visible
target_bearing
target_distance
```

### target_visible

存在通过阈值过滤的 echinus Detection：

```text
1
```

否则：

```text
0
```

### 多目标选择

```text
每个 echinus bbox
→ 估计 Depth
→ 选择 target_distance 最小的有效目标
```

### target_bearing

bbox 中心：

```text
u_c = (x1 + x2) / 2
```

相机内参：

```text
bearing = atan((u_c - cx) / fx)
```

### target_distance

目标 bbox 中心 ROI 有效 Depth 中位数。

若：

```text
target_visible = 0
```

则：

```text
target_bearing  = 0
target_distance = 0
```

---

## 16. 像素 + Depth → 世界坐标

RGB 与 Depth 已对齐。

```text
Xc = (u - cx) * d / fx
Yc = (v - cy) * d / fy
Zc = d
```

得到 Camera Point：

```text
Pc
```

然后：

```text
Pw = T_world_base @ T_base_camera @ Pc
```

该投影只用于：

```text
1. echinus / rock 检测区域的语义定位
2. 稀疏 Depth 几何障碍点
```

不要求生成完整稠密点云。

---

## 17. Policy 输入

最终：

```text
Global Map
Local Map
Goal Category
target_visible
target_bearing
target_distance
yaw
```

---

## 18. 网络结构

### Global Encoder

输入：

```text
[7, Hg, Wg]
```

输出：

```text
256-D
```

### Local Encoder

输入：

```text
[5, Hl, Wl]
```

输出：

```text
128-D
```

### Goal Embedding

```text
Goal ID
→ Embedding
→ 8-D
```

### Target Feature

```text
[target_visible,
 target_bearing,
 target_distance]
→ Linear(3,16)
→ ReLU
→ 16-D
```

### Orientation

```text
sin(yaw)
cos(yaw)
```

得到：

```text
2-D
```

### Feature Fusion

```text
Global Feature       256
Local Feature        128
Goal Feature           8
Target Feature        16
Orientation Feature    2
------------------------
Total                 410
```

### MLP Policy

```text
410
→ Linear(410,256)
→ ReLU
→ Linear(256,128)
→ ReLU
→ Linear(128,4)
```

输出：

```text
0 FORWARD
1 TURN_LEFT
2 TURN_RIGHT
3 STOP
```

---

## 19. Behavior Cloning

训练样本：

```text
(state_t, expert_action_t)
```

损失：

```python
CrossEntropyLoss(logits, expert_action)
```

第一版训练：

```text
Global CNN
Local CNN
Goal Embedding
Target Feature Encoder
MLP Policy
```

GFL Detector 冻结。

---

## 20. 专家行为状态

建议额外记录：

```text
behavior_state
```

可选：

```text
SEARCH
OBSERVE
APPROACH
AVOID
```

第一版只用于分析。

### SEARCH

```text
target_visible = 0
→ 根据 Global Rock Channel 找 Rock-rich Direction
→ 移动
```

### OBSERVE

```text
移动一段
→ 原地旋转
→ 更新 Detection 和地图
```

### APPROACH

```text
target_visible = 1
→ 对准最近 echinus
→ FORWARD
→ STOP
```

### AVOID

```text
Obstacle / Obstacle Distance 表明前方风险
→ 人类专家示范 TURN_LEFT / TURN_RIGHT
```

---

## 21. STOP

仅表示任务完成：

```text
target_visible = 1
+
target_distance <= success_distance
→ STOP
```

不能因为 rock 太近而标记 STOP。

---

## 22. 推理流程

```text
RGB + Depth + Pose
        ↓
       GFL
        ↓
echinus / rock Detection
        ↓
Detection ROI + Depth
        ↓
更新 Echinus / Rock Semantic Channels
        +
Sparse Depth Geometry
        ↓
更新 Obstacle
        ↓
更新 Explored / Visited / Robot
        ↓
Obstacle Distance
        ↓
Global Map + Local Map
        ↓
target_visible
target_bearing
target_distance
        ↓
Global CNN + Local CNN
+ Goal + Target Feature + yaw
        ↓
MLP Policy
        ↓
FORWARD / TURN_LEFT / TURN_RIGHT / STOP
        ↓
执行一个固定动作原语
        ↓
下一 Observation
```

---

## 23. 第一版代码模块

```text
objectnav_bc/
│
├── perception/
│   ├── gfl_semantic_detector.py
│   ├── detection_depth_fusion.py
│   └── depth_obstacle_extractor.py
│
├── mapping/
│   ├── global_semantic_map.py
│   ├── local_egocentric_map.py
│   └── obstacle_distance.py
│
├── dataset/
│   ├── preprocess_dataset.py
│   └── bc_dataset.py
│
├── models/
│   ├── global_map_encoder.py
│   ├── local_map_encoder.py
│   └── policy_mlp.py
│
├── train/
│   └── train_bc.py
│
├── eval/
│   ├── eval_offline.py
│   └── eval_closed_loop.py
│
└── config/
    └── objectnav_bc.yaml
```

---

## 24. 第一版配置建议

```yaml
classes:
  echinus: 0
  rock: 1

thresholds:
  echinus_threshold_file: echinus_threshold.json
  rock_score_threshold: 0.5

depth:
  min_depth_m: 0.05
  max_depth_m: 30.0
  rock_depth_stride: 4
  obstacle_depth_stride: 4
  rock_depth_gate_m: 0.20

map:
  global_resolution_m: 0.05
  local_resolution_m: 0.025
  local_size_m: 3.0

policy:
  actions:
    - FORWARD
    - TURN_LEFT
    - TURN_RIGHT
    - STOP
```

具体参数后续通过仿真实验调节。

---

## 25. 当前方案核心

语义层：

```text
GFL echinus / rock Detection
→ 只对有语义价值的 bbox 区域结合 Depth 定位
→ Echinus / Rock Semantic Map
```

几何层：

```text
Sparse Depth Geometry
→ 只保留进入机器人碰撞高度包络的点
→ Obstacle Map
```

状态层：

```text
Explored
Visited
Robot
Obstacle Distance
```

策略：

```text
Global / Local CNN
+
Target Cue
+
MLP
+
Behavior Cloning
```

最终不再采用：

```text
整幅 RGB-D
→ 完整语义点云
→ 所有环境类别 Semantic Map
```
