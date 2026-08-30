# 水下 ObjectNav：数据处理与 Behavior Cloning 训练方案

## 1. 目标

将采集得到的专家示范数据：

```text
RGB
Depth
Robot Pose
Goal Category
Expert Action
```

处理为策略网络输入，并使用 Behavior Cloning（BC）训练离散动作策略：

```text
FORWARD
TURN_LEFT
TURN_RIGHT
STOP
```

整体流程：

```text
RGB + Depth + Pose
        ↓
ResNet50 + FPN
        ↓
语义检测 / 分割
        ↓
Semantic Map
        ↓
CNN Map Encoder
        ↓
Map Feature
        +
Goal Embedding
        +
Orientation Feature
        ↓
MLP Policy
        ↓
FORWARD / LEFT / RIGHT / STOP
```

---

## 2. 原始数据格式

每个 step 至少包含：

```text
rgb_path
depth_path
x
y
z
roll
pitch
yaw
goal_category
expert_action
episode_id
step_id
timestamp
```

动作编码：

```text
0 = FORWARD
1 = TURN_LEFT
2 = TURN_RIGHT
3 = STOP
```

---

## 3. RGB 语义处理

RGB 输入视觉网络：

```text
RGB
 ↓
ResNet50
 ↓
FPN
 ↓
Detection / Segmentation Head
```

第一版建议语义类别：

```text
Sea Urchin
Rock
Sand
```

后续可增加：

```text
Algae
Reef
Other Habitat Classes
```

视觉网络输出至少包含：

```text
semantic class
confidence
mask / bbox
```

第一阶段建议冻结 ResNet50 + FPN，不参与 BC 反向传播。

---

## 4. Depth + Pose 几何投影

对于像素 \((u,v)\)，深度为 \(d\)，根据相机内参：

\[
X_c=\frac{(u-c_x)d}{f_x}
\]

\[
Y_c=\frac{(v-c_y)d}{f_y}
\]

\[
Z_c=d
\]

得到：

\[
P_c=[X_c,Y_c,Z_c,1]^T
\]

利用相机外参 \(T_{base}^{camera}\) 和机器人世界位姿 \(T_{world}^{base}\)：

\[
P_w
=
T_{world}^{base}
T_{base}^{camera}
P_c
\]

---

## 5. Semantic Map

第一版建立二维多通道 Semantic Map：

\[
M_t\in R^{C\times H\times W}
\]

建议通道：

```text
Channel 0 : Obstacle
Channel 1 : Explored
Channel 2 : Current Robot Location
Channel 3 : Visited
Channel 4 : Sea Urchin
Channel 5 : Rock
Channel 6 : Sand
```

即：

```text
C = 7
```

建议地图参数：

```yaml
map_width_m: 10.0
map_height_m: 10.0
map_resolution_m: 0.05
```

---

## 6. 地图更新

每个 step：

```text
RGB_t
 ↓
语义检测 / 分割

Depth_t + Pose_t
 ↓
语义点投影到世界坐标

        ↓

更新 Semantic Map_t
```

地图需要累计：

```text
Obstacle
Explored
Visited
Semantic Classes
```

不能只使用当前帧。

---

## 7. Policy 输入

### 7.1 Map Feature

Semantic Map 先经过 CNN：

```text
7 × H × W
   ↓
Conv
   ↓
Conv
   ↓
Conv
   ↓
Global Pool / Flatten
   ↓
256-D Map Feature
```

第一版可采用：

```text
Conv(C, 32, 3, stride=2, padding=1)
ReLU

Conv(32, 64, 3, stride=2, padding=1)
ReLU

Conv(64, 128, 3, stride=2, padding=1)
ReLU

AdaptiveAvgPool2d(1)

Flatten

Linear(128, 256)
ReLU
```

得到：

\[
f_t^M\in R^{256}
\]

### 7.2 Goal Feature

```text
Goal Category
 ↓
Goal ID
 ↓
Embedding
 ↓
8-D
```

得到：

\[
f_t^G\in R^8
\]

### 7.3 Orientation Feature

yaw 转换为：

\[
f_t^P=
[
\sin(\theta_t),
\cos(\theta_t)
]
\]

因此：

\[
f_t^P\in R^2
\]

绝对位置 \(x,y,z\) 主要用于建图，不建议直接作为 Policy 主输入。

---

## 8. 最终状态向量

拼接：

\[
s_t=
[
f_t^M;
f_t^G;
f_t^P
]
\]

维度：

\[
256+8+2=266
\]

即：

```text
Policy Input = 266-D
```

---

## 9. MLP Policy

第一版建议：

```text
266
 ↓
Linear(266, 256)
 ↓
ReLU
 ↓
Linear(256, 128)
 ↓
ReLU
 ↓
Linear(128, 4)
 ↓
Logits
```

输出对应：

```text
logit[0] = FORWARD
logit[1] = TURN_LEFT
logit[2] = TURN_RIGHT
logit[3] = STOP
```

Softmax：

\[
\pi_\theta(a|s_t)
=
Softmax(z_t)
\]

推理时：

\[
a_t
=
\arg\max_a
\pi_\theta(a|s_t)
\]

---

## 10. Behavior Cloning Loss

专家标签：

\[
a_t^E
\]

BC 使用 Cross Entropy：

\[
L_{BC}
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log
\pi_\theta(a_t^E|s_t)
\]

PyTorch：

```python
torch.nn.CrossEntropyLoss
```

输入：

```text
logits: [B, 4]
labels: [B]
```

---

## 11. 类别不平衡

训练前统计：

```text
num_forward
num_left
num_right
num_stop
```

建议使用 Weighted Cross Entropy：

\[
L
=
-w_{a_t}
\log
\pi_\theta(a_t^E|s_t)
\]

类别权重可取：

\[
w_c
=
\frac{N}{K N_c}
\]

其中：

```text
N   = 总样本数
K   = 4
N_c = 第 c 类样本数
```

---

## 12. Dataset 划分

必须按照 Episode 划分：

```text
Train      70%
Validation 15%
Test       15%
```

禁止把同一 Episode 的连续帧随机拆到不同集合。

---

## 13. Dataset Loader 输出

每条训练样本：

```python
{
    "semantic_map": Tensor[C, H, W],
    "goal_id": int,
    "yaw": float,
    "action": int,
    "episode_id": int,
    "step_id": int,
}
```

Batch：

```text
semantic_map : [B, C, H, W]
goal_id      : [B]
yaw          : [B]
action       : [B]
```

---

## 14. 预处理缓存

建议先离线生成 Semantic Map，再训练 Policy：

```text
原始 Dataset
    ↓
RGB + Depth + Pose
    ↓
Semantic Mapping
    ↓
保存每个 step 的 Semantic Map
    ↓
Policy Dataset Loader
```

例如：

```text
episode_0001/
├── semantic_map/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
```

---

## 15. BC 训练流程

```text
1. 加载 Batch
2. Semantic Map → CNN Encoder
3. Goal ID → Embedding
4. yaw → sin/cos
5. 拼接得到 266-D Feature
6. MLP 输出 4-D logits
7. 与 Expert Action 计算 Cross Entropy
8. backward()
9. optimizer.step()
10. Validation
```

伪代码：

```python
for batch in train_loader:
    semantic_map = batch["semantic_map"]
    goal_id = batch["goal_id"]
    yaw = batch["yaw"]
    action = batch["action"]

    map_feat = map_encoder(semantic_map)
    goal_feat = goal_embedding(goal_id)

    orientation_feat = torch.stack(
        [torch.sin(yaw), torch.cos(yaw)],
        dim=-1
    )

    state = torch.cat(
        [map_feat, goal_feat, orientation_feat],
        dim=-1
    )

    logits = policy_mlp(state)
    loss = criterion(logits, action)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 16. 第一版训练配置

```yaml
batch_size: 32
learning_rate: 0.0003
optimizer: Adam
epochs: 30
weight_decay: 0.00001
```

按照 Validation Loss 保存：

```text
best_policy.pt
```

---

## 17. 离线评价

至少统计：

```text
Overall Accuracy
Per-Class Accuracy
Confusion Matrix
STOP Precision
STOP Recall
Validation Loss
```

禁止只看 Overall Accuracy。

---

## 18. 闭环评价

训练完成后重新放回仿真环境：

```text
RGB-D
 ↓
Semantic Map
 ↓
Policy
 ↓
离散动作
 ↓
Robot
 ↓
下一 Observation
```

重点评价：

```text
Success Rate
平均导航步数
平均路径长度
碰撞次数
重复探索次数
STOP 成功率
```

---

## 19. 第一版代码模块

```text
objectnav_bc/
│
├── perception/
│   ├── semantic_detector.py
│   └── depth_projection.py
│
├── mapping/
│   └── semantic_mapper.py
│
├── dataset/
│   ├── preprocess_dataset.py
│   └── bc_dataset.py
│
├── models/
│   ├── map_encoder.py
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
    └── bc.yaml
```

---

## 20. 第一阶段训练范围

第一阶段训练：

```text
CNN Map Encoder
+
Goal Embedding
+
MLP Policy
```

冻结：

```text
ResNet50 + FPN
```

后续再考虑：

```text
Fine-tune perception
DAgger
GNN / Attention
ACT / Diffusion / Flow Matching
```

第一版优先完成：

```text
Semantic Map
→ CNN
→ MLP
→ BC
→ Closed-loop ObjectNav
```
