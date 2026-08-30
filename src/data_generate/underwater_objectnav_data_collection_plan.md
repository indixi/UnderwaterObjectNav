# 水下 ObjectNav 模仿学习数据采集方案

## 1. 任务目标

采集用于 Behavior Cloning（BC）训练的专家示范数据，使策略网络学习：

```text
Observation
    ↓
Policy
    ↓
Forward / Turn Left / Turn Right / Stop
```

目标任务：机器人通过第一视角搜索并靠近海胆。

---

## 2. 离散动作定义

动作空间：

```text
FORWARD
TURN_LEFT
TURN_RIGHT
STOP
```

建议第一版动作原语：

- `FORWARD`：沿当前航向前进固定距离，例如 0.20 m
- `TURN_LEFT`：原地左转固定角度，例如 20°
- `TURN_RIGHT`：原地右转固定角度，例如 20°
- `STOP`：结束当前导航任务

动作由底层闭环控制器执行，不采用“推进器运行固定时间”的方式。

每个动作需要设置：

- 目标位移或目标角度
- 完成误差阈值
- 最大执行时间 timeout

---

## 3. 每个决策步保存的数据

每个 step 保存：

```text
RGB
Depth
Robot Pose
Goal Category
Expert Action
Timestamp
Episode ID
Step ID
```

Robot Pose 建议保存：

```text
x
y
z
roll
pitch
yaw
```

Episode 结束时额外保存：

```text
success
failure_reason
```

仿真中建议额外保存但禁止作为 Policy 输入：

```text
target_gt_position
scene_id
```

---

## 4. 数据采集时序

严格按照以下顺序：

```text
1. 获取当前 RGB_t
2. 获取当前 Depth_t
3. 获取当前 Pose_t
4. 显示第一视角
5. 人类专家选择 Expert Action_t
6. 保存当前 Observation_t + Action_t
7. 机器人完整执行一个离散动作原语
8. 动作结束后获取下一时刻 Observation_(t+1)
9. 重复
```

即：

\[
o_t \rightarrow a_t \rightarrow o_{t+1}
\]

必须保证动作标签与“动作执行前”的观测对应。

---

## 5. Episode 采集方式

每个 Episode 随机初始化：

```text
机器人初始位置
机器人初始 yaw
海胆位置
岩石位置
沙地区域
其他环境布局
```

专家只能根据机器人第一视角进行控制，不查看海胆真实位置。

完整 Episode 应包含：

```text
搜索
↓
选择潜在区域
↓
接近候选区域
↓
发现海胆
↓
调整方向
↓
靠近海胆
↓
STOP
```

---

## 6. 必须覆盖的场景

数据集中至少包含：

```text
1. 海胆初始可见
2. 海胆初始不可见，岩石可见
3. 海胆初始不可见，同时存在岩石和沙地
4. 多块岩石，只有部分岩石附近有海胆
5. 搜索某块岩石后未发现海胆，需要离开继续探索
6. 海胆被岩石部分遮挡
7. 不同机器人初始位置
8. 不同机器人初始朝向
9. 不同海胆位置
10. 不同岩石布局
```

重点保证存在大量：

```text
海胆不可见
+
岩石 / 沙地等环境语义可见
+
专家依据环境信息选择搜索方向
```

的示范数据。

---

## 7. 数据目录结构

建议：

```text
underwater_objectnav_dataset/
│
├── metadata.yaml
│
├── episode_0001/
│   ├── rgb/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   │
│   ├── depth/
│   │   ├── 000000.npy
│   │   ├── 000001.npy
│   │   └── ...
│   │
│   ├── trajectory.csv
│   └── episode.yaml
│
├── episode_0002/
│   └── ...
│
└── ...
```

---

## 8. metadata.yaml

建议记录：

```yaml
dataset_name: underwater_objectnav_bc

goal_categories:
  - sea_urchin

actions:
  0: FORWARD
  1: TURN_LEFT
  2: TURN_RIGHT
  3: STOP

forward_distance_m: 0.20
turn_angle_deg: 20.0

camera:
  rgb_topic: ""
  depth_topic: ""
  intrinsic_file: camera_intrinsic.yaml
  extrinsic_file: camera_extrinsic.yaml
```

---

## 9. trajectory.csv

每个 Episode 的 `trajectory.csv` 至少包含：

```text
step_id
timestamp
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
```

示例：

```csv
step_id,timestamp,rgb_path,depth_path,x,y,z,roll,pitch,yaw,goal_category,expert_action
0,0.00,rgb/000000.png,depth/000000.npy,1.20,2.10,0.50,0,0,0.35,sea_urchin,TURN_RIGHT
1,0.42,rgb/000001.png,depth/000001.npy,1.20,2.10,0.50,0,0,0.70,sea_urchin,FORWARD
```

---

## 10. episode.yaml

建议保存：

```yaml
episode_id: 1
scene_id: scene_001
goal_category: sea_urchin

success: true
failure_reason: null

start_pose:
  x: 1.2
  y: 2.1
  z: 0.5
  yaw: 0.35

target_gt_position:
  x: 4.8
  y: 3.2
  z: 0.5
```

`target_gt_position` 仅用于评价、调试和离线分析，不允许作为策略输入。

---

## 11. 数据采集程序功能要求

Agent 需要实现一个 ROS 数据采集节点，至少具备：

```text
1. 订阅 RGB
2. 订阅 Depth
3. 订阅 Robot Pose
4. 接收离散专家动作
5. 按时间戳同步数据
6. 动作执行前保存 Observation + Expert Action
7. 自动创建 Episode 目录
8. 自动生成 trajectory.csv
9. Episode 结束时保存 success / failure
10. 支持开始记录、停止记录、取消当前 Episode
```

建议提供以下控制接口：

```text
START_EPISODE
END_EPISODE_SUCCESS
END_EPISODE_FAILURE
CANCEL_EPISODE
```

---

## 12. 数据质量要求

采集时必须保证：

```text
RGB、Depth、Pose、Action 时间对齐
动作标签对应动作执行前的观测
每个 Episode 保持 step 连续
不能只采集成功靠近海胆后的过程
必须包含目标不可见阶段
必须包含无目标岩石等负样本场景
```

---

## 13. 第一阶段数据量

第一版建议先采集：

```text
100～300 个完整 Episode
```

之后根据训练和闭环测试结果继续扩充。

不要仅依据单帧分类准确率判断数据是否足够，应重点观察：

```text
Success Rate
平均导航步数
平均路径长度
Stop 正确率
重复探索情况
失败类型
```

---

## 14. 后续训练接口

采集完成后，可离线处理：

```text
RGB
↓
ResNet50 + FPN
↓
Visual / Semantic Feature
```

再与：

```text
Depth Feature
Robot Pose
Goal Category
```

组合后输入 MLP：

```text
Feature Vector
    ↓
MLP
    ↓
FORWARD / TURN_LEFT / TURN_RIGHT / STOP
```

第一版采用 Behavior Cloning：

\[
L_{BC}
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log \pi_\theta(a_t^E|s_t)
\]

其中：

- \(s_t\)：当前状态特征
- \(a_t^E\)：专家动作
- \(\pi_\theta\)：待训练策略网络
