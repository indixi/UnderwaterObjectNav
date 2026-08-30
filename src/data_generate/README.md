# 水下 ObjectNav 数据采集包

本目录实现方案文档要求的 ROS1 数据采集节点，默认适配项目中的 `bricsbot` Stonefish 场景。

## 启动

```bash
roslaunch data_generate data_collector.launch
```

开始记录时只需要场景编号和目标类别，不需要知道海胆的真实位置：

```bash
rosservice call /objectnav_data_collector/start_episode \
  "scene_id: 'scene_001'
goal_category: 'sea_urchin'"
```

专家输入直接复用 `robot_control` 中已有的键盘发布器，建议在独立终端启动，以确保终端 stdin 可用。键盘按键映射如下：

```text
W：FORWARD       A：TURN_LEFT
D：TURN_RIGHT    SPACE：STOP
Q：退出键盘发布器
```

每个按键默认延迟 5 秒执行。延迟时间在 `src/robot_control/config/keyboard_reference.yaml` 中设置：

```yaml
keyboard_reference_publisher:
  command_delay_s: 5.0
```

单独启动 robot_control 键盘发布器，默认延迟 5 秒：

```bash
roslaunch robot_control keyboard_reference.launch
```

也可以直接通过 launch 参数修改延迟，例如延迟 8 秒：

```bash
roslaunch robot_control keyboard_reference.launch command_delay_s:=8.0
```

该发布器同时发布原有的 `PoseStamped` 控制目标和
`/underwater_objectnav/expert_action` 离散动作；采集器订阅这两个话题，
因此 CSV 标签与实际控制目标保持一致。键盘动作可以连续输入，采集器会按到达顺序排队执行：

```bash
rostopic pub -1 /underwater_objectnav/expert_action std_msgs/String "data: 'FORWARD'"
```

Episode 结束：

```bash
rosservice call /objectnav_data_collector/end_episode "success: true failure_reason: ''"
rosservice call /objectnav_data_collector/end_episode "success: false failure_reason: '目标未找到'"
rosservice call /objectnav_data_collector/cancel_episode
```

数据输出为 `underwater_objectnav_dataset/episode_NNNN/`，包含 RGB PNG、Depth NPY、连续的 `trajectory.csv` 和 `episode.yaml`。

注意：Depth 通过 `cv_bridge` 以 `passthrough` 读取并统一保存为 `float32`；目标真值只出现在 `episode.yaml`，不会写入 trajectory 的策略观测字段。

## 代码结构与工作原理

本功能由两个 ROS 节点配合完成。

### robot_control 键盘专家节点

文件：`src/robot_control/scripts/keyboard_reference_publisher.py`

该节点是项目统一的键盘输入源，读取键盘后完成两件事：

1. 根据最新的 Odometry 计算目标 PoseStamped；
2. 发布与该目标对应的离散专家动作。

它发布的两个话题分别是：

```text
/aquaflow/nominal_pose
/underwater_objectnav/expert_action
```

键盘映射为：

```text
W       FORWARD
A       TURN_LEFT
D       TURN_RIGHT
SPACE   STOP
Q       退出键盘节点
```

动作目标的距离和角度由 `robot_control/config/keyboard_reference.yaml` 控制。
默认值为前进 0.5 m、转向 30 度。

### data_generate 采集节点

文件：`src/data_generate/scripts/objectnav_data_collector.py`

该节点不读取键盘，也不重新计算动作目标，而是订阅键盘节点产生的动作和
目标位姿。它使用近似时间同步器同步 RGB、Depth 和 Odometry。

每次动作的处理顺序为：

```text
同步 RGB、Depth、Pose
        ↓
接收键盘动作
        ↓
保存动作执行前的观测和动作标签
        ↓
使用 robot_control 的目标位姿进行闭环执行判断
        ↓
达到误差阈值后允许下一动作
```

因此，trajectory.csv 中的标签与实际 PID 控制目标保持一致。

## ROS 服务接口

### 开始 Episode

```bash
rosservice call /objectnav_data_collector/start_episode \
"scene_id: 'scene_001'
goal_category: 'sea_urchin'"
```

采集系统不会请求、保存或使用海胆真实位置；专家只能依据第一视角观测进行搜索。

### 成功或失败结束 Episode

```bash
rosservice call /objectnav_data_collector/end_episode \
"success: true
failure_reason: ''"
```

```bash
rosservice call /objectnav_data_collector/end_episode \
"success: false
failure_reason: '目标未找到'"
```

### 取消 Episode

```bash
rosservice call /objectnav_data_collector/cancel_episode
```

取消后当前 Episode 目录会被删除。

## 完整使用流程

### 1. 编译工作空间

```bash
cd ~/Aquaflow
catkin_make
source devel/setup.bash
```

### 2. 启动仿真和底层控制器

```bash
roslaunch aquaflow_stonefish tracking.launch \
vehicle_name:=bricsbot vehicle_model:=bricsbot scene:=random_pillars
```

### 3. 启动数据采集器

```bash
roslaunch data_generate data_collector.launch
```

可以通过 dataset_root 指定数据保存目录：

```bash
roslaunch data_generate data_collector.launch \
dataset_root:=/tmp/underwater_objectnav_dataset
```

### 4. 启动 robot_control 键盘发布器

必须在独立的交互式终端中启动，并确保该终端拥有焦点：

```bash
roslaunch robot_control keyboard_reference.launch
```

不建议在 data_generate 中重复启动键盘节点，否则可能产生重复控制目标。

### 5. 开始和结束采集

先调用 start_episode，再使用 W、A、D、SPACE 控制机器人。采集结束后调用
end_episode 标记成功或失败。

如果前一个动作还未完成、观测过期或三路数据没有同步，当前动作不会被记录。

## 参数配置

键盘参数位于：

```text
src/robot_control/config/keyboard_reference.yaml
```

```yaml
keyboard_reference_publisher:
  forward_distance_m: 0.5
  turn_angle_deg: 30.0
  command_delay_s: 5.0
  key_repeat_guard_s: 0.10
```

command_delay_s 是固定执行延迟。例如设置为 5.0 后，按键发生后的第 5 秒
才会发布对应的目标位姿和动作标签；设置为 8.0 则第 8 秒发布。延迟期间的多个按键会进入队列，不会被
因为按键过快而丢弃。

采集器参数位于：

```text
src/data_generate/config/collector.yaml
```

重要参数包括：

```yaml
sync_slop_s: 0.08
max_observation_age_s: 0.25
forward_position_tolerance_m: 0.035
turn_angle_tolerance_deg: 3.0
action_timeout_s: 15.0
auto_end_on_stop: false
```

其中 action_timeout_s 是单个动作的最大执行时间；超时后 Episode 会以失败
结束。auto_end_on_stop 设为 true 时，SPACE 会自动成功结束 Episode。

## 输出数据

默认目录结构如下：

```text
underwater_objectnav_dataset/
├── metadata.yaml
└── episode_0001/
    ├── rgb/000000.png
    ├── depth/000000.npy
    ├── trajectory.csv
    └── episode.yaml
```

trajectory.csv 的字段为：

```text
step_id,timestamp,rgb_path,depth_path,x,y,z,roll,pitch,yaw,goal_category,expert_action
```

每行对应一个动作执行前的观测。Depth 保存为 float32 NumPy 数组。

episode.yaml 保存 Episode 编号、场景、目标类别、起始位姿、成功状态、失败
原因和可选的目标真实位置。目标真实位置不能输入策略网络。

## 常见问题

### 键盘没有反应

确认 keyboard_reference.launch 在独立终端运行，并且终端有焦点。该节点要求
stdin 是交互式终端。

### 动作没有记录

检查 Episode 是否已经通过 start_episode 开始，并确认以下话题存在：

```bash
rostopic list | grep bricsbot
rostopic echo /underwater_objectnav/collector_status
```

### 动作距离没有变化

实际动作距离和角度由 robot_control 配置控制。修改配置后需要重启键盘发布器，
采集器自身的动作参数不会覆盖 robot_control 的参数。
