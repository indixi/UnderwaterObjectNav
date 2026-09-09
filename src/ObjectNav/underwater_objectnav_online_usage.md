# 水下 ObjectNav 在线运行架构与使用说明

## 1. 本次要解决的问题

离线阶段已经完成：冻结的 GFL 检测器把 RGB 图像转换为 `echinus/rock` 检测结果，RGB-D 与世界位姿用于构造 Global/Local Map，BC 网络输出离散动作。在线阶段需要复用完全相同的感知、地图和策略代码，并完成下面的闭环：

```text
启动 Stonefish 与控制器
        ↓
调用服务提交目标 echinus
        ↓
同步取得 RGB + Depth + Odometry
        ↓
冻结 GFL → bbox/score/class_name
        ↓
BBox-Depth 融合 + 世界坐标 Global Map + 机器人中心 Local Map
        ↓
BC → FORWARD / TURN_LEFT / TURN_RIGHT / STOP
        ↓
发布 ActionRequest，等待 ActionStatus
        ↓
动作成功且达到配置的时间间隔后，进入下一轮
        ↓
BC 输出 STOP → 执行 STOP → 保存结果 → 清空地图
```

一次导航结束后，在线节点继续运行。再次调用服务会从机器人当前位置建立一个全新的地图。若需要恢复 Stonefish 中的初始位姿，直接重新启动场景。

## 2. 为什么采用两个进程

ROS Noetic 节点使用系统 Python 3.8：

```text
/usr/bin/python3
Python 3.8.10
```

GFL 与 BC 推理使用 Conda：

```text
/home/yzw/miniforge3/envs/duo-gfl/bin/python
Python 3.10.x
```

不能把依赖 ROS Python 3.8 ABI 的 `rospy/cv_bridge` 直接混入 Python 3.10 推理进程。因此采用 localhost TCP 双进程结构：

```text
ROS 协调节点（系统 Python 3.8）
  - rospy、cv_bridge、message_filters
  - 服务、传感器同步、动作状态机
  - 在线数据录制、调试图发布
                  │
                  │ 127.0.0.1:29500
                  │ 长度前缀 JSON + 原始 NumPy 数组，不使用 pickle
                  ▼
推理 Worker（Conda Python 3.10）
  - MMDetection/GFL
  - Depth 融合与 Global/Local Map
  - PyTorch BC Policy
```

推理 Worker 完全不导入 ROS。启动器会把 `PYTHONPATH` 限制为 `ObjectNav` 源码目录，防止 Noetic 的 Python 3.8 包污染 Conda 3.10 环境。Worker 异常退出时当前任务立即失败，不做自动重启；这是便于验证和定位错误的 Demo 设计。

## 3. Catkin 包与代码结构

文件夹仍为 `src/ObjectNav`，Catkin 包名必须符合 ROS 命名规则，因此注册名为小写 `objectnav`：

```text
src/ObjectNav/
├── package.xml
├── CMakeLists.txt
├── setup.py
├── config/online.yaml
├── launch/objectnav_online.launch
├── msg/NavigationStatus.msg
├── srv/StartNavigation.srv
├── srv/CancelNavigation.srv
├── scripts/
│   ├── online_navigation_node.py
│   └── inference_worker_launcher.py
└── objectnav_bc/online/
    ├── ipc_protocol.py
    ├── inference_client.py
    ├── inference_worker.py
    ├── online_recorder.py
    └── debug_visualizer.py
```

所以启动命令是：

```bash
roslaunch objectnav objectnav_online.launch
```

不是 `roslaunch ObjectNav ...`。

## 4. 权重、配置和类别 ID

### 4.1 建议文件位置

把冻结检测器的部署文件复制到：

```text
src/ObjectNav/objectnav_bc/perception/model/
├── gfl_r50_fpn_underwater_objectnav_inference.py
├── best_echinus_recall_at_precision_95_epoch_5.pth
└── echinus_threshold.json
```

BC 权重默认位置：

```text
src/ObjectNav/work_dirs/objectnav_bc/best_policy.pt
```

实际路径可通过 launch 的 `checkpoint` 和 `bc_config` 参数替换，不必修改 Python 代码。`bc.yaml` 中 GFL 配置、GFL 权重和阈值文件采用相对路径时，是相对于该 `bc.yaml` 所在目录解析的。

### 4.2 不要混淆三种编号

检测模型类别编号保持训练时的定义：

```text
GFL class_id 0 = echinus
GFL class_id 1 = rock
```

不需要、也不能为了 ObjectNav 改成别的 `class_id`。适配器会把检测标签转换为 `class_name`，并在模型元数据存在时检查类别顺序是否为 `("echinus", "rock")`。

导航目标编号是另一套独立编号：

```text
goal_id 0 = echinus
```

当前 `rock` 不是导航终点，不注册成第二个目标；它以置信度写入语义地图，辅助 BC 学习搜索海胆的位置关系。地图通道又是第三套固定顺序：

```text
Global: obstacle, explored, visited, robot, echinus, rock
Local:  obstacle, rock, echinus, visited
```

BC checkpoint 会校验这些通道、目标名称和网络维度。训练与在线配置不一致时 Worker 会直接报错，避免静默使用错误特征。

阈值规则保持：

- `echinus` 使用 `echinus_threshold.json` 的 `recommended_threshold`；
- `rock` 使用 `bc.yaml` 中默认的 `score_threshold: 0.50`，后续可以单独校准；
- 阈值 JSON 内记录的 checkpoint 文件名必须与 GFL checkpoint 文件名一致。

## 5. 3.3 m 的准确含义

`3.3 m` 表示 RGB-D 传感器原点在 `world_ned` 中的 z，不是机器人 base 的 z。NED 中 z 轴向下为正。

在线节点使用里程计姿态和 `bc.yaml` 中：

```yaml
camera:
  nominal_sensor_world_z_m: 3.3
  T_base_camera:
    translation_m: [0.20, 0.0, -0.02]
```

计算实际传感器世界 z，并要求其在 `depth_tolerance_m` 内稳定 `depth_settle_time_s` 后才开始第一次推理。

现有 `action_executor` 控制的是 base z。launch 默认按水平姿态换算：

```text
base_target_z = sensor_target_z - camera_z_offset
              = 3.3 - (-0.02)
              = 3.32 m
```

若修改 `T_base_camera.translation_m[2]`，启动时也应同步传入 `camera_z_offset_m`。地图投影本身始终使用完整的实时 roll/pitch/yaw 和外参，不使用上述水平近似。

## 6. 动作节拍与状态机

`start_navigation` 服务只调用一次，用于开始一个目标任务；它不负责每 2 秒反复调用。实际周期由在线节点内部控制，默认：

```yaml
online:
  inference_interval_s: 2.0
```

节拍规则为：

1. 发布策略动作并收到 `STARTED`，记录动作实际开始时刻；
2. 必须先收到该动作的 `SUCCEEDED`；
3. 若距离开始时刻不足 2 秒，等待满 2 秒；
4. 若动作本身超过 2 秒才成功，成功后立即使用最新同步观测推理；
5. 同一时刻只允许一个推理或一个未结束动作。

在线 launch 只把现有 `action_executor/command_delay_s` 覆盖为 `0.0`，避免两处重复等待；不修改 `action_executor.py`，也不覆盖它原有的 `action_timeout_s: 15.0`。

BC 输出 `STOP` 时仍会发布真实的 `ActionRequest(STOP)`。收到执行器的 `STOPPED` 后才把本次导航标记为 `SUCCEEDED`、完成数据写入并清空 Worker 内的 Global/Local Map。

主要导航状态：

```text
NOT_READY → IDLE → WAITING_FOR_DEPTH → RUNNING
          → INFERENCING → WAITING_FOR_ACTION → RUNNING ...
          → SUCCEEDED / FAILED / CANCELLED
```

默认最多 100 个策略步骤。传感器过期、转换错误、推理错误、非法动作、动作超时或超过最大步数会进入失败流程。如果失败发生在动作执行中，会先等当前动作结束，再发布 `STOP`；收到 `STOPPED` 后才结束并清图，因为现有执行器没有抢占接口。

## 7. 编译前检查

在 Ubuntu 20.04 的 Catkin 工作区中：

```bash
cd ~/work_khd/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
rospack find objectnav
```

最后一条应输出类似：

```text
/home/yzw/work_khd/catkin_ws/src/Aquaflow/src/ObjectNav
```

检查系统 Python ROS 侧依赖：

```bash
/usr/bin/python3 -c "import rospy, cv_bridge, message_filters, cv2, numpy, yaml; print('ROS Python OK')"
```

检查 Conda 推理侧依赖：

```bash
/home/yzw/miniforge3/bin/conda run -n duo-gfl python -c \
"import torch, mmengine, mmdet, cv2, numpy, yaml; print(torch.cuda.is_available())"
```

Conda 环境不需要安装 `rospy` 或 `rospkg`，因为 ROS API 只存在于系统 Python 进程。

可在 `src/ObjectNav` 下运行非 ROS 单元测试：

```bash
cd ~/work_khd/catkin_ws/src/Aquaflow/src/ObjectNav
/home/yzw/miniforge3/bin/conda run -n duo-gfl \
  env PYTHONPATH="$PWD" python -m pytest tests -q
```

## 8. 启动方法

第一次建议显式写出关键路径：

```bash
cd ~/work_khd/catkin_ws
source devel/setup.bash

roslaunch objectnav objectnav_online.launch \
  scene:=rock_seaurchin \
  vehicle_name:=bricsbot \
  vehicle_model:=bricsbot \
  enable:=true \
  checkpoint:=/absolute/path/to/best_policy.pt \
  bc_config:=/absolute/path/to/bc.yaml \
  online_config:=$(rospack find objectnav)/config/online.yaml \
  inference_interval_s:=2.0 \
  record_root:=/home/yzw/work_khd/catkin_ws/underwater_objectnav_online_runs \
  visualize:=true
```

`enable` 默认是 `false`，用于避免仿真一启动就使能推进器；正式验证运动时必须显式传 `enable:=true`。

若文件已经放到默认位置，可简化为：

```bash
roslaunch objectnav objectnav_online.launch enable:=true
```

启动后先观察：

```bash
rostopic echo /underwater_objectnav/navigation_status
```

看到 `state: "IDLE"` 和 `message: "inference worker ready"`，且 Stonefish 已发布新鲜 RGB-D-Pose 后，再提交目标：

```bash
rosservice call /underwater_objectnav/start_navigation \
"goal_category: 'echinus'"
```

服务会立即返回是否接受和 episode ID。它不会等待整次导航完成；最终成功或失败通过状态话题查看：

```bash
rostopic echo /underwater_objectnav/navigation_status
```

手动取消当前任务：

```bash
rosservice call /underwater_objectnav/cancel_navigation "{}"
```

任务终止后节点仍保持运行。再次调用 `start_navigation` 会清空旧地图并从当前位置开始。若要回到场景初始位姿，停止并重新执行 `roslaunch`。

## 9. 在线数据保存

在线数据与专家数据完全分开，默认根目录是：

```text
/home/yzw/work_khd/catkin_ws/underwater_objectnav_online_runs
```

每次运行生成：

```text
episode_0001/
├── episode.yaml
├── trajectory.csv
├── policy.jsonl
├── detections.jsonl
├── bc.yaml
├── online.yaml
├── rgb/000000.png
├── depth/000000.npy
├── global_map/000000.npy
└── local_map/000000.npy
```

保存的是“策略实际进行推理的同步观测”，不是专家动作开始附近的观测：

- RGB、原始 float32 Depth、base 世界位姿；
- GFL NMS 并按部署阈值过滤后的全部 bbox、score、class name；
- 四个动作概率、最终动作、Target Cue、推理耗时；
- 当步完整 Global Map 与 Local Map；
- 观测、推理开始/结束、动作请求/开始/结束时刻及执行状态；
- 场景、目标、成功/失败原因、权重路径和配置快照。

地图以 `float16 .npy` 保存以减少磁盘写入量，RGB/Depth/地图写入在线程中异步执行。磁盘写入或最终文件保存失败只记录警告，不改变导航算法的成功/失败判定。

这些目录的 `source: policy`，且默认根目录不在 `underwater_objectnav_dataset` 内，因此不会混入专家 BC 训练。不要手工把它们复制进专家数据目录，除非后续明确设计了数据筛选和再训练流程。

## 10. 非阻塞可视化

`visualize:=true` 时只发布调试话题，不自动打开窗口：

```text
/underwater_objectnav/debug/detections_image
/underwater_objectnav/debug/global_map
/underwater_objectnav/debug/local_map
/underwater_objectnav/debug/map_comparison
/underwater_objectnav/debug/action_probabilities
/underwater_objectnav/debug/inference_time
```

手工查看地图对比：

```bash
rqt_image_view /underwater_objectnav/debug/map_comparison
```

或分别查看检测图与两张地图。可视化在独立线程中运行，队列长度为 1；来不及处理时丢弃旧帧，不阻塞推理和动作循环。需要排除可视化开销时传：

```bash
visualize:=false
```

Global Map 图中同时画出机器人位置与世界朝向；Local Map 中机器人位于中心且前方固定朝图像上方，因此它保留的是机器人坐标系下的局部空间布局。

## 11. 常见问题排查

### Catkin 报找不到 `cola2_msgs`

当前 ObjectNav、`robot_control` 和 `aquaflow_stonefish` 的实际源码都没有使用 `cola2_msgs` 消息。`aquaflow_stonefish` 原先由 Catkin 模板遗留了这项强制依赖，本次已从其 `CMakeLists.txt` 和 `package.xml` 删除，因此不需要为了本 Demo 安装 COLA2。更新代码后强制重新执行 CMake 并构建：

```bash
cd ~/work_khd/catkin_ws
catkin build aquaflow_stonefish objectnav --force-cmake
source devel/setup.bash
```

### 服务返回 `inference_worker_not_ready`

检查 GFL/BC 权重路径、Conda 可执行文件与环境名，并查看 launch 终端是否出现：

```text
ObjectNav inference worker READY on 127.0.0.1:29500
ObjectNav inference worker connected and READY
```

### 服务返回没有新鲜 RGB-D-Pose

检查：

```bash
rostopic hz /bricsbot/rgb/image_color
rostopic hz /bricsbot/depth/image_depth
rostopic hz /bricsbot/odometry
```

三者使用 ApproximateTimeSynchronizer，默认 `queue=20`、`slop=0.08 s`，推理时观测最大年龄为 `0.25 s`。

### Worker 报图像尺寸不一致

在线 RGB 与 Depth 必须与 `bc.yaml` 的 `camera.width/height` 一致，默认是 `640×480`，并且二者已经像素对齐。

### 机器人收到动作但不运动

确认启动时使用了 `enable:=true`，并查看：

```bash
rostopic echo /underwater_objectnav/action_request
rostopic echo /underwater_objectnav/action_status
```

### 一启动就提示高度未到达

3.3 m 检查的是传感器原点。确认 `target_z`、`camera_z_offset_m` 和 `bc.yaml` 的相机外参一致；默认 base 目标约为 3.32 m。

### 第一次推理超时

默认 `inference_timeout_s: 1.8`。先检查 GPU 是否可用和模型是否确实加载到 `cuda:0`。若仅第一次 CUDA 初始化超过限制，可在 `config/online.yaml` 暂时增大超时来定位，但 2 秒闭环目标仍要求稳定推理耗时小于动作周期。

## 12. 哪些模块保持解耦

- `objectnav_bc/perception`、`mapping`、`models` 同时供离线和在线使用，不包含 ROS；
- 在线 ROS 节点只消费 Worker 返回的动作、概率、检测和地图，不了解 GFL/BC 内部网络结构；
- `action_executor.py` 不做修改，在线层只通过已有 `ActionRequest/ActionStatus` 接口交互；
- 将来修改 ObjectNav 策略结构时，只要保持 Worker 响应和动作接口，ROS 调度、记录和可视化无需随网络一起重写；
- 将来单独校准 rock 阈值，只需修改 `bc.yaml` 或其阈值配置并重新启动 Worker，不需要更改导航类别 ID。
