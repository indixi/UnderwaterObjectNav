# Underwater ObjectNav

本目录现在包含两部分相互解耦的流程：

- 离线阶段：冻结 `echinus/rock` GFL，融合 RGB-D 与世界位姿构建地图，再用专家动作训练和评估 BC 策略；
- 在线阶段：结合 Stonefish 与 `robot_control`，周期性获取同步 RGB-D-Pose，完成 GFL、建图和 BC 推理，发布动作并保存实际运行数据。

核心约定：

- 检测类别为 `0=echinus, 1=rock`，不修改 GFL `class_id`；
- 当前唯一导航目标为 `echinus`，导航 `goal_id=0`；
- `rock` 不是终点，但以置信度写入语义地图辅助搜索；
- 每个 episode 独立建图，结束后清空地图；
- 在线策略数据与专家数据使用不同目录，不自动留作专家标签；
- `episode_false*` 始终不参与离线训练；
- 地图、相机、深度、网络和训练参数集中在 `objectnav_bc/config/bc.yaml`。

文档入口：

- [离线训练与评估](objectnav_bc/README.md)
- [需求分析、坐标和网络输入说明](underwater_objectnav_gfl_depth_bc_reference.md)
- [在线 ROS 架构、启动、服务、可视化与记录](underwater_objectnav_online_usage.md)

在线 Catkin 包名为小写 `objectnav`，目录仍为 `ObjectNav`：

```bash
roslaunch objectnav objectnav_online.launch enable:=true
```
