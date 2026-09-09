# Underwater ObjectNav

本目录当前实现第一阶段离线 ObjectNav：冻结 `echinus/rock` GFL 图像检测器，融合对齐深度和机器人世界位姿，生成 Global/Local 地图，再用专家动作训练 Behavior Cloning 策略。

当前代码不包含 ROS 在线节点。完整配置、数据格式、权重放置位置和执行命令见 [`objectnav_bc/README.md`](objectnav_bc/README.md)。

本次需求分析、设计取舍、坐标计算、网络输入和常见问题的集中说明见 [`underwater_objectnav_gfl_depth_bc_reference.md`](underwater_objectnav_gfl_depth_bc_reference.md)。

核心约定：

- 检测类别：`0=echinus, 1=rock`；
- 导航目标：`echinus`，目标 ID 为 0；
- `rock` 不是终点，但作为带置信度的地图语义参与策略学习；
- 每个 episode 独立建图，episode 结束后清零；
- 只使用成功 episode，`episode_false*` 始终不参与训练；
- 所有地图、相机、深度和训练参数集中在 `objectnav_bc/config/bc.yaml`。

离线执行顺序：

```text
cache_detections.py
  -> preprocess_dataset.py
  -> train_bc.py
  -> eval_offline.py
```
