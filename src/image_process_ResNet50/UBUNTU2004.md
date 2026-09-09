# Ubuntu 20.04 + RTX 4060 运行说明

本工程固定使用 Python 3.10、PyTorch 2.1.0（CUDA 12.1）、torchvision 0.16.0、MMCV 2.1.0、MMEngine 0.10.7 和 MMDetection 3.3.0。PyTorch wheel 自带 CUDA 12.1 runtime；宿主机仍需安装支持 CUDA 12.1 且能识别 RTX 4060 的 NVIDIA 驱动。

## 1. 复制工程

将整个 `image_process_ResNet50` 文件夹复制到 Ubuntu。数据已经位于工程内部，不需要另外修改绝对路径。推荐使用纯 ASCII 路径：

```text
~/projects/image_process_ResNet50/
└── data/underwater_objectnav_rgb/
```

配置已经自包含，不再依赖项目目录外的 `mmdetection/configs` 源码树。

## 2. 创建环境

```bash
nvidia-smi
conda create -n objectnav-gfl python=3.10 -y
conda activate objectnav-gfl
cd ~/projects/image_process_ResNet50
bash scripts/install_ubuntu2004.sh
```

安装脚本使用 PyTorch CUDA 12.1 wheel 和匹配的 MMCV 预编译 wheel，不需要本地编译 CUDA 扩展。

## 3. 检查数据

```bash
python tools/inspect_dataset.py \
  --data-root data/underwater_objectnav_rgb \
  --samples 20
```

确认 `outputs/underwater_objectnav_dataset_audit/dataset_report.md` 中缺失、损坏、尺寸不符及标注问题均为 0，并人工查看 `annotation_preview/`。

## 4. 训练

```bash
bash scripts/train.sh
```

等价的完整命令：

```bash
python tools/train.py \
  --config configs/gfl_r50_fpn_underwater_objectnav.py \
  --data-root data/underwater_objectnav_rgb \
  --work-dir work_dirs/gfl_r50_fpn_underwater_objectnav \
  --gpu 0 --seed 42 --amp --resume auto \
  --min-echinus-precision 0.95
```

RTX 4060 8 GB 若出现 CUDA OOM：

```bash
python tools/train.py \
  --batch-size 1 --accumulation 8 --workers 2 \
  --amp --resume auto
```

首次训练会从 ImageNet 预训练的 ResNet-50 初始化；新数据只有两类，因此不要用旧四分类 checkpoint 执行 `--resume`。`--resume auto` 只用于当前新工作目录中的断点续训。

## 5. 校准并推理

训练后在验证集上选择满足 precision ≥ 95% 的海胆阈值：

```bash
python tools/calibrate_threshold.py \
  --config configs/gfl_r50_fpn_underwater_objectnav.py \
  --checkpoint work_dirs/gfl_r50_fpn_underwater_objectnav/BEST.pth \
  --min-precision 0.95
```

使用该阈值推理：

```bash
python tools/infer.py \
  --config configs/gfl_r50_fpn_underwater_objectnav.py \
  --checkpoint work_dirs/gfl_r50_fpn_underwater_objectnav/BEST.pth \
  --input /path/to/images \
  --threshold-file outputs/echinus_threshold.json
```

最后再运行一次 test 评估：

```bash
python tools/evaluate.py  --config configs/gfl_r50_fpn_underwater_objectnav.py   --checkpoint work_dirs/gfl_r50_fpn_underwater_objectnav/BEST.pth   --split test   --threshold-file outputs/echinus_threshold.json
```

这里必须让评估使用 val 阶段已经固定的阈值。最终关注
`echinus/precision_at_operating_threshold`、`recall_at_operating_threshold`
以及 `false_positive_at_operating_threshold`，不要再根据 test 选择新阈值。

## 常见问题

- `nvidia-smi` 不可用：先修复宿主机 NVIDIA 驱动。
- `No matching distribution found for mmcv`：确认系统为 Linux x86_64、Python 3.10、PyTorch 2.1.0 + cu121。
- `libGL.so.1` 缺失：改装 `opencv-python-headless`，或安装系统的 `libgl1`。
- CUDA OOM：使用 batch size 1、梯度累积 8，再降低输入分辨率。
- 不要同时安装 `mmcv` 和 `mmcv-lite`；训练需要带 CUDA 算子的 `mmcv`。
