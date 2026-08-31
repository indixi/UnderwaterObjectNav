# Ubuntu 20.04 + RTX 4060 Ti 部署说明

本方案固定使用 Python 3.10、PyTorch 2.1.0（CUDA 12.1）、MMCV 2.1.0、MMEngine 0.10.7 和 MMDetection 3.3.0。CUDA 运行库由 PyTorch wheel 携带，不需要单独安装完整 CUDA Toolkit；宿主机仍需安装能够支持 CUDA 12.1 的 NVIDIA 驱动。

## 1. 传输目录

把整个 `duo_gfl_project` 目录和 DUO 数据传到 Ubuntu。`third_party/mmcv` 不参与 Linux 安装，可以不传，以显著减小传输体积。推荐仅使用 ASCII 路径，例如：

```text
~/projects/duo_gfl_project
~/projects/duo_gfl_project/data/DUO
```

数据目录必须包含：

```text
DUO/
├── annotations/instances_train.json
├── annotations/instances_test.json
├── images/train/
└── images/test/
```

## 2. 创建环境

先确认驱动能够识别显卡：

```bash
nvidia-smi
```

使用 Conda：

```bash
conda create -n duo-gfl python=3.10 -y
conda activate duo-gfl
cd ~/projects/duo_gfl_project
bash scripts/install_ubuntu2004.sh
```

安装脚本强制下载 MMCV 的预编译 wheel；如果不存在匹配包会直接失败，不会悄悄开始源码编译。

## 3. 检查数据并训练

```bash
python tools/inspect_duo_dataset.py --data-root data/DUO --samples 20

bash scripts/train.sh
```

也可以直接指定参数：

```bash
python tools/train.py \
  --config configs/gfl_r50_fpn_duo_base.py \
  --data-root data/DUO \
  --work-dir work_dirs/gfl_r50_fpn_duo \
  --gpu 0 --seed 42 --amp --resume auto
```

4060 Ti 常见为 8 GB 或 16 GB。默认配置为 batch size 2、梯度累积 4；若 8 GB 型号出现显存不足，使用：

```bash
python tools/train.py \
  --config configs/gfl_r50_fpn_duo_base.py \
  --data-root data/DUO \
  --work-dir work_dirs/gfl_r50_fpn_duo \
  --batch-size 1 --accumulation 8 --workers 2 --amp --resume auto
```

## 4. 常见问题

- `nvidia-smi` 不存在或看不到显卡：先修复宿主机 NVIDIA 驱动，Python 环境无法解决。
- `No matching distribution found for mmcv`：确认系统为 x86_64、Python 为 3.10，且没有更改脚本中的 PyTorch/CUDA 版本。
- `libGL.so.1` 缺失：服务器无桌面环境时执行 `pip uninstall -y opencv-python && pip install opencv-python-headless`。
- CUDA OOM：先改成 batch size 1、梯度累积 8；仍不足再降低图像尺寸。
- 不要同时安装 `mmcv` 和 `mmcv-lite`，后者不包含训练所需的 CUDA 算子。
