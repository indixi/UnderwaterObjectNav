#!/usr/bin/env bash
set -Eeuo pipefail

# Ubuntu 20.04 + RTX 4060 Ti 的可复现 DUO/GFL 环境安装脚本。
#
# 使用前提：
#   1. 宿主机已经安装 NVIDIA 驱动，并且 nvidia-smi 能识别显卡；
#   2. 已创建并激活 Python 3.10 的 Conda/venv 环境；
#   3. 当前目录是 duo_gfl_project 项目根目录。
#
# 本脚本安装的是 PyTorch 自带的 CUDA 12.1 Runtime，因此正常训练不要求
# 系统另外安装完整 CUDA Toolkit。只有从源码编译 CUDA 扩展时才需要 nvcc，
# 而本脚本明确要求使用 MMCV 官方预编译 wheel，以避免源码编译。

# 默认使用当前虚拟环境中的 python。若机器上命令名称不同，可以这样运行：
#   PYTHON_BIN=python3.10 bash scripts/install_ubuntu2004.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# PyTorch 官方 CUDA 12.1 wheel 仓库。
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# OpenMMLab 为 CUDA 12.1 + PyTorch 2.1 构建的 MMCV wheel 仓库。
MMCV_INDEX_URL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"

# 不允许 root 直接安装，防止依赖写入系统 Python 并污染 Ubuntu 环境。
if [[ "${EUID}" -eq 0 ]]; then
  echo "Error: do not run this script as root. Activate a user virtual environment." >&2
  exit 1
fi

# 先检查 Python 命令是否存在，避免后续出现难以理解的 command not found。
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found." >&2
  exit 1
fi

# MMDetection 3.3 和本项目统一使用 Python 3.10。固定小版本可以减少
# pycocotools、MMCV 等含二进制扩展的软件包发生 ABI 不匹配的风险。
"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required; found {sys.version.split()[0]}")
PY

# nvidia-smi 来自宿主机 NVIDIA 驱动，与 Conda 环境无关。如果这里失败，
# 应先安装/修复驱动，而不是尝试重装 PyTorch。
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi was not found. Install a current NVIDIA driver first." >&2
  exit 1
fi

echo "[1/5] NVIDIA driver"
# 输出显卡型号、驱动版本和显存，便于保存实验环境记录。
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "[2/5] Python packaging tools"
# 对打包工具设置上限，避免旧版 OpenMMLab 项目与过新的 pip/setuptools
# 出现安装行为或元数据解析差异。
"${PYTHON_BIN}" -m pip install --upgrade "pip<25" "setuptools<70" wheel

echo "[3/5] PyTorch 2.1.0 with CUDA 12.1 runtime"
# RTX 4060 Ti 属于 Ada 架构（计算能力 8.9），该 CUDA 构建原生支持。
# index-url 确保 torch 来自 PyTorch CUDA 仓库，而不是安装成 CPU 版本。
"${PYTHON_BIN}" -m pip install \
  torch==2.1.0 torchvision==0.16.0 \
  --index-url "${TORCH_INDEX_URL}"

echo "[4/5] Prebuilt MMCV and OpenMMLab packages"
# --only-binary=mmcv 是安全阀：只接受预编译 wheel。如果版本/平台不匹配，
# 安装会立即失败，而不会悄悄下载 tar.gz 并开始耗时的本地 CUDA 编译。
"${PYTHON_BIN}" -m pip install \
  mmcv==2.1.0 \
  --only-binary=mmcv \
  --find-links "${MMCV_INDEX_URL}"
# 再安装与 MMCV 2.1.0 兼容的 MMEngine、MMDetection 和数据处理依赖。
"${PYTHON_BIN}" -m pip install -r requirements-ubuntu2004.txt

echo "[5/5] CUDA/MMCV validation"
# strict 模式不仅检查 torch.cuda.is_available()，还会真正执行：
#   - CUDA 张量前向、反向传播和参数更新；
#   - MMCV 编译扩展导入；
#   - GPU 上的 NMS 算子。
# 所有项目通过后，才能认为训练环境完整可用。
"${PYTHON_BIN}" tools/check_env.py --strict

echo "Environment installation completed successfully."
echo "Place the dataset in data/DUO, then run: bash scripts/train.sh"
