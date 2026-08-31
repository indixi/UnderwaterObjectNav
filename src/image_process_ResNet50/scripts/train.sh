#!/usr/bin/env bash

# Linux/Ubuntu 一键训练脚本。
#
# 用法一：数据放在默认位置时，直接运行
#   bash scripts/train.sh
#
# 用法二：数据放在别的位置时，把数据路径作为第一个参数
#   bash scripts/train.sh /path/to/DUO
#
# 这个脚本本身不写模型结构，真正的训练逻辑在 tools/train.py，
# 模型和超参数在 configs/gfl_r50_fpn_duo_base.py。

# 遇到错误立即退出；未定义变量时报错；管道中任一命令失败就整体失败。
# 这能让脚本在环境或路径出错时尽早停止，而不是继续跑出更难懂的错误。
set -euo pipefail

# 无论用户从哪个目录调用此脚本，都先计算脚本自身和项目根目录的位置。
# 这样 configs/、tools/ 和 work_dirs/ 使用的相对路径始终保持正确。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

# 第一个位置参数是 DUO 数据根目录。如果没有提供，则使用新环境布局：
#   duo_gfl_project/data/DUO
# 因此数据放在默认位置时，直接运行 bash scripts/train.sh 即可。
DATA_ROOT="${1:-${PROJECT_ROOT}/data/DUO}"

# 切换到项目根目录后再启动训练，确保配置继承和输出目录均可解析。
cd "${PROJECT_ROOT}"

# --resume auto：work_dir 中存在 latest checkpoint 时自动续训；首次训练时
# MMEngine 会从头开始。--amp 启用混合精度，降低显存占用并提高吞吐量。
# 如果 RTX4060 8GB 显存不足，可以不用这个脚本，改为手动执行：
#   python tools/train.py ... --batch-size 1 --accumulation 8 --workers 2
python tools/train.py \
  --config configs/gfl_r50_fpn_duo_base.py \
  --data-root "${DATA_ROOT}" \
  --work-dir work_dirs/gfl_r50_fpn_duo \
  --seed 42 --amp --resume auto
