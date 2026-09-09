#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
DATA_ROOT="${1:-${PROJECT_ROOT}/data/underwater_objectnav_rgb}"

cd "${PROJECT_ROOT}"
python tools/train.py \
  --config configs/gfl_r50_fpn_underwater_objectnav.py \
  --data-root "${DATA_ROOT}" \
  --work-dir work_dirs/gfl_r50_fpn_underwater_objectnav \
  --seed 42 --amp --epochs 48 --min-echinus-precision 0.95
