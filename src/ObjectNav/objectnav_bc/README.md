# Underwater ObjectNav BC

This package implements the first-stage pipeline in
`underwater_objectnav_bc_training_plan.md`:

`RGB-D + pose -> cumulative 7-channel semantic map -> 256-D map feature + 8-D goal embedding + sin/cos yaw -> 4-action policy`

The sibling `../image_process_ResNet50` directory contains detector training and inference.
Use its GFL checkpoint as the optional semantic detector during preprocessing;
the BC policy is trained separately on the collected episode dataset.

## Expected input

The collector format is:

```text
dataset/
  episode_0001/{rgb,depth,trajectory.csv}
```

`trajectory.csv` must contain `step_id,timestamp,rgb_path,depth_path,x,y,z,roll,pitch,yaw,goal_category,expert_action`.
Depth files are `.npy`; poses use world coordinates and radians.

## Commands

Run from `UnderwaterObjectNav/src/ObjectNav` with the workspace on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Get-Location).Path
pip install -r objectnav_bc/requirements.txt
python -m objectnav_bc.dataset.preprocess_dataset `
  --dataset-root DATASET `
  --output-root DATASET/processed `
  --intrinsics FX FY CX CY `
  --T-base-camera T_base_camera.npy `
  --config objectnav_bc/config/bc.yaml

python -m objectnav_bc.train.train_bc --data-root DATASET/processed --work-dir work_dirs/objectnav_bc
python -m objectnav_bc.eval.eval_offline --data-root DATASET/processed --checkpoint work_dirs/objectnav_bc/best_policy.pt
```

预处理默认读取 `objectnav_bc/config/bc.yaml` 的 `dataset` 部分，因此地图范围、分辨率和原点不需要写死在命令或代码中。`MapConfig` 中的默认值只作为其他 Python 程序直接调用、且没有提供 YAML 时的兜底。

Detector config, checkpoint, class order and score threshold are set under
`perception.detector` in `objectnav_bc/config/bc.yaml`. Relative paths are
resolved from the YAML directory. Set `enabled: false` to preprocess without
MMDetection. Cached JSON from `../image_process_ResNet50/tools/infer.py` can be
consumed by `JsonDetectionDetector`.

The dataset split is made once per episode (70/15/15), so adjacent frames from
one trajectory cannot leak across train/validation/test. The trainer reports
weighted CE loss, per-class accuracy, confusion matrix, and STOP precision and
recall. `eval_closed_loop.ClosedLoopPolicy` is the ROS-facing inference adapter.

With the current 0.5 m map resolution, `robot` and `visited` use one-cell
marking: `robot` keeps only the current cell, while `visited` accumulates all
cells occupied by the robot during the Episode.
