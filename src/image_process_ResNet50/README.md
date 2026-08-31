# DUO 海胆检测训练工程

本工程按需求文档实现 ResNet-50 + FPN + GFL 四类检测基线，`echinus`（海胆）是下游导航目标。已核验本地 DUO：训练集 6671 图/63998 实例，测试集 1111 图/10517 实例，标注为标准 COCO JSON，类别依次为 `holothurian, echinus, scallop, starfish`。

## 1. 数据与评估约定

新环境的数据路径为 `data/DUO`（从本工程目录出发），包含官方 `train` 和 `test`，没有独立 `val`。配置保留官方划分并暂用 test 做每轮验证及最终评估；正式论文若需无偏测试，应另取得官方验证方案，或在不发生相邻帧泄漏的前提下从 train 按视频/场景划分验证集，不能随机逐帧拆分。

## 2. 环境安装

建议新建 Python 3.10 环境。RTX 5060 必须先从 PyTorch 官方安装器选择当前支持 Blackwell 的 CUDA 构建，不要复制旧版 cu118 命令。随后安装 OpenMMLab 组件：

```powershell
conda create -n duo-gfl python=3.10 -y
conda activate duo-gfl
# 在 https://pytorch.org/get-started/locally/ 复制当前 Windows/CUDA 安装命令
python tools/check_env.py
```

先只安装并验证 PyTorch；`check_env.py` 会执行真实 CUDA 前向、反向与参数更新，并保存环境信息。不要在当前 PyTorch 环境安装 `openmim`：其 OpenDataLab 依赖会把 setuptools 降到 60.2.0，与新版本 PyTorch 冲突。

MMCV 必须与 PyTorch/CUDA 精确匹配。MMDetection 3.3 要求 `mmcv>=2.0.0,<2.2.0`，而 OpenMMLab 没有为所有新 PyTorch/CUDA 组合发布预编译包；确认匹配方案后，再安装 MMCV、MMEngine、MMDetection 和本文件的其余依赖。若安装日志出现 `.tar.gz` 而非 `.whl`，说明正在源码编译 MMCV，不应在未安装匹配 CUDA Toolkit 与 MSVC 构建工具时继续。

## 3. 训练前数据检查

```powershell
python tools/inspect_duo_dataset.py --data-root "data/DUO" --samples 20
```

结果位于 `outputs/dataset_audit/`，含 JSON/Markdown 报告和标注预览。确认框、类别及越界检查无误后再训练。

## 4. 训练

```powershell
python tools/train.py --config configs/gfl_r50_fpn_duo_base.py `
  --data-root "data/DUO" --work-dir work_dirs/gfl_r50_fpn_duo `
  --seed 42 --amp --resume auto
```

基线：960×640 保持比例、batch 2、梯度累积 4（有效 batch 约 8）、AMP、SGD 0.005、24 epoch、16/22 epoch 衰减。保存 `latest`、最近三个 checkpoint 和验证 mAP 最优权重，并输出逐类 COCO AP。常用覆盖参数：

```text
--batch-size 1 --accumulation 8 --workers 2 --lr 0.005 --epochs 24
--gpu 0 --resume auto --deterministic
```

20 图过拟合烟雾测试：

```powershell
python tools/train.py --config configs/gfl_r50_fpn_duo_base.py --data-root "data/DUO" --work-dir work_dirs/overfit20 --overfit 20 --epochs 12 --amp
```

若 OOM：先改 batch 1/累积 8；仍不足则把配置中尺度改为 `(800, 544)` 或 `(768, 512)`；再降低 workers。不要每轮调用 `empty_cache()`。高分辨率实验配置仅用于基线稳定后的独立消融。

## 5. 评估、推理与特征

```powershell
python tools/evaluate.py --config configs/gfl_r50_fpn_duo_base.py --checkpoint work_dirs/gfl_r50_fpn_duo/best_coco_bbox_mAP_epoch_*.pth --data-root "data/DUO"
python tools/infer.py --config configs/gfl_r50_fpn_duo_base.py --checkpoint MODEL.pth --input IMAGE_OR_FOLDER --echinus-only
python tools/export_features.py --config configs/gfl_r50_fpn_duo_base.py --checkpoint MODEL.pth --image IMAGE.jpg
```

推理输出包含 xyxy 框、分数、标签、类名、图像尺寸及归一化 cx/cy/w/h；`filter_target()` 可供导航代码直接调用。`export_features.py` 保存 P3–P7，从而为后续共享 Backbone+FPN 的导航头预留接口。

## 6. 结果解释

MMDetection 的 `CocoMetric(classwise=True)` 会报告 mAP@[.50:.95]、AP50、AP75、small/medium/large AP/AR 及每类 AP，其中 `echinus` 行即 AP_echinus。导航部署的 Precision/Recall/F1 与置信度阈值应在独立验证集上扫描后确定；COCO AP 评估不要使用部署时的 0.3 阈值截断。
