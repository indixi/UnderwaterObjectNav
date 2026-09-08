# Underwater ObjectNav 海胆检测

本工程使用 GFL + ResNet-50 + FPN 检测 `echinus`（海胆）和 `rock`（岩石）。类别及原始 `category_id` 以 `annotations/coco_detection.json` 的 `categories` 为准。模型输出标签采用 MMDetection 连续索引：`0=echinus`、`1=rock`。

设计目标是降低“把岩石或背景误报成海胆”的风险：保留岩石辅助类别，加入 130 张经人工确认不含海胆的负样本，并在验证集上选择满足海胆 precision ≥ 95% 时 recall 最高的阈值。95% 是验证集上的选择约束，不是对未知场景的性能保证。

## 数据目录

数据已复制到项目内部：

```text
data/underwater_objectnav_rgb/
├── annotations/
│   ├── coco_detection_original.json
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── images/train/
├── images/val/
├── images/test/
├── image_mapping.csv
├── split_manifest.csv
├── classes.txt
└── dataset_summary.json
```

划分使用 `image_mapping.csv` 的 `source_episode`，固定随机种子 42，比例约为 80%/10%/10%。同一 episode 不会跨集合。`split_manifest.csv` 记录每张图片所属集合、标注数量及是否为负样本。

如需从原始导出重新生成数据副本：

```bash
python tools/prepare_dataset.py \
  --source-root /path/to/underwater_objectnav_rgb_export \
  --output-root data/underwater_objectnav_rgb \
  --overwrite
```

## Ubuntu 20.04 + RTX 4060

完整安装步骤见 `UBUNTU2004.md`。安装完成后先检查数据：

```bash
python tools/inspect_dataset.py --data-root data/underwater_objectnav_rgb --samples 20
```

开始训练：

```bash
bash scripts/train.sh
```

默认使用 AMP、batch size 2、梯度累积 4，按 `echinus/recall_at_precision_95` 保存最佳 checkpoint。RTX 4060 若显存不足：

```bash
python tools/train.py \
  --batch-size 1 --accumulation 8 --workers 2 --amp --resume auto
```

若要求 precision ≥ 98%，增加 `--min-echinus-precision 0.98`。该参数同时控制验证指标和最佳 checkpoint 的选择规则。

## 评估与阈值校准

先使用验证集校准海胆阈值：

```bash
python tools/calibrate_threshold.py \
  --config configs/gfl_r50_fpn_underwater_objectnav.py \
  --checkpoint MODEL.pth --min-precision 0.95
```

再使用固定阈值进行最终 test 评估；测试集只用于一次无偏评估，不用于选择阈值：

```bash
python tools/evaluate.py \
  --config configs/gfl_r50_fpn_underwater_objectnav.py \
  --checkpoint MODEL.pth --split test \
  --threshold-file outputs/echinus_threshold.json
```

应先在 val 上运行校准，再用同一个 checkpoint 和生成的阈值文件评估 test。
测试日志中的 `echinus/*_at_operating_threshold` 是固定部署阈值下的最终结果；
不要在 test 上重新选择 score。

结果写入 `outputs/echinus_threshold.json`，其中包括推荐阈值、TP/FP/FN、precision、recall、F1，以及是否存在“检出至少一个海胆且满足 precision 约束”的有效工作点。匹配标准为 IoU ≥ 0.5。

## 推理与特征导出

推理默认只输出海胆，建议加载验证集校准出的阈值：

```bash
python tools/infer.py \
  --config configs/gfl_r50_fpn_underwater_objectnav.py \
  --checkpoint MODEL.pth --input IMAGE_OR_FOLDER \
  --threshold-file outputs/echinus_threshold.json
```

可用 `--score-thr 0.7` 手动指定阈值，用 `--all-classes` 同时输出海胆和岩石。导航 JSON 包含像素坐标 `xyxy` 与归一化 `cx/cy/w/h`。

导出 Backbone + FPN 的 P3–P7 特征：

```bash
python tools/export_features.py \
  --config configs/gfl_r50_fpn_underwater_objectnav.py \
  --checkpoint MODEL.pth --image IMAGE.png
```

旧配置文件名 `gfl_r50_fpn_duo_base.py` 和 `gfl_r50_fpn_duo_highres.py` 保留为兼容别名，新实验应使用 Underwater ObjectNav 命名的配置。
