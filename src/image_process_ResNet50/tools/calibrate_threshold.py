"""在验证集上校准海胆检测置信度阈值。

脚本使用训练完成的 checkpoint 对整个 val 集推理，在 IoU>=0.5 的一对一
匹配规则下，寻找满足指定 precision（默认 95%）且 recall 最高的分数阈值。
阈值只能用 val 选择，不能用 test 调参，否则最终测试结果会产生数据泄漏。
输出 JSON 可由 ``tools/infer.py --threshold-file`` 直接读取。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# 确保配置中的项目自定义 Metric 可以被 MMEngine 导入。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.target_metrics import evaluate_target


def xywh_to_xyxy(box):
    """把 COCO 的 [x, y, width, height] 转为检测器使用的 xyxy。"""
    x, y, width, height = box
    return [x, y, x + width, y + height]


def main():
    """执行验证集推理、PR 工作点搜索并保存推荐阈值。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='data/underwater_objectnav_rgb')
    parser.add_argument('--output', default='outputs/echinus_threshold.json')
    parser.add_argument('--min-precision', type=float, default=0.95)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    # 校准前只丢弃极低分候选，避免预先假设最终部署阈值。
    parser.add_argument('--prediction-floor', type=float, default=0.001)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if not 0 < args.min_precision <= 1:
        parser.error('--min-precision must be in (0, 1]')

    from mmdet.apis import DetInferencer

    root = Path(args.data_root).resolve()
    # 类别顺序仍由 COCO categories 的 id 排序决定；模型 label 是该顺序的
    # 从零索引，因此 category_id=1 的海胆对应 target_label=0。
    doc = json.loads(
        (root / 'annotations' / 'instances_val.json').read_text(encoding='utf-8'))
    categories = sorted(doc['categories'], key=lambda item: item['id'])
    class_names = [item['name'] for item in categories]
    if 'echinus' not in class_names:
        raise ValueError(f'echinus absent from categories: {class_names}')
    target_label = class_names.index('echinus')
    echinus_category_id = next(item['id'] for item in categories if item['name'] == 'echinus')

    # 阈值目标只关心海胆。岩石预测会被模型显式分类，但不作为海胆 TP；若
    # 岩石被误分为海胆，它会自然进入海胆预测并被计为 FP。
    annotations = defaultdict(list)
    for ann in doc['annotations']:
        if ann['category_id'] == echinus_category_id:
            annotations[ann['image_id']].append(xywh_to_xyxy(ann['bbox']))

    # DetInferencer 同时加载配置和 checkpoint，并把预测框恢复到原图坐标。
    inferencer = DetInferencer(
        model=args.config, weights=args.checkpoint, device=args.device)
    records = []
    for index, image in enumerate(doc['images'], start=1):
        path = root / 'images' / 'val' / image['file_name']
        result = inferencer(
            str(path),
            pred_score_thr=args.prediction_floor,
            no_save_vis=True,
            no_save_pred=True,
            return_datasamples=True,
            show_progress=False,
        )
        pred = result['predictions'][0].pred_instances.cpu()
        # target_metrics 使用统一记录结构；这里仍保留所有预测标签，函数内部
        # 再过滤目标类，便于以后复用到其他类别。
        records.append({
            'pred_boxes': pred.bboxes.numpy().tolist(),
            'pred_scores': pred.scores.numpy().tolist(),
            'pred_labels': pred.labels.numpy().tolist(),
            'gt_boxes': annotations[image['id']],
            'gt_labels': [target_label] * len(annotations[image['id']]),
        })
        if index % 50 == 0 or index == len(doc['images']):
            print(f'calibration inference: {index}/{len(doc["images"])}')

    # 在所有实际预测分数处构造候选阈值，先满足 precision 约束，再最大化 recall。
    report = evaluate_target(
        records,
        target_label=target_label,
        iou_threshold=args.iou_thr,
        min_precision=args.min_precision,
    )
    output = {
        'split': 'val',
        'target_class': 'echinus',
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'recommended_threshold': report['recommended']['threshold'],
        **report,
    }
    # recommended_threshold 位于顶层，方便 infer.py 无需理解完整报告即可读取。
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
