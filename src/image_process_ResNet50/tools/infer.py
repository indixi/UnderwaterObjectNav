"""对单张图片或目录推理，并输出适合导航模块使用的 JSON。

默认只返回海胆，置信度阈值可直接读取验证集校准文件。每个检测同时保存
像素坐标 xyxy 和归一化中心点/宽高，后者不依赖相机分辨率，更方便下游
控制器使用。``--all-classes`` 可用于调试时同时观察岩石预测。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


# 让 MMEngine 能导入配置引用的 tools.echinus_precision_metric。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 这些键与检测框一一对应，过滤目标类别时必须用同一组索引同步筛选；
# image 和 image_size 属于图像级字段，不能因为列表长度巧合而被过滤。
PER_DETECTION_KEYS = frozenset({
    'boxes_xyxy', 'scores', 'labels', 'class_names',
    'boxes_cxcywh_normalized',
})


def serialize(sample, image_path: Path, threshold: float, classes):
    """把 GPU 上的 DetDataSample 转成可 JSON 序列化的普通字典。"""
    pred = sample.pred_instances.cpu()
    # 二次过滤保证 JSON 与可视化调用使用同一阈值。
    keep = pred.scores >= threshold
    boxes = pred.bboxes[keep].numpy().tolist()
    scores = pred.scores[keep].numpy().tolist()
    labels = pred.labels[keep].numpy().tolist()
    if any(label < 0 or label >= len(classes) for label in labels):
        raise ValueError(f'prediction label is outside checkpoint classes: {classes}')

    with Image.open(image_path) as image:
        width, height = image.size
    # 归一化格式依次为中心 x、中心 y、宽、高，数值通常位于 [0,1]。
    normalized = [
        [
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]
        for x1, y1, x2, y2 in boxes
    ]
    return {
        'image': str(image_path),
        'boxes_xyxy': boxes,
        'scores': scores,
        'labels': labels,
        'class_names': [classes[label] for label in labels],
        'boxes_cxcywh_normalized': normalized,
        'image_size': [height, width],
    }


def filter_target(result, target_classes):
    """只保留指定类别，并维持所有逐检测字段严格对齐。"""
    target_classes = set(target_classes)
    indices = [
        index for index, name in enumerate(result['class_names'])
        if name in target_classes
    ]
    return {
        key: [value[index] for index in indices] if key in PER_DETECTION_KEYS else value
        for key, value in result.items()
    }


def load_threshold(score_threshold, threshold_file):
    """解析阈值来源；校准文件优先方案与手工阈值互斥。"""
    if score_threshold is not None and threshold_file:
        raise ValueError('--score-thr and --threshold-file are mutually exclusive')
    if threshold_file:
        doc = json.loads(Path(threshold_file).read_text(encoding='utf-8'))
        return float(doc['recommended_threshold'])
    # 未指定时使用偏保守的 0.5，而不是 MMDetection 可视化常用的 0.3。
    return 0.5 if score_threshold is None else score_threshold


def main():
    """加载模型、枚举输入图片、执行推理并写出汇总 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='outputs/inference')
    parser.add_argument('--score-thr', type=float)
    parser.add_argument('--threshold-file')
    parser.add_argument('--target-class', nargs='+', metavar='CLASS')
    parser.add_argument('--all-classes', action='store_true')
    parser.add_argument('--echinus-only', action='store_true',
                        help='deprecated compatibility option; echinus is already the default')
    args = parser.parse_args()
    if args.all_classes and args.target_class:
        parser.error('--all-classes and --target-class cannot be used together')

    threshold = load_threshold(args.score_thr, args.threshold_file)
    if not 0 <= threshold <= 1:
        parser.error('score threshold must be in [0, 1]')

    from mmdet.apis import DetInferencer

    input_path = Path(args.input)
    # 目录模式只读取常见位图后缀，忽略同目录 JSON 或其他辅助文件。
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(
            path for path in input_path.iterdir()
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    else:
        raise FileNotFoundError(input_path)
    if not paths:
        raise ValueError(f'no supported images found in {input_path}')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    inferencer = DetInferencer(model=args.config, weights=args.checkpoint)
    # 类别名称从模型/配置元数据读取，不再硬编码旧 DUO 四类，避免标签错位。
    classes = tuple(inferencer.model.dataset_meta.get('classes', ()))
    if not classes:
        raise ValueError('checkpoint/config does not provide dataset class metadata')

    # 海胆是导航目标，因此未传筛选参数时默认只输出 echinus。
    target_classes = None if args.all_classes else (args.target_class or ['echinus'])
    unknown = set(target_classes or ()) - set(classes)
    if unknown:
        raise ValueError(f'unknown target classes {sorted(unknown)}; available: {classes}')

    records = []
    for path in paths:
        result = inferencer(
            str(path),
            pred_score_thr=threshold,
            out_dir=str(output / 'visualizations'),
            no_save_pred=True,
            return_datasamples=True,
        )
        record = serialize(
            result['predictions'][0], path, threshold, classes)
        records.append(
            filter_target(record, target_classes) if target_classes else record)

    # 在文件顶层记录本次阈值和类别筛选条件，便于下游追溯推理配置。
    payload = {
        'score_threshold': threshold,
        'target_classes': target_classes or list(classes),
        'predictions': records,
    }
    (output / 'predictions.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output / 'predictions.json')


if __name__ == '__main__':
    main()
