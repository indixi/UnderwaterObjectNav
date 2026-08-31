"""对单张图片或文件夹做推理，并输出适合导航使用的 JSON。

推理和训练不同：
训练会更新模型参数；推理只是把图片输入训练好的模型，得到检测框和分数。

输出内容包括：
1. 原始 xyxy 检测框；
2. 置信度分数；
3. 类别名称；
4. 归一化后的 cx/cy/w/h，方便后续导航模块使用；
5. 可视化图片。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')


def serialize(sample, image_path: Path, threshold: float):
    """把 MMDetection 的预测结果转成普通 Python 字典。

    sample.pred_instances 里包含模型预测出的 bboxes、scores、labels。
    这些数据一开始可能在 GPU 上，所以先 .cpu() 移到 CPU，再转成 list，
    这样 json.dumps 才能保存。
    """
    pred = sample.pred_instances.cpu()

    # 只保留置信度大于阈值的检测结果。
    keep = pred.scores >= threshold
    boxes = pred.bboxes[keep].numpy().tolist()
    scores = pred.scores[keep].numpy().tolist()
    labels = pred.labels[keep].numpy().tolist()

    with Image.open(image_path) as image:
        width, height = image.size

    # xyxy: 左上角 x/y + 右下角 x/y，单位是像素。
    # cxcywh_normalized: 中心点 x/y + 宽高，除以图片宽高后变成 0 到 1，
    # 更适合传给后续导航或控制模块。
    normalized = []
    for x1, y1, x2, y2 in boxes:
        normalized.append([
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        ])

    return {
        'image': str(image_path),
        'boxes_xyxy': boxes,
        'scores': scores,
        'labels': labels,
        'class_names': [CLASSES[i] for i in labels],
        'boxes_cxcywh_normalized': normalized,
        'image_size': [height, width],
    }


def filter_target(result, target_class='echinus', score_threshold=.3):
    """只保留目标类别，默认只保留海胆 echinus。

    result 里的某些字段是“每个检测框一个值”的列表，例如 scores、boxes；
    这些字段需要按 ids 过滤。image、image_size 不是逐框字段，原样保留。
    """
    ids = [
        i for i, (name, score) in enumerate(zip(result['class_names'], result['scores']))
        if name == target_class and score >= score_threshold
    ]

    filtered = {}
    for key, value in result.items():
        is_per_detection_list = isinstance(value, list) and len(value) == len(result['scores'])
        filtered[key] = [value[i] for i in ids] if is_per_detection_list else value
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='outputs/inference')
    parser.add_argument('--score-thr', type=float, default=.3)
    parser.add_argument('--echinus-only', action='store_true')
    args = parser.parse_args()

    from mmdet.apis import DetInferencer

    inp = Path(args.input)
    if inp.is_file():
        paths = [inp]
    else:
        paths = sorted(
            x for x in inp.iterdir()
            if x.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # DetInferencer 是 MMDetection 提供的高级推理接口。
    inferencer = DetInferencer(model=args.config, weights=args.checkpoint)

    records = []
    for path in paths:
        result = inferencer(
            str(path),
            pred_score_thr=args.score_thr,
            out_dir=str(out / 'visualizations'),
            no_save_pred=True,
            return_datasamples=True,
        )
        record = serialize(result['predictions'][0], path, args.score_thr)
        records.append(filter_target(record) if args.echinus_only else record)

    (out / 'predictions.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
