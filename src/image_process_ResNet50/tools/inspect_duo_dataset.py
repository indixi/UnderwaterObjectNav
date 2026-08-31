"""训练前检查 DUO/COCO 数据集。

这个脚本不训练模型，只做数据“验货”：
1. 读取 instances_train.json 和 instances_test.json；
2. 统计图片数、标注框数、每类实例数；
3. 检查图片是否缺失/损坏，bbox 是否有明显问题；
4. 随机抽一些图片，把标注框画出来，方便人工看类别和框是否对。

输出位置默认是 outputs/dataset_audit/。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='outputs/dataset_audit')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_root)
    out = Path(args.output)
    (out / 'annotation_preview').mkdir(parents=True, exist_ok=True)

    report = {'data_root': str(root.resolve()), 'splits': {}}
    random.seed(args.seed)

    for split in ('train', 'test'):
        # COCO 标注文件包含 images、annotations、categories 三个核心字段。
        ann_file = root / 'annotations' / f'instances_{split}.json'
        doc = json.loads(ann_file.read_text(encoding='utf-8'))

        # cats: category_id -> category_name，例如 2 -> echinus。
        cats = {c['id']: c['name'] for c in doc['categories']}

        # byimg: image_id -> 这张图片上的所有标注框。
        byimg = defaultdict(list)
        counts = Counter()
        issues = Counter()
        areas = []

        for ann in doc['annotations']:
            byimg[ann['image_id']].append(ann)
            counts[cats.get(ann['category_id'], 'INVALID')] += 1

            # COCO bbox 格式是 [x, y, width, height]，不是 [x1, y1, x2, y2]。
            x, y, w, h = ann['bbox']
            areas.append(w * h)

            if w <= 0 or h <= 0:
                issues['zero_or_negative_area'] += 1
            if x < 0 or y < 0:
                issues['negative_coordinate'] += 1
            if ann['category_id'] not in cats:
                issues['invalid_category'] += 1

        images = {x['id']: x for x in doc['images']}
        missing = []
        corrupt = []

        for im in doc['images']:
            path = root / 'images' / split / im['file_name']
            if not path.exists():
                missing.append(im['file_name'])
                continue

            try:
                # verify() 只检查图片能否被打开，不会完整解码成数组，速度较快。
                with Image.open(path) as pic:
                    pic.verify()
            except Exception: corrupt.append(im['file_name'])

            for ann in byimg[im['id']]:
                x, y, w, h = ann['bbox']
                if x + w > im['width'] + .01 or y + h > im['height'] + .01:
                    issues['out_of_bounds'] += 1

        # 随机选择一些图片，画出标注框。这样可以肉眼确认框和类别文字是否对应。
        chosen = random.sample(list(images), min(args.samples, len(images)))
        colors = ['red', 'lime', 'yellow', 'cyan']

        for iid in chosen:
            im = images[iid]
            src = root / 'images' / split / im['file_name']
            if not src.exists():
                continue

            pic = Image.open(src).convert('RGB')
            drawer = ImageDraw.Draw(pic)

            for ann in byimg[iid]:
                x, y, w, h = ann['bbox']
                color_index = ann['category_id'] - 1
                color = colors[color_index % len(colors)]
                drawer.rectangle((x, y, x + w, y + h), outline=color, width=3)
                drawer.text((x, y), cats[ann['category_id']], fill=color)

            preview_name = f'{split}_{Path(im["file_name"]).name}'
            pic.save(out / 'annotation_preview' / preview_name)

        report['splits'][split] = {
            'images': len(images),
            'annotations': len(doc['annotations']),
            'instances_per_class': dict(counts),
            'empty_images': sum(not byimg[i] for i in images),
            'missing_images': missing,
            'corrupt_images': corrupt,
            'issues': dict(issues),
            'area': {
                'min': min(areas),
                'max': max(areas),
                'mean': sum(areas) / len(areas),
            },
        }

    # 保存机器可读的 JSON 报告，适合以后写脚本继续分析。
    (out / 'dataset_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    # 同时保存人更容易看的 Markdown 报告。
    lines = ['# DUO 数据检查报告', '', f"数据根目录：`{report['data_root']}`", '']
    for split, value in report['splits'].items():
        lines += [
            f'## {split}',
            '',
            f"- 图像：{value['images']}",
            f"- 实例：{value['annotations']}",
            f"- 每类实例：{value['instances_per_class']}",
            f"- 问题：{value['issues']}",
            f"- 缺失/损坏：{len(value['missing_images'])}/{len(value['corrupt_images'])}",
            '',
        ]

    report_path = out / 'dataset_report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(report_path)


if __name__ == '__main__':
    main()
