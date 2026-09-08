"""训练前审计 train/val/test 三个 COCO 集合并生成标注预览。

该脚本不加载模型，只检查数据本身：类别是否一致、图片是否缺失或损坏、
实际尺寸是否匹配 COCO、bbox 是否越界/退化、每类实例和负样本数量是否
合理。每个集合还会随机绘制若干标注图，供人工确认框与语义是否对应。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


# 颜色按 categories 的 id 排序后分配，不假设 category_id 必须从 1 连续增长。
SPLITS = ('train', 'val', 'test')
COLORS = ('red', 'lime', 'yellow', 'cyan', 'magenta', 'orange')


def main():
    """执行审计并输出 JSON、Markdown 报告和可视化 PNG。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/underwater_objectnav_rgb')
    parser.add_argument('--output', default='outputs/underwater_objectnav_dataset_audit')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    out = Path(args.output)
    preview_dir = out / 'annotation_preview'
    preview_dir.mkdir(parents=True, exist_ok=True)
    # JSON 报告保留完整问题列表，Markdown 报告提供快速人工阅读摘要。
    report = {'data_root': str(root), 'splits': {}}
    expected_categories = None

    for split_index, split in enumerate(SPLITS):
        ann_file = root / 'annotations' / f'instances_{split}.json'
        doc = json.loads(ann_file.read_text(encoding='utf-8'))
        # 三个集合必须拥有完全相同的 category_id/name 对应关系，否则模型
        # 在不同阶段可能把同一个连续 label 解释成不同语义。
        categories = tuple(
            (item['id'], item['name'])
            for item in sorted(doc['categories'], key=lambda item: item['id']))
        if expected_categories is None:
            expected_categories = categories
        elif categories != expected_categories:
            raise ValueError(f'{split} categories differ from train: {categories}')

        category_names = dict(categories)
        category_colors = {
            category_id: COLORS[index % len(COLORS)]
            for index, (category_id, _) in enumerate(categories)
        }
        # 建立 image_id 索引，避免为每条 annotation 线性扫描 images 列表。
        images = {item['id']: item for item in doc['images']}
        by_image = defaultdict(list)
        counts = Counter()
        issues = Counter()
        areas = []

        # COCO bbox 使用 [x, y, width, height]，不是右下角坐标。
        for ann in doc['annotations']:
            if ann['image_id'] not in images:
                issues['invalid_image_id'] += 1
                continue
            if ann['category_id'] not in category_names:
                issues['invalid_category_id'] += 1
                continue
            by_image[ann['image_id']].append(ann)
            counts[category_names[ann['category_id']]] += 1
            x, y, width, height = ann['bbox']
            areas.append(width * height)
            if width <= 0 or height <= 0:
                issues['zero_or_negative_area'] += 1
            if x < 0 or y < 0:
                issues['negative_coordinate'] += 1
            image = images[ann['image_id']]
            if x + width > image['width'] + 0.01 or y + height > image['height'] + 0.01:
                issues['out_of_bounds'] += 1

        missing = []
        corrupt = []
        dimension_mismatch = []
        # verify() 验证图像编码完整性而不把全部像素长期加载进内存；尺寸则在
        # verify 前读取并与 COCO 元数据比较。
        for image in doc['images']:
            path = root / 'images' / split / image['file_name']
            if not path.is_file():
                missing.append(image['file_name'])
                continue
            try:
                with Image.open(path) as picture:
                    if picture.size != (image['width'], image['height']):
                        dimension_mismatch.append(image['file_name'])
                    picture.verify()
            except Exception:
                corrupt.append(image['file_name'])

        # 每个 split 使用不同但固定的随机序列，确保重复执行获得相同预览，
        # 同时不会让 train/val/test 恰好抽到相同位置序号。
        rng = random.Random(args.seed + split_index)
        chosen = rng.sample(list(images), min(args.samples, len(images)))
        for image_id in chosen:
            image = images[image_id]
            source = root / 'images' / split / image['file_name']
            if not source.is_file():
                continue
            picture = Image.open(source).convert('RGB')
            drawer = ImageDraw.Draw(picture)
            # 空标注负样本不会画框，但仍会保存预览，方便检查它是否确实无海胆。
            for ann in by_image[image_id]:
                x, y, width, height = ann['bbox']
                color = category_colors[ann['category_id']]
                drawer.rectangle((x, y, x + width, y + height), outline=color, width=3)
                drawer.text((x, y), category_names[ann['category_id']], fill=color)
            picture.save(preview_dir / f'{split}_{image["file_name"]}')

        # empty_images 包括有 COCO images 记录但没有任何 annotation 的图片，
        # 本数据中它们就是用于抑制误报的 130 张确认负样本。
        report['splits'][split] = {
            'images': len(images),
            'annotations': len(doc['annotations']),
            'instances_per_class': dict(counts),
            'empty_images': sum(not by_image[image_id] for image_id in images),
            'missing_images': missing,
            'corrupt_images': corrupt,
            'dimension_mismatch': dimension_mismatch,
            'issues': dict(issues),
            'area': {
                'min': min(areas) if areas else None,
                'max': max(areas) if areas else None,
                'mean': sum(areas) / len(areas) if areas else None,
            },
        }

    # 机器可读 JSON 适合自动检查，Markdown 适合训练前快速验收。
    out.mkdir(parents=True, exist_ok=True)
    (out / 'dataset_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Underwater ObjectNav 数据检查报告', '', f'数据根目录：`{root}`', '']
    for split, values in report['splits'].items():
        lines.extend([
            f'## {split}', '',
            f'- 图像：{values["images"]}',
            f'- 实例：{values["annotations"]}',
            f'- 每类实例：{values["instances_per_class"]}',
            f'- 负样本图像：{values["empty_images"]}',
            f'- 标注问题：{values["issues"]}',
            f'- 缺失/损坏/尺寸不符：{len(values["missing_images"])}/'
            f'{len(values["corrupt_images"])}/{len(values["dimension_mismatch"])}', '',
        ])
    (out / 'dataset_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out / 'dataset_report.md')


if __name__ == '__main__':
    main()
