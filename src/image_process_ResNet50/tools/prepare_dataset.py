"""把原始 RGB 导出整理成可随工程迁移的 COCO 数据集。

脚本完成以下工作：
1. 读取原始 ``coco_detection.json`` 和 ``image_mapping.csv``；
2. 把 COCO 未登记但人工确认不含海胆的 130 张图补成空标注负样本；
3. 依据 ``source_episode`` 整组划分 train/val/test，阻止相邻帧泄漏；
4. 在多个候选划分中选择图像数、类别实例数和负样本数最均衡的一组；
5. 复制 PNG、生成三份 COCO JSON、类别文件和可追溯的划分清单。

原始导出目录只读，所有新文件写入 image_process_ResNet50/data。除非显式
传入 ``--overwrite``，脚本不会覆盖已经生成的数据集。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from PIL import Image


# 无论从哪个工作目录启动，都根据脚本自身位置推导稳定的默认路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT.parents[2] / 'underwater_objectnav_rgb_export'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'underwater_objectnav_rgb'
SPLITS = ('train', 'val', 'test')


def parse_args():
    """解析数据源、输出目录、比例和随机种子等命令行参数。"""
    parser = argparse.ArgumentParser(
        description='Copy the RGB export and split it by source episode.')
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=float, nargs=3, default=(0.8, 0.1, 0.1),
                        metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--search-trials', type=int, default=30000,
                        help='random episode assignments evaluated for balance')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace an existing generated dataset')
    return parser.parse_args()


def load_source(source: Path):
    """读取并验证原始导出文件。

    返回原始 COCO 文档、映射表各行、CSV 字段、按文件名索引的 COCO 图片
    元数据以及原图目录。这里将 COCO ``categories`` 视为类别定义的唯一
    真值；``image_mapping.csv`` 只负责文件与 episode 的来源追踪。
    """
    coco_path = source / 'annotations' / 'coco_detection.json'
    mapping_path = source / 'image_mapping.csv'
    image_dir = source / 'images'
    for path in (coco_path, mapping_path, image_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    doc = json.loads(coco_path.read_text(encoding='utf-8'))
    with mapping_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        mapping_fields = reader.fieldnames or []

    required_fields = {
        'exported_filename', 'source_episode', 'source_filename',
        'source_relative_path',
    }
    if not required_fields.issubset(mapping_fields):
        missing = sorted(required_fields - set(mapping_fields))
        raise ValueError(f'image_mapping.csv missing columns: {missing}')

    # 训练标签顺序由 category_id 的升序确定。提前检查可防止海胆与岩石
    # 在模型输出中发生静默错位。
    categories = sorted(doc['categories'], key=lambda item: item['id'])
    if [item['name'] for item in categories] != ['echinus', 'rock']:
        raise ValueError(
            'Expected COCO categories ordered by id to be echinus, rock; found '
            f'{[(item["id"], item["name"]) for item in categories]}')

    # exported_filename 是复制、COCO file_name 和清单之间的主键，必须唯一。
    filenames = [row['exported_filename'] for row in rows]
    if len(filenames) != len(set(filenames)):
        duplicates = [name for name, count in Counter(filenames).items() if count > 1]
        raise ValueError(f'duplicate exported filenames: {duplicates[:10]}')

    coco_images = {item['file_name']: item for item in doc['images']}
    unknown = sorted(set(coco_images) - set(filenames))
    if unknown:
        raise ValueError(f'COCO images absent from image_mapping.csv: {unknown[:10]}')

    missing_files = [name for name in filenames if not (image_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f'missing exported PNG files: {missing_files[:10]}')

    return doc, rows, mapping_fields, coco_images, image_dir


def enrich_images(doc, rows, coco_images, image_dir: Path):
    """为映射表中的每张图片生成完整 COCO ``images`` 元数据。

    已在原 COCO 中登记的图片保留原始 image id，并核对实际宽高；未登记的
    图片分配新 id，但不创建 annotation，因此会成为合法的空标注负样本。
    同时返回每张图片的标注数量，供划分均衡和清单输出使用。
    """
    annotations_by_image = defaultdict(list)
    for ann in doc['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # 新负样本 id 从原始最大 image id 的下一个值开始，避免与原标注冲突。
    next_id = max((item['id'] for item in doc['images']), default=-1) + 1
    all_images = {}
    image_annotation_count = {}

    for row in rows:
        name = row['exported_filename']
        path = image_dir / name
        # 只读取图片头部即可获得尺寸；Pillow 的上下文管理器确保句柄及时关闭。
        with Image.open(path) as image:
            width, height = image.size

        if name in coco_images:
            item = deepcopy(coco_images[name])
            if (item['width'], item['height']) != (width, height):
                raise ValueError(
                    f'image size mismatch for {name}: COCO '
                    f'{item["width"]}x{item["height"]}, file {width}x{height}')
        else:
            item = {
                'id': next_id,
                'file_name': name,
                'width': width,
                'height': height,
                'license': 0,
                'url': '',
                'date_captured': '',
            }
            next_id += 1

        all_images[name] = item
        image_annotation_count[name] = len(annotations_by_image[item['id']])

    return all_images, image_annotation_count


def group_statistics(rows, all_images, annotations, category_ids):
    """按 episode 汇总图像数、空图数和各类别实例数。

    划分的最小单位是 episode，而不是单帧。这样同一轨迹中的高度相似帧
    不会同时出现在训练集和测试集，评估结果更接近未知轨迹上的表现。
    """
    by_image = defaultdict(Counter)
    for ann in annotations:
        by_image[ann['image_id']][ann['category_id']] += 1

    groups = defaultdict(lambda: Counter(images=0, empty_images=0))
    for row in rows:
        episode = row['source_episode']
        image = all_images[row['exported_filename']]
        counts = by_image[image['id']]
        groups[episode]['images'] += 1
        if not counts:
            groups[episode]['empty_images'] += 1
        for category_id in category_ids:
            groups[episode][f'category_{category_id}'] += counts[category_id]
    return dict(groups)


def choose_episode_split(groups, ratios, seed, trials, target_category_id):
    """从大量随机 episode 分配中选择最均衡且可复现的方案。

    评分同时考虑总图像数、空标注负样本数、每类实例数和 episode 数量。
    图像数权重最高，负样本次之；并强制三个集合都有海胆和负样本，保证
    precision 校准与最终评估有意义。固定 seed 后，相同输入会得到相同结果。
    """
    rng = random.Random(seed)
    episodes = sorted(groups)
    metric_names = ['images', 'empty_images'] + sorted(
        {key for stats in groups.values() for key in stats if key.startswith('category_')})
    totals = Counter()
    for stats in groups.values():
        totals.update(stats)

    cumulative = (ratios[0], ratios[0] + ratios[1])
    best = None
    best_score = float('inf')

    # 数据只有几十个 episode，随机搜索 30000 次成本很低，却比简单按 episode
    # 数量切片更能平衡大小差异明显的轨迹。
    for _ in range(trials):
        assignment = {}
        split_groups = Counter()
        split_stats = {split: Counter() for split in SPLITS}
        for episode in episodes:
            value = rng.random()
            split = 'train' if value < cumulative[0] else (
                'val' if value < cumulative[1] else 'test')
            assignment[episode] = split
            split_groups[split] += 1
            split_stats[split].update(groups[episode])

        if any(split_groups[split] == 0 for split in SPLITS):
            continue
        target_key = f'category_{target_category_id}'
        if any(split_stats[split][target_key] == 0 for split in SPLITS):
            continue
        if any(split_stats[split]['empty_images'] == 0 for split in SPLITS):
            continue

        # 采用相对误差平方，避免数量级较大的岩石实例完全支配评分。
        score = 0.0
        for split, ratio in zip(SPLITS, ratios):
            for metric in metric_names:
                target = totals[metric] * ratio
                weight = 4.0 if metric == 'images' else 2.0 if metric == 'empty_images' else 1.0
                score += weight * ((split_stats[split][metric] - target) / max(target, 1.0)) ** 2
            target_groups = len(episodes) * ratio
            score += 0.25 * ((split_groups[split] - target_groups) / max(target_groups, 1.0)) ** 2

        if score < best_score:
            best_score = score
            best = assignment

    if best is None:
        raise RuntimeError('could not find a valid episode-level split')
    return best


def ensure_output(output: Path, overwrite: bool):
    """创建输出骨架，并对覆盖操作实施路径安全检查。

    只有显式指定 --overwrite 才会删除旧结果；删除前还要求目标位于工程根
    目录内部，防止错误参数递归删除工作区或更高层目录。
    """
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f'{output} already exists; pass --overwrite to regenerate it')
        resolved = output.resolve()
        if resolved == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() not in resolved.parents:
            raise ValueError(f'refusing to replace unsafe output path: {resolved}')
        shutil.rmtree(resolved)
    for split in SPLITS:
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(parents=True, exist_ok=True)


def write_dataset(source, output, doc, rows, mapping_fields, all_images,
                  annotation_counts, assignment, image_dir, seed, ratios, overwrite):
    """复制图片并写出拆分 COCO、追踪清单和统计摘要。"""
    ensure_output(output, overwrite)
    row_by_name = {row['exported_filename']: row for row in rows}
    split_by_name = {
        name: assignment[row_by_name[name]['source_episode']] for name in all_images
    }
    # 标注通过 image_id 找所属集合，图片通过文件名找所属 episode；两个索引
    # 将 COCO 与 image_mapping.csv 严格连接起来。
    split_by_id = {item['id']: split_by_name[name] for name, item in all_images.items()}

    summary = {
        'source_root': str(source.resolve()),
        'seed': seed,
        'requested_ratios': dict(zip(SPLITS, ratios)),
        'categories': sorted(doc['categories'], key=lambda item: item['id']),
        'splits': {},
    }

    annotations_by_split = {split: [] for split in SPLITS}
    # annotation id 与 image id 均保持原值，便于回溯原始 coco_detection.json。
    for ann in doc['annotations']:
        annotations_by_split[split_by_id[ann['image_id']]].append(ann)

    images_by_split = {split: [] for split in SPLITS}
    for name, item in all_images.items():
        images_by_split[split_by_name[name]].append(item)

    for split in SPLITS:
        images = sorted(images_by_split[split], key=lambda item: item['id'])
        annotations = sorted(annotations_by_split[split], key=lambda item: item['id'])
        # 三份 JSON 都保留原始 info/licenses/categories，只替换当前集合的
        # images 和 annotations。空标注图只出现在 images 中，这是标准 COCO 表达。
        split_doc = {
            'info': {
                **deepcopy(doc.get('info', {})),
                'description': f'Underwater ObjectNav RGB {split} split (episode-level)',
                'split_seed': seed,
            },
            'licenses': deepcopy(doc.get('licenses', [])),
            'categories': deepcopy(doc['categories']),
            'images': images,
            'annotations': annotations,
        }
        annotation_path = output / 'annotations' / f'instances_{split}.json'
        annotation_path.write_text(
            json.dumps(split_doc, ensure_ascii=False, indent=2), encoding='utf-8')

        category_counts = Counter(ann['category_id'] for ann in annotations)
        episodes = sorted({
            row_by_name[item['file_name']]['source_episode'] for item in images
        })
        summary['splits'][split] = {
            'episodes': len(episodes),
            'episode_names': episodes,
            'images': len(images),
            'annotations': len(annotations),
            'empty_images': sum(annotation_counts[item['file_name']] == 0 for item in images),
            'instances_per_category_id': {
                str(category['id']): category_counts[category['id']]
                for category in sorted(doc['categories'], key=lambda item: item['id'])
            },
        }

        # 物理复制到独立 split 目录，让 MMDetection 的 data_prefix 简单明确，
        # 也使整个 image_process_ResNet50 文件夹可以直接打包迁移。
        for item in images:
            name = item['file_name']
            shutil.copy2(image_dir / name, output / 'images' / split / name)

    # 保存未经拆分的原始 COCO 与原映射表，后续审计时可逐项追溯。
    shutil.copy2(source / 'annotations' / 'coco_detection.json',
                 output / 'annotations' / 'coco_detection_original.json')
    shutil.copy2(source / 'image_mapping.csv', output / 'image_mapping.csv')
    category_names = [
        category['name'] for category in sorted(doc['categories'], key=lambda item: item['id'])
    ]
    (output / 'classes.txt').write_text('\n'.join(category_names) + '\n', encoding='utf-8')

    # split_manifest.csv 是最终划分的权威清单：既保留原来源字段，又增加
    # split、是否为负样本和标注数，训练前可据此检查 episode 泄漏。
    manifest_fields = mapping_fields + ['split', 'is_negative', 'annotation_count']
    with (output / 'split_manifest.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for row in rows:
            name = row['exported_filename']
            writer.writerow({
                **row,
                'split': split_by_name[name],
                'is_negative': int(annotation_counts[name] == 0),
                'annotation_count': annotation_counts[name],
            })

    (output / 'dataset_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    """执行完整的数据验证、划分与复制流程。"""
    args = parse_args()
    if any(value <= 0 for value in args.ratios) or abs(sum(args.ratios) - 1.0) > 1e-9:
        raise SystemExit('--ratios must contain three positive values that sum to 1')

    source = args.source_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    doc, rows, fields, coco_images, image_dir = load_source(source)
    all_images, annotation_counts = enrich_images(
        doc, rows, coco_images, image_dir)
    category_ids = [item['id'] for item in doc['categories']]
    target_category_id = next(
        item['id'] for item in doc['categories'] if item['name'] == 'echinus')
    groups = group_statistics(rows, all_images, doc['annotations'], category_ids)
    assignment = choose_episode_split(
        groups, tuple(args.ratios), args.seed, args.search_trials,
        target_category_id)
    summary = write_dataset(
        source, output, doc, rows, fields, all_images, annotation_counts,
        assignment, image_dir, args.seed, tuple(args.ratios), args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
