"""Underwater ObjectNav RGB 数据集的 GFL 训练入口。

职责包括：训练前验证三份 COCO 与 split_manifest、读取 MMEngine 配置、用
命令行覆盖常用超参数、设置海胆 precision 约束、保存最终解析配置并启动
Runner。模型结构仍由 configs 下的配置文件描述，本脚本负责安全地把一次
具体实验“组装并运行”起来。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# 将工程根目录放入模块搜索路径，使配置中的
# custom_imports=['tools.echinus_precision_metric'] 在任何启动目录下都可导入。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SPLITS = ('train', 'val', 'test')


def parse_args():
    """定义并校验训练命令行参数。

    大部分参数是对配置文件的可选覆盖；不提供时使用配置中的稳定基线值。
    ``--resume auto`` 只恢复当前 work_dir 的新两分类 checkpoint。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='configs/gfl_r50_fpn_underwater_objectnav.py')
    parser.add_argument('--data-root', default='data/underwater_objectnav_rgb')
    parser.add_argument(
        '--work-dir', default='work_dirs/gfl_r50_fpn_underwater_objectnav')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--accumulation', type=int)
    # 最佳模型不是按总体 mAP 选择，而是在这个海胆 precision 下最大化 recall。
    parser.add_argument('--min-echinus-precision', type=float, default=0.95)
    parser.add_argument(
        '--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument(
        '--overfit', type=int, metavar='N',
        help='repeat a fixed N-image subset for pipeline debugging')
    args = parser.parse_args()
    if not 0 < args.min_echinus_precision <= 1:
        parser.error('--min-echinus-precision must be in (0, 1]')
    return args


def validate_data(root: Path):
    """在导入重型训练组件前完整验证数据布局与划分关系。

    检查内容：三份标注和图片目录是否存在、类别顺序是否一致、COCO 图片
    是否都有实体文件、annotation 是否引用合法 image_id、manifest 与 COCO
    文件名是否一致，以及同一 source_episode 是否跨集合。

    返回模型连续类别顺序和各 split 的基础统计信息。
    """
    needed = []
    for split in SPLITS:
        needed.extend([
            root / 'annotations' / f'instances_{split}.json',
            root / 'images' / split,
        ])
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError('dataset is incomplete:\n  ' + '\n  '.join(missing))

    # 第一份标注建立预期类别顺序，后两份必须完全相同。
    expected_categories = None
    info = {}
    filenames_by_split = {}
    manifest_path = root / 'split_manifest.csv'
    for split in SPLITS:
        path = root / 'annotations' / f'instances_{split}.json'
        doc = json.loads(path.read_text(encoding='utf-8'))
        categories = tuple(
            item['name'] for item in sorted(doc['categories'], key=lambda item: item['id']))
        if expected_categories is None:
            expected_categories = categories
        elif categories != expected_categories:
            raise ValueError(f'{split} categories {categories} != {expected_categories}')
        image_files = {item['file_name'] for item in doc['images']}
        filenames_by_split[split] = image_files
        missing_images = [
            name for name in image_files if not (root / 'images' / split / name).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(
                f'{split} is missing image files: {missing_images[:10]}')
        # 标注引用不存在的 image_id 会在训练中造成难定位的数据加载错误，
        # 因此在创建 Runner 前就给出明确异常。
        image_ids = {item['id'] for item in doc['images']}
        invalid_annotations = [
            ann['id'] for ann in doc['annotations'] if ann['image_id'] not in image_ids
        ]
        if invalid_annotations:
            raise ValueError(
                f'{split} annotations reference missing images: {invalid_annotations[:10]}')
        annotated_image_ids = {ann['image_id'] for ann in doc['annotations']}
        info[split] = {
            'images': len(doc['images']),
            'annotations': len(doc['annotations']),
            'empty_images': len(image_ids - annotated_image_ids),
        }

    if expected_categories != ('echinus', 'rock'):
        raise ValueError(
            'COCO categories ordered by id must be (echinus, rock); found '
            f'{expected_categories}')
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    # utf-8-sig 同时兼容普通 UTF-8 与带 BOM 的 CSV。
    with manifest_path.open('r', encoding='utf-8-sig', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    episode_splits = defaultdict(set)
    manifest_names = {split: set() for split in SPLITS}
    for row in manifest:
        split = row.get('split')
        if split not in manifest_names:
            raise ValueError(f'invalid split in manifest: {split}')
        episode_splits[row['source_episode']].add(split)
        manifest_names[split].add(row['exported_filename'])
    leaking = sorted(
        episode for episode, split_names in episode_splits.items()
        if len(split_names) != 1)
    if leaking:
        raise ValueError(f'episodes cross dataset splits: {leaking[:10]}')
    for split in SPLITS:
        if manifest_names[split] != filenames_by_split[split]:
            raise ValueError(f'{split} manifest filenames differ from COCO images')
    return expected_categories, info


def set_dataset_root(loader, root, split, classes):
    """把一个 dataloader 指向指定 split，并同步 COCO 类别元数据。"""
    dataset = loader.dataset
    dataset.data_root = str(root) + os.sep
    dataset.ann_file = f'annotations/instances_{split}.json'
    dataset.data_prefix = dict(img=f'images/{split}/')
    # 保留配置中可视化 palette，只用 COCO 验证出的类别覆盖 classes。
    metainfo = dict(dataset.get('metainfo', {}))
    metainfo['classes'] = classes
    dataset.metainfo = metainfo


def set_coco_annotation(evaluators, annotation_path):
    """更新评估器列表中的 CocoMetric 标注路径。

    自定义 TargetPrecisionMetric 直接读取 data sample，不需要 ann_file。
    """
    for evaluator in evaluators:
        if evaluator.type == 'CocoMetric':
            evaluator.ann_file = str(annotation_path)


def main():
    """构造最终 MMEngine 配置并执行单卡训练。"""
    args = parse_args()
    # 在导入 torch/mmengine 前限制可见 GPU，保证 --gpu 参数真正生效。
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as error:
        raise SystemExit(
            'Missing MMDetection environment. Follow UBUNTU2004.md.') from error

    # 数据验证先于模型构建，能够快速发现路径或标签问题，避免 GPU 初始化后
    # 才失败。类别顺序来自 COCO，而不是 image_mapping.csv。
    root = Path(args.data_root).expanduser().resolve()
    classes, info = validate_data(root)
    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)
    # 检测头类别数和三个 dataloader 的 metainfo 统一由已验证的 COCO 决定。
    cfg.model.bbox_head.num_classes = len(classes)

    set_dataset_root(cfg.train_dataloader, root, 'train', classes)
    set_dataset_root(cfg.val_dataloader, root, 'val', classes)
    set_dataset_root(cfg.test_dataloader, root, 'test', classes)
    set_coco_annotation(
        cfg.val_evaluator, root / 'annotations' / 'instances_val.json')
    set_coco_annotation(
        cfg.test_evaluator, root / 'annotations' / 'instances_test.json')

    # CocoDataset 会按 metainfo.classes 将原始 category_id 映射成连续标签。
    # 此处动态查找海胆索引，避免未来 category_id 变化导致指标评错类别。
    target_label = classes.index('echinus')
    precision_suffix = int(round(args.min_echinus_precision * 100))
    for evaluators in (cfg.val_evaluator, cfg.test_evaluator):
        for evaluator in evaluators:
            if evaluator.type == 'TargetPrecisionMetric':
                evaluator.target_label = target_label
                evaluator.min_precision = args.min_echinus_precision
    cfg.default_hooks.checkpoint.save_best = (
        f'echinus/recall_at_precision_{precision_suffix}')

    # 以下只覆盖用户显式传入的值，未传参数继续使用配置基线。
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size
    if args.workers is not None:
        for loader in (
                cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
            loader.num_workers = args.workers
            loader.persistent_workers = args.workers > 0
    if args.lr:
        cfg.optim_wrapper.optimizer.lr = args.lr
    if args.epochs:
        cfg.train_cfg.max_epochs = args.epochs
    if args.accumulation:
        cfg.optim_wrapper.accumulative_counts = args.accumulation
    # AmpOptimWrapper 启用自动混合精度；关闭 AMP 时退回标准 FP32 优化器包装。
    cfg.optim_wrapper.type = 'AmpOptimWrapper' if args.amp else 'OptimWrapper'

    if args.overfit:
        # 烟雾测试只取固定前 N 张训练图并重复使用，目的是确认数据链路和模型
        # 能快速过拟合，不用于正式性能评估。
        if args.overfit <= 0:
            raise ValueError('--overfit must be positive')
        base_dataset = cfg.train_dataloader.dataset
        base_dataset.indices = args.overfit
        cfg.train_dataloader.dataset = dict(
            type='RepeatDataset',
            times=max(1, 200 // args.overfit),
            dataset=base_dataset)

    if args.resume:
        # auto 令 MMEngine 在 work_dir 中寻找 latest.pth；传具体路径则从该
        # checkpoint 恢复优化器、epoch 和模型状态。
        cfg.resume = True
        cfg.load_from = None if args.resume == 'auto' else args.resume

    # 保存所有继承与命令行覆盖后的最终配置，是之后复现实验的权威记录。
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    cfg.dump(str(Path(cfg.work_dir) / 'resolved_config.py'))
    print('Classes:', dict(enumerate(classes)))
    print('Dataset:', info)
    print('Echinus minimum precision:', args.min_echinus_precision)
    print('Work dir:', cfg.work_dir)

    try:
        Runner.from_cfg(cfg).train()
    except RuntimeError as error:
        # 保留原异常栈，同时针对 8GB RTX 4060 最常见的 OOM 给出可执行建议。
        if 'out of memory' in str(error).lower():
            print(
                '\nCUDA OOM: use --batch-size 1 --accumulation 8 '
                '--workers 2.', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
