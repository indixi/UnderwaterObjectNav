"""DUO 数据集的 MMDetection 训练入口。

这个脚本相当于“训练命令的总开关”：
1. 读取命令行参数，例如数据路径、训练轮数、batch size；
2. 检查 DUO 数据集目录和类别顺序是否正确；
3. 读取 configs/gfl_r50_fpn_duo_base.py 配置文件；
4. 根据命令行参数覆盖配置里的部分值；
5. 调用 MMEngine Runner 真正开始训练。

初学时可以把它理解成：
配置文件负责“模型和训练方案是什么”，本脚本负责“把方案跑起来”。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 类别顺序必须和 DUO COCO 标注文件中的 category id 顺序一致。
# 如果顺序错了，模型可能把“海胆”学成“海星”，所以 validate_data 会专门检查。
CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')


def parse_args():
    """解析命令行参数。

    例子：
        python tools/train.py --data-root data/DUO --amp --resume auto

    argparse 会把命令行参数变成一个对象，后面可以用 a.data_root、
    a.epochs 这种方式读取。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/gfl_r50_fpn_duo_base.py')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--work-dir', default='work_dirs/gfl_r50_fpn_duo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--accumulation', type=int)

    # BooleanOptionalAction 会同时生成 --amp 和 --no-amp。
    # 默认开启 AMP，因为 RTX4060 上混合精度通常更省显存。
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)

    # --resume auto 表示自动从 work_dir/latest.pth 续训；
    # --resume xxx.pth 表示从指定 checkpoint 续训。
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--deterministic', action='store_true')

    # 只取前 N 张训练图反复训练，用来做烟雾测试。它不是正式训练。
    parser.add_argument('--overfit', type=int, metavar='N',
                        help='repeat a fixed N-image subset for pipeline debugging')
    return parser.parse_args()


def validate_data(root: Path):
    """检查 DUO 数据集是否完整，并确认类别顺序没有错。

    参数:
        root: DUO 数据根目录，例如 data/DUO。

    返回:
        一个简单字典，记录 train/test 的图片数量和标注数量。
    """
    needed = [
        root / 'annotations/instances_train.json',
        root / 'annotations/instances_test.json',
        root / 'images/train',
        root / 'images/test',
    ]
    missing = [str(x) for x in needed if not x.exists()]    #从 needed 里找出缺失的文件或目录
    if missing:
        raise FileNotFoundError('DUO data is incomplete:\n  ' + '\n  '.join(missing))

    info = {}
    for split in ('train', 'test'):
        ann_file = root / 'annotations' / f'instances_{split}.json'
        data = json.loads(ann_file.read_text(encoding='utf-8')) # 读取并解析 JSON 标注文件

        # COCO categories 通常用 id 表示类别。按 id 排序后得到的类别名
        # 必须等于 CLASSES，否则训练出来的标签会和真实类别错位。
        names = tuple(c['name'] for c in sorted(data['categories'], key=lambda x: x['id'])) #tuple是不可变的列表，sorted是排序函数，key=lambda x: x['id']表示按id排序，这里是按类别id排序后得到的类别名
        if names != CLASSES:
            raise ValueError(f'{split} category mapping {names} != {CLASSES}')

        info[split] = {
            'images': len(data['images']),
            'annotations': len(data['annotations']),
        }
    return info


def main():
    """训练主流程。"""
    args = parse_args()

    # 让程序只看到指定 GPU。单卡训练时通常用 --gpu 0。
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as e:
        raise SystemExit('Missing MMDetection environment. Follow README installation steps.') from e

    # 先把数据路径转成绝对路径，避免从不同目录启动命令时找错数据。
    root = Path(args.data_root).expanduser().resolve()
    info = validate_data(root)

    # 读取配置文件。Config.fromfile 会处理 _base_ 继承关系。
    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(Path(args.work_dir).resolve())

    # 设置随机种子。deterministic=True 复现性更强，但可能稍慢。
    cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)

    # 配置文件里有默认 data_root；这里用命令行 --data-root 覆盖它。
    # train/val/test 三个 dataloader 都要改，否则可能训练读新路径、评估读旧路径。
    for loader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        loader.dataset.data_root = str(root) + os.sep

    # COCO 评估器也需要知道标注文件的绝对路径。
    ann = str(root / 'annotations/instances_test.json')
    cfg.val_evaluator.ann_file = ann
    cfg.test_evaluator.ann_file = ann

    # 下面这些 if 是命令行覆盖配置文件。
    # 例如配置里 batch_size=2，但命令行传 --batch-size 1，就以命令行为准。
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size

    if args.workers is not None:
        cfg.train_dataloader.num_workers = args.workers
        cfg.train_dataloader.persistent_workers = args.workers > 0
        cfg.val_dataloader.num_workers = args.workers
        cfg.val_dataloader.persistent_workers = args.workers > 0

    if args.lr:
        cfg.optim_wrapper.optimizer.lr = args.lr

    if args.epochs:
        cfg.train_cfg.max_epochs = args.epochs

    if args.accumulation:
        cfg.optim_wrapper.accumulative_counts = args.accumulation

    # 根据 --amp/--no-amp 选择是否使用混合精度。
    cfg.optim_wrapper.type = 'AmpOptimWrapper' if args.amp else 'OptimWrapper'

    if args.overfit:
        # RepeatDataset 会反复使用一个小子集，方便确认模型能否“记住”少量图片。
        # 这是调试训练链路用的，不代表正式测试性能。
        cfg.train_dataloader.dataset = dict(
            type='RepeatDataset',
            times=max(1, 200 // args.overfit),
            dataset=cfg.train_dataloader.dataset,
        )
        cfg.train_dataloader.dataset.dataset.indices = args.overfit

    if args.resume:
        cfg.resume = True
        cfg.load_from = None if args.resume == 'auto' else args.resume  #不是从零训练而是继续训练，如果等于auto则会自动从 work_dir/latest.pth 续训，否则从指定 checkpoint 续训，如果不是就需要提供具体的 checkpoint 路径

    # 保存一份最终解析后的配置。以后复现实验时，看这个文件最准确。
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    cfg.dump(str(Path(cfg.work_dir) / 'resolved_config.py'))    #把这次训练的配置保存到 work_dir/resolved_config.py，方便复现

    print('Classes:', dict(enumerate(CLASSES)))
    print('Dataset:', info)
    print('Work dir:', cfg.work_dir)

    try:
        Runner.from_cfg(cfg).train()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('\nCUDA OOM: use --batch-size 1 --accumulation 8; then lower image scale to (800,544).', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
