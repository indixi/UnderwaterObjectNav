"""在验证集或测试集上完整评估指定 checkpoint。

默认评估 test，用于训练和阈值选择完成后的最终无偏报告；也可传 ``--split
val`` 复查训练期指标。Runner 会同时执行标准 COCO bbox 指标和海胆 precision
优先指标，但本脚本不会更新模型参数。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# 自定义指标位于工程 tools 包中，需保证工程根目录可被 Python 找到。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """按命令行选择数据 split，覆盖配置路径后调用 Runner.test()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='data/underwater_objectnav_rgb')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--work-dir', default='work_dirs/evaluation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--min-echinus-precision', type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.min_echinus_precision <= 1:
        parser.error('--min-echinus-precision must be in (0, 1]')

    # 必须在导入 MMEngine/PyTorch 前设置可见显卡。
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from mmengine.config import Config
    from mmengine.runner import Runner

    root = Path(args.data_root).resolve()
    annotation = root / 'annotations' / f'instances_{args.split}.json'
    image_dir = root / 'images' / args.split
    if not annotation.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f'incomplete {args.split} split under {root}')

    # 配置默认指向 test；这里根据 --split 同时更新 dataloader 和 CocoMetric，
    # 防止出现“读取 val 图片却使用 test 标注”的隐蔽错误。
    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.load_from = args.checkpoint
    cfg.test_dataloader.dataset.data_root = str(root) + os.sep
    cfg.test_dataloader.dataset.ann_file = f'annotations/instances_{args.split}.json'
    cfg.test_dataloader.dataset.data_prefix = dict(img=f'images/{args.split}/')
    # CocoMetric 需要具体 ann_file；自定义指标只需同步 precision 约束。
    for evaluator in cfg.test_evaluator:
        if evaluator.type == 'CocoMetric':
            evaluator.ann_file = str(annotation)
        elif evaluator.type == 'TargetPrecisionMetric':
            evaluator.min_precision = args.min_echinus_precision
    # test() 只做前向推理和指标汇总，不创建优化器，也不会修改 checkpoint。
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
