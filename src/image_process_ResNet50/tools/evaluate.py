"""用训练好的 checkpoint 在 DUO test 集上单独评估。

训练过程中每个 epoch 后会自动验证；这个脚本用于训练结束后，
拿某一个指定权重文件重新算 COCO 指标。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--work-dir', default='work_dirs/evaluation')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # 指定使用哪张 GPU。单卡机器一般就是 0。
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    from mmengine.config import Config
    from mmengine.runner import Runner

    root = Path(args.data_root).resolve()
    ann = str(root / 'annotations/instances_test.json')

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    # load_from 表示加载哪个模型权重进行测试。
    cfg.load_from = args.checkpoint

    # 用命令行传入的数据路径覆盖配置里的默认路径。
    cfg.test_dataloader.dataset.data_root = str(root) + os.sep
    cfg.test_evaluator.ann_file = ann

    # Runner.test() 只评估，不训练。
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
