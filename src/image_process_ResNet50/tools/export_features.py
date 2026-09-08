"""导出 Backbone + FPN 的中间特征。

这个脚本不是常规推理，而是给后续研究/导航模块用的：
把一张图片送进训练好的检测模型，取出 FPN 的 P3-P7 多尺度特征，
保存成 .pt 文件。

给一张水下图片，使用训练好的 GFL 模型，把图片经过 ResNet-50 + FPN 后
得到的多尺度视觉特征 P3～P7 提取出来，并保存成 ``.pt`` 文件。该输出可供
ObjectNav 的感知融合或额外导航头使用；它不是最终检测框。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 配置包含项目自定义指标导入，因此即使这里只导出特征，也要保证 tools 包
# 能被 Config.fromfile 正常解析。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """加载指定模型和单张图片，前向提取并保存 P3-P7 张量。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', default='outputs/features.pt')
    args = parser.parse_args()

    import torch
    from mmcv.transforms import Compose
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import TRANSFORMS

    # 只借用模型定义和测试 pipeline，不创建 Runner 或优化器。
    cfg = Config.fromfile(args.config)

    # init_detector 会根据配置创建模型，并加载训练好的 checkpoint。
    model = init_detector(cfg, args.checkpoint, device='cuda:0')
    # 评估模式会固定 BatchNorm 统计量并关闭 Dropout，保证同图输出稳定。
    model.eval()

    # 复用测试阶段的数据处理 pipeline，保证输入图片的 Resize/Normalize
    # 和训练评估时保持一致。
    # 这是config里面定义的，这里把这些配置创建成可运行的数据处理步骤，Compose的作用是把这些步骤组合成一个整体的处理流程
    # TRANSFORMS.build 把配置字典实例化，再由 Compose 串成与评估一致的流程。
    pipeline = Compose([TRANSFORMS.build(x) for x in cfg.test_pipeline])
    # img_id 只是单图数据样本需要的元数据占位，不对应训练 COCO 的 image_id。
    data = pipeline(dict(img_path=args.image, img_id=0))

    # pipeline 做完基础的数据处理以后，再进行一次“模型真正需要的输入预处理”，data_preprocessor 会完成归一化、pad、打包 batch 等模型前处理。
    # data_preprocessor 增加 batch 维、执行归一化并 pad 到 32 的倍数。
    batch = model.data_preprocessor(
        dict(inputs=[data['inputs']], data_samples=[data['data_samples']]),
        False,
    )

    # no_grad 表示只前向计算，不记录梯度，省显存。
    with torch.no_grad():
        # 返回五个张量，每个形状为 (batch, channels, height, width)；层级越高，
        # 空间分辨率越低、感受野越大。
        feats = model.extract_feat(batch['inputs'])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 保存前转到 CPU，避免之后 torch.load 时强制要求原 CUDA 设备存在。
    torch.save(
        {'features': [x.cpu() for x in feats], 'levels': ['P3', 'P4', 'P5', 'P6', 'P7']},
        out,
    )
    print([tuple(x.shape) for x in feats])


if __name__ == '__main__':
    main()
