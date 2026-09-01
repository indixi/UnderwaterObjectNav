"""导出 Backbone + FPN 的中间特征。

这个脚本不是常规推理，而是给后续研究/导航模块用的：
把一张图片送进训练好的检测模型，取出 FPN 的 P3-P7 多尺度特征，
保存成 .pt 文件。

给一张水下图片，使用你训练好的 GFL 模型，把图片经过 ResNet50 + FPN 后得到的多尺度视觉特征 P3～P7 提取出来，并保存成 .pt 文件
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
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

    cfg = Config.fromfile(args.config)  #加载配置文件

    # init_detector 会根据配置创建模型，并加载训练好的 checkpoint。
    model = init_detector(cfg, args.checkpoint, device='cuda:0')
    model.eval()    #设置为评估模式，关闭 dropout 和 batchnorm 等训练特性

    # 复用测试阶段的数据处理 pipeline，保证输入图片的 Resize/Normalize
    # 和训练评估时保持一致。
    # 这是config里面定义的，这里把这些配置创建成可运行的数据处理步骤，Compose的作用是把这些步骤组合成一个整体的处理流程
    pipeline = Compose([TRANSFORMS.build(x) for x in cfg.test_pipeline])    #创建数据处理 pipeline，cfg.test_pipeline 是一个列表，包含了每个数据处理步骤的配置字典
    # 也是按照config里面定义的，将数据处理成MMDet的DataSample格式，方便后续模型处理
    data = pipeline(dict(img_path=args.image, img_id=0))

    # pipeline 做完基础的数据处理以后，再进行一次“模型真正需要的输入预处理”，data_preprocessor 会完成归一化、pad、打包 batch 等模型前处理。
    batch = model.data_preprocessor(
        dict(inputs=[data['inputs']], data_samples=[data['data_samples']]),
        False,
    )

    # no_grad 表示只前向计算，不记录梯度，省显存。
    with torch.no_grad():
        feats = model.extract_feat(batch['inputs']) #提取特征，返回的是一个列表，包含了 P3-P7 多尺度特征图，每个特征图的 shape 是 (B, C, H, W)，其中 B 是 batch size，C 是通道数，H 和 W 是特征图的高和宽。

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # feats 是 P3-P7 多尺度特征图。保存到 CPU 上，方便之后在无 GPU 环境读取。
    torch.save(
        {'features': [x.cpu() for x in feats], 'levels': ['P3', 'P4', 'P5', 'P6', 'P7']},
        out,
    )
    print([tuple(x.shape) for x in feats])


if __name__ == '__main__':
    main()
