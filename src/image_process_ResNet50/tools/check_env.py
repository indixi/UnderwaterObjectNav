"""检查 PyTorch、CUDA 和 OpenMMLab 是否满足本项目的训练要求。

默认模式只检查 PyTorch 和 GPU，适合安装 MMCV 之前排查驱动问题。
``--strict`` 模式还会导入全部 OpenMMLab 组件，并在 GPU 上执行 MMCV NMS，
用于确认下载到的是包含 CUDA 算子的完整 ``mmcv``，而不是 ``mmcv-lite``。
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def parse_args():
    """解析命令行选项。"""
    parser = argparse.ArgumentParser(description='Validate the CUDA/OpenMMLab environment.')
    parser.add_argument('--strict', action='store_true',
                        help='also require compatible MMCV, MMEngine and MMDetection packages')
    return parser.parse_args()

def main():
    args = parse_args()

    # 第一步只导入 PyTorch。若这里失败，说明基础深度学习环境尚未安装，
    # 此时继续检查 MMCV 没有意义。
    try:
        import torch
    except ImportError as e: raise SystemExit('PyTorch is not installed.') from e

    # 打印并保存这些信息，便于复现实验或远程定位不同机器的版本差异。
    print('OS:', platform.platform())
    print('Python:', sys.version)
    print('PyTorch:', torch.__version__)
    print('CUDA runtime:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())

    # torch.cuda.is_available() 为 False 通常意味着安装了 CPU 版 PyTorch、
    # NVIDIA 驱动不可用，或者驱动版本低于当前 PyTorch CUDA Runtime 的要求。
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; check driver and PyTorch CUDA build.')

    print('GPU:', torch.cuda.get_device_name(0))
    print('Capability:', torch.cuda.get_device_capability(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2))

    # 不能只依赖 is_available()：这里实际创建 GPU 张量并执行矩阵计算、
    # 反向传播和优化器更新，以覆盖训练所需的基本 CUDA 路径。
    x = torch.randn(32, 32, device='cuda', requires_grad=True)
    loss = (x @ x).square().mean()
    loss.backward()
    torch.optim.SGD([x], lr=.01).step()
    torch.cuda.synchronize()
    print('CUDA forward/backward/update: OK')

    if args.strict:
        # MMDetection 3.3.0 依赖 MMCV >=2.0.0,<2.2.0。除了读取版本，导入
        # mmcv.ops.nms 还能确认 MMCV 的 C++/CUDA 扩展能够被动态加载。
        try:
            import mmcv, mmengine, mmdet
            from mmcv.ops import nms
        except (ImportError, ModuleNotFoundError) as e:
            raise SystemExit(f'OpenMMLab import failed: {e}') from e
        from packaging.version import Version
        versions = {
            'MMCV': mmcv.__version__,
            'MMEngine': mmengine.__version__,
            'MMDetection': mmdet.__version__,
        }
        print('OpenMMLab:', versions)
        if not (Version('2.0.0') <= Version(mmcv.__version__) < Version('2.2.0')):
            raise SystemExit(f'Incompatible MMCV {mmcv.__version__}; expected >=2.0.0,<2.2.0')

        # 构造两个高度重叠的检测框。在 IoU 阈值 0.5 下，NMS 应仅保留
        # 得分最高的一个框。张量位于 CUDA，因而此处验证的是 GPU 算子。
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda')
        scores = torch.tensor([0.9, 0.8], device='cuda')
        _, keep = nms(boxes, scores, 0.5)
        torch.cuda.synchronize()
        if keep.numel() != 1:
            raise SystemExit('MMCV CUDA NMS returned an unexpected result.')
        print('MMCV CUDA NMS: OK')

    # 将完整依赖列表和 nvidia-smi 输出写入实验目录。即使 nvidia-smi
    # 子进程异常，也不覆盖前面已经完成的核心 CUDA 检查结果。
    out = Path('outputs/environment')
    out.mkdir(parents=True, exist_ok=True)

    commands = [
        ('pip_freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
        ('nvidia_smi.txt', ['nvidia-smi']),
    ]
    for name, cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            (out / name).write_text(result.stdout, encoding='utf-8')
        except OSError:
            # 有些机器可能没有 nvidia-smi；前面的核心检查已经给出明确错误。
            pass


if __name__ == '__main__':
    main()
