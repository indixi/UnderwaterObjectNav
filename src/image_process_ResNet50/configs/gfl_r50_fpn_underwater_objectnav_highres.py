"""面向远距离小海胆的可选高分辨率实验配置。

该文件继承基础配置，只覆盖训练图像尺度、batch size 和梯度累积次数。
应先确保基础配置稳定收敛，再把本配置作为独立消融实验；RTX 4060 8GB
更容易在这个配置下显存不足。
"""

# MMEngine 会递归合并基础配置，未在本文件出现的模型、评估器和数据路径
# 全部沿用 gfl_r50_fpn_underwater_objectnav.py。
_base_ = ['./gfl_r50_fpn_underwater_objectnav.py']

# 将训练目标尺度从 960x640 提高到 1200x800，以保留小目标纹理；缩放范围
# 收窄到 0.9~1.1，避免单批图像尺寸波动过大造成显存峰值。
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=(1200, 800), ratio_range=(0.9, 1.1), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_delta=5),
    dict(type='PackDetInputs'),
]

# 高分辨率下单次只送入一张图，通过累积 8 次梯度保持有效 batch 约为 8，
# 从而尽量维持与基础实验可比的优化行为。
train_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=train_pipeline))
optim_wrapper = dict(accumulative_counts=8)
