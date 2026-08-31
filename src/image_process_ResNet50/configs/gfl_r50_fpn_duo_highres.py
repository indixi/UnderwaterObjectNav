# 高分辨率实验配置。
#
# 这个文件继承基础配置，只改动训练输入尺寸、batch size 和梯度累积。
# 用途：基线模型训练稳定后，尝试更高分辨率，看小目标海胆是否能进一步提升。
# 风险：显存占用更高，RTX4060 8GB 上更容易 OOM。
_base_ = ['./gfl_r50_fpn_duo_base.py']

# 训练流水线基本保持不变，只把 RandomResize 的目标尺度从 960x640 提高到 1200x800。
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),

    # 更大的输入图像能保留更多细节，通常对远处小目标有帮助。
    # ratio_range=(0.9, 1.1) 比基础配置更窄，避免显存波动太大。
    dict(type='RandomResize', scale=(1200, 800), ratio_range=(0.9, 1.1), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.9, 1.1),
         saturation_range=(0.9, 1.1), hue_delta=5),
    dict(type='PackDetInputs'),
]

# 高分辨率更占显存，所以单次 batch 降到 1。
train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))

# 用 8 次梯度累积，让有效 batch size 仍约等于 8，和基础配置保持可比。
optim_wrapper = dict(accumulative_counts=8)
