# 这是 MMDetection 的配置文件，不是普通 Python 训练脚本。
# 训练时 tools/train.py 会读取这个文件，然后把这里的配置交给 MMEngine Runner。
#
# 本配置的目标：
#   1. 继承 MMDetection 官方 GFL + ResNet50 + FPN 检测器；
#   2. 把 COCO 的 80 类检测头改成 DUO 的 4 类；
#   3. 指定 DUO 数据路径、数据增强、训练轮数、优化器和评估指标。

# _base_ 表示“继承已有配置”。这里继承的是你本地下载的 MMDetection
# 官方 GFL 配置。官方配置里已经写好了 GFL 模型的大部分结构，例如
# ResNet-50 backbone、FPN neck、GFLHead、ATSSAssigner 和 NMS 参数。
#
# 注意：这个相对路径要求本项目和 mmdetection 源码目录同在 image_process 下：
#   image_process/
#   ├── duo_gfl_project/
#   └── mmdetection/
_base_ = [
    '../../mmdetection/configs/gfl/gfl_r50_fpn_1x_coco.py',
]

# DUO 数据集的 4 个类别。顺序必须和 COCO 标注文件 categories 的 id 顺序一致。
# 本项目最关心的是 echinus（海胆），但训练时保留四类能帮助模型区分类似海底生物。
classes = ('holothurian', 'echinus', 'scallop', 'starfish')

# metainfo 会传给 CocoDataset 和评估器，用于把类别 id 显示成类别名称。
metainfo = dict(classes=classes)

# Ubuntu 新环境的默认数据位置（相对于项目根目录）。训练入口传入
# --data-root 时会覆盖该值，因此也支持把数据存放在其他磁盘。
data_root = 'data/DUO/'

# 这里覆盖官方 GFL 配置里的部分 model 字段。
model = dict(
    # 使用 torchvision 的 ImageNet 预训练 ResNet-50 权重作为初始化。
    # 训练日志里出现 fc.weight/fc.bias 不匹配是正常的，因为分类模型的 fc 层
    # 在目标检测模型里用不到。
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

    # 官方 COCO 配置是 80 类；DUO 只有 4 类，所以检测头必须改成 4。
    bbox_head=dict(num_classes=4),

    # 输入图片会被 pad 到 32 的倍数，方便 FPN 多尺度特征图对齐。
    data_preprocessor=dict(pad_size_divisor=32),
)

# 训练阶段的数据处理流水线。
# 每张图片会按顺序经过：读图 -> 读框 -> 随机缩放 -> 随机翻转 -> 颜色扰动 -> 打包。
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),

    # 把图片缩放到接近 960x640，并在 0.8 到 1.2 之间随机改变尺度。
    # keep_ratio=True 表示保持原图宽高比，不把目标拉变形。
    dict(type='RandomResize', scale=(960, 640), ratio_range=(0.8, 1.2), keep_ratio=True),

    # 50% 概率左右翻转，提升模型对方向变化的鲁棒性。
    dict(type='RandomFlip', prob=0.5),

    # 轻微改变亮度、对比度、饱和度和色调，模拟水下光照变化。
    dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.9, 1.1),
         saturation_range=(0.9, 1.1), hue_delta=5),

    # 打包成 MMDetection 模型需要的 DetDataSample 格式。
    dict(type='PackDetInputs'),
]

# 验证/测试阶段的数据处理流水线。
# 评估时不能使用随机增强，否则每次指标会不稳定，所以这里只做固定 Resize。
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(960, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),

    # meta_keys 是评估和可视化需要保留的图像元信息。
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# 训练数据加载器。
train_dataloader = dict(
    # 每张显卡一次读 2 张图。RTX4060 8GB 如果 OOM，可以在命令行改成 --batch-size 1。
    batch_size=2, num_workers=2, persistent_workers=True,

    # shuffle=True 表示每个 epoch 打乱训练图片顺序。
    sampler=dict(type='DefaultSampler', shuffle=True),

    # CocoDataset 用 COCO JSON 标注读取图像和 bbox。
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/instances_train.json', data_prefix=dict(img='images/train/'),

                 # 过滤没有标注框的训练图；min_size=1 表示不额外过滤小图。
                 filter_cfg=dict(filter_empty_gt=True, min_size=1), pipeline=train_pipeline))

# 验证数据加载器。DUO 官方只有 train/test，没有单独 val；
# 当前项目暂时用 test 做每轮验证和最终评估。
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=metainfo, test_mode=True,
                 ann_file='annotations/instances_test.json', data_prefix=dict(img='images/test/'),
                 pipeline=test_pipeline))

# MMDetection 里 test_dataloader 用于单独评估命令；这里直接复用 val 设置。
test_dataloader = val_dataloader

# COCO bbox 指标评估器。
# classwise=True 会输出每个类别的 AP，例如 echinus 的 mAP。
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_test.json',
                     metric='bbox', classwise=True, format_only=False)
test_evaluator = val_evaluator

# 训练 24 个 epoch，并且每个 epoch 后验证一次。
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器包装器。
optim_wrapper = dict(
    # AmpOptimWrapper 表示开启自动混合精度训练，省显存、通常也更快。
    type='AmpOptimWrapper', loss_scale='dynamic', accumulative_counts=4,

    # SGD 是经典目标检测优化器。lr=0.005 对“有效 batch size 约 8”比较保守。
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),

    # 梯度裁剪：防止偶发的大梯度让训练不稳定。
    clip_grad=dict(max_norm=35, norm_type=2))

# 学习率调度。
param_scheduler = [
    # 前 500 个 iteration 从很小学习率慢慢升到目标学习率，叫 warmup。
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),

    # 第 16 和 22 个 epoch 把学习率乘以 0.1，帮助后期细调收敛。
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=24, milestones=[16, 22], gamma=0.1),
]

# 默认 hook 是训练过程中的“自动动作”，例如保存权重、打印日志。
default_hooks = dict(
    # 每个 epoch 保存一次；只保留最近 3 个；同时保存验证 mAP 最好的模型。
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP', rule='greater',
                    max_keep_ckpts=3, save_last=True),

    # 每 50 个 batch 打印一次 loss、学习率、显存等信息。
    logger=dict(type='LoggerHook', interval=50))

# 随机种子用于提高复现实验的概率。deterministic=False 通常速度更快。
randomness = dict(seed=42, deterministic=False)

# 自动按 batch size 缩放学习率。这里关闭，避免初学阶段出现“参数自动变了”的困惑。
auto_scale_lr = dict(enable=False, base_batch_size=8)
