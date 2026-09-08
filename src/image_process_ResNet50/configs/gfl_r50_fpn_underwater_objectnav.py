"""Underwater ObjectNav RGB 数据集的 GFL + ResNet-50 + FPN 配置。

这个文件是完整、自包含的 MMDetection 配置，不再依赖工程目录之外的
``mmdetection/configs`` 源码。训练入口 ``tools/train.py`` 会读取本文件，
再用命令行参数覆盖数据目录、训练轮数、batch size 等可变设置。

类别约定：COCO 原始 ``category_id`` 为 1/2，而 MMDetection 会把它们映射
为从 0 开始的连续训练标签，因此模型内部是 0=echinus、1=rock。
"""

# 让 MMEngine 在构建 Runner 前导入本项目自定义指标。该指标用于计算
# “海胆 precision 不低于 95% 时可达到的最大 recall”，并据此保存最佳权重。
custom_imports = dict(
    imports=['tools.echinus_precision_metric'], allow_failed_imports=False)

# classes 的顺序必须与三个 COCO 文件中 categories 按 id 排序后的顺序一致。
# palette 只影响可视化颜色：海胆为红色，岩石为蓝色，不参与模型计算。
classes = ('echinus', 'rock')
metainfo = dict(classes=classes, palette=[(255, 64, 64), (80, 180, 255)])
dataset_type = 'CocoDataset'
# 默认数据路径相对于 image_process_ResNet50 工程根目录；训练脚本也允许
# 通过 --data-root 覆盖，因此把整个目录复制到 Ubuntu 后不需要改绝对路径。
data_root = 'data/underwater_objectnav_rgb/'
# None 表示使用本地文件系统。若以后改用对象存储，可在这里配置 backend。
backend_args = None

# ------------------------------ 模型结构 ------------------------------
# GFL 是单阶段检测器；ResNet-50 提取视觉特征，FPN 将不同分辨率的特征融合，
# GFLHead 在 P3-P7 五个尺度上同时完成分类和边界框回归。
model = dict(
    type='GFL',
    # 使用 ImageNet 的均值/标准差归一化。bgr_to_rgb=True 是因为 MMDetection
    # 默认读入 BGR 图像，而 torchvision 的预训练 ResNet 使用 RGB 顺序。
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # 冻结 stem 和第一个 stage，保留通用低层特征并降低小数据集过拟合风险。
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        # 训练时固定 BatchNorm 的运行均值/方差，适合单卡小 batch 场景。
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # 从 ResNet 的 C3 开始构造 FPN，并额外产生更低分辨率的 P6、P7。
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        # 新数据集只有海胆和岩石两类，旧四分类 checkpoint 的检测头不能续训。
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # 每个位置只使用 1:1 基础 anchor；多尺度目标由五层 FPN 覆盖。
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            # Quality Focal Loss 同时建模类别置信度与定位质量。
            type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        # DFL 学习离散边界距离分布，GIoU 约束最终边界框几何位置。
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        # ATSS 根据各 FPN 层候选框的统计特性自适应选择正样本。
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    # 评估阶段保留最低 0.001 分数的候选框，避免在校准 95% precision 阈值前
    # 就提前丢掉预测；实际部署阈值由 tools/calibrate_threshold.py 决定。
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# ------------------------------ 数据增强 ------------------------------
# 训练流水线顺序为：读图 -> 读框 -> 随机缩放 -> 随机翻转 -> 水下颜色扰动 -> 打包。
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # 保持宽高比，长宽目标尺度为 960x640，并在 0.8~1.2 倍间随机变化。
    dict(type='RandomResize', scale=(960, 640), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        # 轻微扰动亮度、对比度、饱和度和色相，模拟水下照明与颜色漂移；
        # 幅度刻意保持较小，避免把合成图像改得脱离真实物理外观。
        brightness_delta=16,
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_delta=5),
    dict(type='PackDetInputs'),
]

# 验证和测试必须是确定性的，不能使用随机增强，否则同一 checkpoint 每次
# 得到的指标会变化。这里特意在 Resize 之后才 LoadAnnotations：这样读入的
# GT 框保持 COCO 原图坐标；检测器也会把预测框还原到原图坐标，两者可以由
# 自定义 echinus precision 指标直接比较。
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(960, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# ------------------------------ 数据加载 ------------------------------
# 单卡 batch=2 配合累积 4 次梯度，相当于有效 batch 约为 8。
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        # 关键设置：130 张无海胆图片是人工确认的困难负样本。若这里设为 True，
        # CocoDataset 会静默过滤它们，无法训练模型抑制“岩石/背景误报为海胆”。
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args))

# val 用于每轮选模型和校准阈值，绝不能与 test 混用。
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# test 只用于最终无偏评估，不参与 checkpoint 或置信度阈值选择。
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# ------------------------------ 评估指标 ------------------------------
# target_label=0 指模型内部的海胆连续标签，不是 COCO 原始 category_id=1。
# IoU>=0.5 才记作正确定位；目标是 precision>=0.95 时尽可能提高 recall。
target_metric = dict(
    type='TargetPrecisionMetric',
    target_label=0,
    target_name='echinus',
    iou_threshold=0.5,
    min_precision=0.95,
    reference_threshold=0.5)
val_evaluator = [
    # 标准 COCO AP/AP50/AP75 便于与其他检测方法横向比较。
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_val.json',
        metric='bbox',
        classwise=True,
        format_only=False,
        backend_args=backend_args),
    # 业务指标用于体现“宁可少报，也不能把其他物体误报成海胆”的目标。
    target_metric,
]
# test 使用同样的指标，但其结果只用于最终报告。
test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_test.json',
        metric='bbox',
        classwise=True,
        format_only=False,
        backend_args=backend_args),
    target_metric,
]

# ------------------------------ 训练计划 ------------------------------
# 共训练 24 epoch，每个 epoch 后验证一次。小数据集每轮迭代较少，因此比
# COCO 官方 12 epoch 基线适当延长；学习率在第 16、22 轮各衰减 10 倍。
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    # AMP 使用 FP16/FP32 混合精度以节省 RTX 4060 显存；动态 loss scale
    # 可减少半精度梯度下溢。accumulative_counts=4 表示累积四个小 batch。
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    accumulative_counts=4,
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    # 梯度裁剪用于限制偶发异常大梯度，提高训练稳定性。
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=24,
         milestones=[16, 22], gamma=0.1),
]

# ------------------------------ 运行时钩子 ------------------------------
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        # 不按岩石 AP 或总体 mAP 选模型，而是在满足海胆 precision>=95%
        # 的前提下选择 recall 最高的 checkpoint，与任务风险偏好保持一致。
        save_best='echinus/recall_at_precision_95',
        rule='greater',
        max_keep_ckpts=3,
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
# cudnn_benchmark=False 有利于输入尺寸变化时保持复现性；单卡训练不会实际
# 使用 NCCL，但保留配置可兼容未来 torchrun 多卡训练。
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
# seed 固定数据顺序与增强随机性；deterministic=False 默认换取更高速度，
# 可由 tools/train.py 的 --deterministic 参数覆盖为严格复现模式。
randomness = dict(seed=42, deterministic=False)
# 禁止 MMEngine 根据 GPU 数自动缩放学习率，避免迁移机器后超参数暗中变化。
auto_scale_lr = dict(enable=False, base_batch_size=8)
