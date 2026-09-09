"""Self-contained inference config for the frozen echinus/rock detector.

This intentionally contains no dataset paths, evaluators, custom metrics, or
training hooks from image_process_ResNet50.  It is the deployment-side model
contract used by ObjectNav.
"""

classes = ("echinus", "rock")
metainfo = dict(
    classes=classes,
    palette=[(255, 64, 64), (80, 180, 255)],
)

model = dict(
    type="GFL",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        # The complete trained state comes from the local checkpoint. Avoid a
        # deployment-time download of torchvision initialization weights.
        init_cfg=None,
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
    ),
    bbox_head=dict(
        type="GFLHead",
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        loss_cls=dict(
            type="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_dfl=dict(type="DistributionFocalLoss", loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        # Keep candidates here; semantic_detector.py applies the calibrated
        # echinus threshold and the independent rock threshold afterward.
        score_thr=0.001,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(960, 640), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

# DetInferencer reads its preprocessing pipeline and class metadata here. No
# annotation file or image directory is needed for direct inferencer calls.
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root="",
        metainfo=metainfo,
        ann_file="",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)

default_scope = "mmdet"

