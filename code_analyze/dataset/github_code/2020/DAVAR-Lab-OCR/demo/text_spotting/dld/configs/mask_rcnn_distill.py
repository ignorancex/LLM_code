"""
#################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_distill.py
# Abstract       :    Model settings for mask-rcnn-based text spotter distillation

# Current Version:    1.0.0
# Date           :    2020-07-07
#################################################################################################
"""
character = '../../datalist/character_list.txt'
batch_max_length = 32
type='SPOTTER'
student = dict(
    type='KDTwoStageEndToEnd',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # rcg
    rcg_roi_extractor=dict(
        type='MaskedRoIExtractor',
        in_channels=256,
        out_channels=256,
        output_size=(8, 32),
        featmap_strides=[4],),
    rcg_backbone=dict(
        type='LightCRNN',
        in_channels=256,
        out_channels=256
    ),
    rcg_sequence_module=dict(
        type='CascadeRNN',
        rnn_modules=[
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True,),
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True,),]),
    rcg_sequence_head=dict(
        type='AttentionHead',
        input_size=256,
        hidden_size=256,
        batch_max_length=batch_max_length,
        converter=dict(
            type='AttnLabelConverter',
            character=character,
            use_cha_eos=True,),
        loss_att=dict(          
            type='StandardCrossEntropyLoss',
            ignore_index=0,
            reduction='mean',
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        # rcg
        keep_dim=False,
        sequence=(),
        # det
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),),
    test_cfg=dict(
        # rcg
        keep_dim=False,
        sequence=dict(),
        batch_max_length=batch_max_length,
        # det
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100,
            mask_thr_binary=0.5),
        postprocess=dict(
            type='PostMaskRCNNSpot'
        )),
)
teacher = dict(
    type='KDTwoStageEndToEnd',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # rcg
    rcg_roi_extractor=dict(
        type='MaskedRoIExtractor',
        in_channels=256,
        out_channels=256,
        output_size=(8, 32),
        featmap_strides=[4],),
    rcg_backbone=dict(
        type='LightCRNN',
        in_channels=256,
        out_channels=256
    ),
    rcg_sequence_module=dict(
        type='CascadeRNN',
        rnn_modules=[
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True,),
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True,),]),
    rcg_sequence_head=dict(
        type='AttentionHead',
        input_size=256,
        hidden_size=256,
        batch_max_length=batch_max_length,
        converter=dict(
            type='AttnLabelConverter',
            character=character,
            use_cha_eos=True,),
        loss_att=dict(          
            type='StandardCrossEntropyLoss',
            ignore_index=0,
            reduction='mean',
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        # rcg
        keep_dim=False,
        sequence=(),
        # det
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),),
    test_cfg=dict(
        # rcg
        keep_dim=False,
        sequence=dict(),
        batch_max_length=batch_max_length,
        # det
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100,
            mask_thr_binary=0.5),
        postprocess=dict(
            type='PostMaskRCNNSpot'
        )),
)
teacher['pretrained'] = '/path/to/mask_rcnn_res50_multi_scale-8b67df8a.pth'
student['pretrained'] = '/path/to/mask_rcnn_res50_multi_scale-8b67df8a.pth'

scale_factor = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

model = dict(
    type='SpotResolutionDistillation',
    pretrained=None,
    student=student,
    teacher=teacher,
    policy=dict(
        type='ResolutionSelector',
        temperature=5.0,
        temperature_decay=0.965,
        scale_factor=scale_factor,
        out_channels=256,
    ),
    kd_cfg=dict(
        scale_factor=scale_factor,
        penalty_coef=0.1
    )
)

train_cfg = None
test_cfg = None

# Training dataset load type
dataset_type = 'DavarMultiDataset'

# File prefix path of the traning dataset
img_prefixes = [
    '/path/to/ICDAR2013-Focused-Scene-Text/',
    '/path/to/ICDAR2015/',
    '/path/to/ICDAR2017_MLT/',
    '/path/to/Total-Text/',
]

# Dataset Name
ann_files = [
    '/path/to/datalist/icdar2013_train_datalist.json',
    '/path/to/datalist/icdar2015_train_datalist.json',
    '/path/to/datalist/icdar2017_trainval_datalist.json',
    '/path/to/datalist/total_text_train_datalist.json',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,            # Bounding Rect
         with_poly_mask=True,       # Mask
         with_poly_bbox=True,       # bouding poly
         with_label=True,           # Bboxes' labels
         with_care=True,            # Ignore or not
         with_text=True,            # Transcription
         with_cbbox=False,          # Character bounding
         text_profile=dict(text_max_length=batch_max_length, sensitive='same', filtered=False)
    ),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DavarRandomCrop'),
    dict(type='RandomRotate', angles=[-15, 15], borderValue=(0, 0, 0)),
    dict(type='DavarResize', img_scale=[(896, 1600)], multiscale_mode='value', keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_texts', 'gt_masks']),
]
test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 3000),
        flip=False,
        transforms=[
            dict(type='DavarResize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='DavarCollect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    sampler=dict(
        type='DistBatchBalancedSampler',  # BatchBalancedSampler and DistBatchBalancedSampler
        mode=-1,
        # model 0:  Balance in batch, calculate the epoch according to the first iterative data set
        # model 1:  Balance in batch, calculate the epoch according to the last iterative data set
        # model 2:  Balance in batch, record unused data
        # model -1: Each dataset is directly connected and shuffled
    ),
    train=dict(
        type=dataset_type,
        batch_ratios=['0.25', '0.25', '0.25', '0.25'],
        dataset=dict(
            type='TextSpotDataset',
            ann_file=ann_files,
            img_prefix=img_prefixes,
            test_mode=False,
            pipeline=train_pipeline)
    ),
    val=dict(
        type='TextSpotDataset',
        ann_file='/path/to/datalist/total_text_test_datalist.json',
        img_prefix='/path/to/Total-Text/',
        pipeline=test_pipeline),
    test=dict(
        type='TextSpotDataset',
        ann_file='/path/to/datalist/total_text_test_datalist.json',
        img_prefix='/path/to/Total-Text/',
        pipeline=test_pipeline))
# optimizer
find_unused_parameters = True
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    step=[30, 40]
)
runner = dict(type='DistillRunner', max_epochs=50)
checkpoint_config = dict(type='DavarCheckpointHook', interval=10, filename_tmpl='checkpoint/mask_rcnn_res50_distill_epoch_{}.pth')

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/path/to/workspace/log/'
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    model_type=type,
    type='DavarDistEvalHook',
    interval=10,
    eval_func_params=dict(
       # SPECIAL_CHARACTERS='[]+-#$()@=_!?,:;/.%&'\">*|<`{~}^\ ',
       IOU_CONSTRAINT=0.5,
       AREA_PRECISION_CONSTRAINT=0.5,
       WORD_SPOTTING=False
    ),
    by_epoch=True,
    eval_mode='general',
    # eval_mode='lightweight',
    save_best='hmean',
    rule='greater',
)