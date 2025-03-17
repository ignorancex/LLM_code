# Copyright (c) OpenMMLab. All rights reserved.
# dataset settings

# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'

dataset_type = 'CocoDatasetCustomClassification'
data_root = 'data/GBCU/'
# data_root = 'data/GBCU-Shared/'
BATCH_SIZE=2

# classes = ('gb',)
classes = ('benign','malignant')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         scale_factor=1.,
        #  img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=1,
    train=dict(type=dataset_type,
               ann_file=data_root + 'new_splits/gb_train_1.json',
               img_prefix=data_root + 'imgs/',
               classes=classes,
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             ann_file=data_root + 'new_splits/gb_test_1.json',
             img_prefix=data_root + 'imgs/',
             classes=classes,
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              ann_file=data_root + 'new_splits/gb_test_1.json',
              img_prefix=data_root + 'imgs/',
              classes=classes,
              pipeline=test_pipeline))


# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(type=dataset_type,
#                ann_file=data_root + 'latest_train.json',
#                img_prefix=data_root + 'imgs/',
#                classes=classes,
#                pipeline=train_pipeline),
#     val=dict(type=dataset_type,
#              ann_file=data_root + 'latest_test.json',
#              img_prefix=data_root + 'imgs/',
#              classes=classes,
#              pipeline=test_pipeline),
#     test=dict(type=dataset_type,
#               ann_file=data_root + 'latest_test.json',
#               img_prefix=data_root + 'imgs/',
#               classes=classes,
#               pipeline=test_pipeline))

evaluation = dict(metric=['bbox'])
