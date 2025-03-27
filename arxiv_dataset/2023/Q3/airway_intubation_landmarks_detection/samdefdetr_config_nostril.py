_base_ = '/media/ren3/disk2_hdd1/tianhang/DNmmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py'

model = dict(
    type='DeformableDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=100,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type='SAMDeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='SAMDeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='SAMDeformableDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=2048,
                            num_heads=8)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)))

dataset_type = 'CocoDataset'
classes = ['nostril']
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/media/ren3/disk2_hdd1/tianhang/dataset/BioID/train/train.json',
        img_prefix='/media/ren3/disk2_hdd1/tianhang/dataset/BioID/train/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/media/ren3/disk2_hdd1/tianhang/dataset/BioID/val/val.json',
        img_prefix='/media/ren3/disk2_hdd1/tianhang/dataset/BioID/val/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/media/ren3/disk2_hdd1/tianhang/dataset/BioID/test/test.json',
        img_prefix='/media/ren3/disk2_hdd1/tianhang/dataset/BioID/test/'))

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
#load_from = '/media/ren3/disk2_hdd1/tianhang/checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
runner = dict(type='EpochBasedRunner', max_epochs=24)
#resume_from = '/media/ren3/disk2_hdd1/tianhang/work_dirs/B_samdefdetr_config/epoch_5.pth'