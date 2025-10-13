_base_ = [
    '../_base_/datasets/nuscenes.py', '../_base_/default_runtime_paper.py'
]

task_name = 'offsetocc-occ3d'

experiment_run = 'baseline'
work_dir = './experiments/' + task_name + '-' + experiment_run

# experiment name on wandb
visualizer = _base_.visualizer
visualizer.vis_backends[0].init_kwargs['name'] = task_name + '-' + experiment_run

# change OccVisualizationHook interval to 20
default_hooks = _base_.default_hooks
default_hooks['visualization'].interval = 20

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

_dim_ = 128
_pos_dim_ = _dim_//2
_ffn_dim_ = 128
_num_levels_ = 4

# output occupancy map size
occ_l = 200
occ_w = 200
occ_h = 16

_base_.val_evaluator.extend([
    dict(type='SSCLossMetric', free_class_index=17)
])

# scale reduction factor
scale_factors = (2.0, 2.0, 2.0)

img_norm_cfg = dict(
    mean=[123.675, 103.530, 116.280], std=[58.395, 57.375, 57.120])

# augmentations to be added before normalization
train_pipeline = _base_.train_pipeline
train_pipeline.insert(1, dict(type='PhotoMetricDistortionMultiViewImage'))

# update the train pipeline with the augmentations
_base_.train_dataloader.dataset.pipeline = train_pipeline

model = dict(
    type='OffsetOcc',
    data_preprocessor=dict(
        type='mmdet3d.Det3DDataPreprocessor',
        **img_norm_cfg,
        bgr_to_rgb=True,
        # pad_size_divisor=32
        batch_augments = [
            dict(type='GridMask', use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        ]
    ),
    backbone=dict(
        type='mmpretrain.ResNet',
        depth=101,
        deep_stem=False,
        num_stages=4,
        out_indices=(0, 1, 2, 3),              # also the lower level 0 can be returned
        frozen_stages=-1,
        with_cp=True,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    transformer=dict(
        type='PerceptionTransformer',
        embed_dims=_dim_,
        occ_l=occ_l,
        occ_w=occ_w,
        occ_h=occ_h,
        scale_factors=scale_factors,
        decoder=dict(
            type='OffsetOccEncoder',
            num_layers=4,
            pc_range=point_cloud_range,
            return_intermediate=False,
            reference_system='ego',
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='SpatialCrossAttention',
                        deformable_attention=dict(
                            type='MultiScaleDeformableAttention2D',
                            embed_dims=_dim_,
                            num_levels=_num_levels_,
                            num_heads=8),
                        embed_dims=_dim_,
                        num_cams=6,
                        dropout=0.1,
                    ),
                    dict(
                        type='VolumetricDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=1),  #TODO hard code it? The scale is always one
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=_dim_,
                    feedforward_channels=_ffn_dim_,
                    ffn_drop=0.1,
                ),
                batch_first=True,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                # operation_order=('cross_attn', 'norm',
                                 'ffn', 'norm')))),
    head=dict(
        type='SegmentationHead',
        embed_dims=_dim_,
        occ_l=occ_l,
        occ_w=occ_w,
        occ_h=occ_h,
        num_classes=17,
        scale_factors=scale_factors,
        mask_camera=False,
        loss_cls=dict(type='SSCLoss', free_class_index=17),
    )
)

