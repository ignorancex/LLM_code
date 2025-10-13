_base_ = [
    '../_base_/datasets/nuscenes_2stage.py', '../_base_/default_runtime_2stage_paper.py'
]

task_name = 'offsetocc-occ3d'

experiment_run = 'objectmodule'
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

# scale reduction factor
scale_factors = (2.0, 2.0, 2.0)

img_norm_cfg = dict(
    mean=[123.675, 103.530, 116.280], std=[58.395, 57.375, 57.120])

panoptic_scene_ce_loss_weight = 0.5
panoptic_scene_l1_loss_weight = 0.02
panoptic_scene_bg_cls_weight = 0.1

panoptic_obj_bince_loss_weight = 0.125
panoptic_obj_l1_loss_weight = 0.0125
panoptic_obj_bg_cls_weight = 0.1

mask_camera_panoptic = False            # default is False

# set the validation loss
_base_.val_evaluator.extend([
    dict(type='PanopticLossMetric2Stage',
         num_classes=17,
         pc_range=point_cloud_range,
         scene_assigner_cfg=dict(
             match_costs=[
                 # dict(type='mmdet.ClassificationCost', weight=panoptic_scene_ce_loss_weight),
                 dict(type='mmdet.FocalLossCost', weight=panoptic_scene_ce_loss_weight),
                 dict(type='L2Cost', weight=panoptic_scene_l1_loss_weight),
             ]
         ),
         obj_assigner_cfg=dict(
             match_costs=[
                 # dict(type='MyCrossEntropyLossCost', use_sigmoid=True, weight=panoptic_obj_bince_loss_weight),
                 dict(type='MyBinaryFocalLossCost', alpha=0.75, gamma=2.0, weight=panoptic_obj_bince_loss_weight),
                 dict(type='L2Cost', weight=panoptic_obj_l1_loss_weight),
             ]
         ),
        loss_scene_cls_cfg=dict(type='mmdet.FocalLoss', use_sigmoid=True, loss_weight=panoptic_scene_ce_loss_weight),
        loss_scene_reg_cfg=dict(type='L2Loss',
                                reduction='mean',
                                loss_weight=panoptic_scene_l1_loss_weight),
        loss_scene_lwh_cfg=dict(type='L1Loss',
                                reduction='mean',
                                loss_weight=0.),
        loss_obj_cls_cfg=dict(type='mmdet.FocalLoss', use_sigmoid=True, alpha=0.75, gamma=2.0, loss_weight=panoptic_obj_bince_loss_weight),
        loss_obj_occ_cfg=dict(type='L2Loss',
                              # type='L2MarginLoss',
                              # margin=[0.2, 0.2, 0.2],
                              reduction='mean',
                              loss_weight=panoptic_obj_l1_loss_weight),
        bg_cls_weight=0.1,
        bg_occ_weight=0.1,
        mask_camera_panoptic=mask_camera_panoptic,
        occ_l=occ_l,
        occ_w=occ_w,
        occ_h=occ_h,
        # obj_classes_indices=list(_base_.map_from_bbox_label_to_occ_label.values())
        obj_classes_indices=list()
    )
])


# augmentations to be added before normalization
train_pipeline = _base_.train_pipeline
train_pipeline.insert(1, dict(type='PhotoMetricDistortionMultiViewImage'))

# update the train pipeline with the augmentations
_base_.train_dataloader.dataset.pipeline = train_pipeline

model = dict(
    type='OffsetOcc2ndStage',
    data_preprocessor=dict(
        type='mmdet3d.Det3DDataPreprocessor',
        **img_norm_cfg,
        bgr_to_rgb=True,
        # pad_size_divisor=32
        batch_augments = [
            dict(type='GridMask', use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        ]
        ),
    # frozen_layers=['transformer.decoder.layers.0.attentions.0.deformable_attention.sampling_offsets',
    #                'transformer.decoder.layers.1.attentions.0.deformable_attention.sampling_offsets'],
    frozen_layers=['head.fc_cls'],
    frozen_chunks=['backbone', 'img_neck', 'transformer.level_embeds', 'transformer.cams_embeds',
                   'transformer.decoder'],
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
        type='PerceptionTransformer3DObjDecoder',
        embed_dims=_dim_,
        occ_l=occ_l,
        occ_w=occ_w,
        occ_h=occ_h,
        scale_factors=scale_factors,
        num_objqueries=900,
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
                        num_levels=1,
                        dropout=0.1),  #TODO hard code it? The scale is always one
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
                                 'ffn', 'norm')
            )
        ),
        obj_decoder=dict(
            type='ObjDecoder3D',
            num_layers=4,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='VolumetricDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=1,
                        dropout=0.1),
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=_dim_,
                    feedforward_channels=_ffn_dim_,
                    ffn_drop=0.1,
                ),
                batch_first=True,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm'))
        )
    ),
    head=dict(
        type='SegmentationHead2Stage',
        embed_dims=_dim_,
        occ_l=occ_l,
        occ_w=occ_w,
        occ_h=occ_h,
        num_classes=17,
        scale_factors=scale_factors,
        mask_camera=True,
    ),
    panoptic_head=dict(
        type='PanopticHead',
        center_ablation=False,
        occ_l=occ_l,
        occ_w=occ_w,
        occ_h=occ_h,
        num_classes=17,
        pc_range=point_cloud_range,
        mask_camera=mask_camera_panoptic,
        obj_classes_indices=list(_base_.map_from_bbox_label_to_occ_label.values()),
        # loss_scene_cls_cfg=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=panoptic_scene_ce_loss_weight),
        loss_scene_cls_cfg=dict(type='mmdet.FocalLoss', use_sigmoid=True, loss_weight=panoptic_scene_ce_loss_weight),
        loss_scene_reg_cfg=dict(type='L2Loss',
                                reduction='mean',
                                loss_weight=panoptic_scene_l1_loss_weight),
        loss_scene_lwh_cfg=dict(type='L1Loss',
                                reduction='mean',
                                loss_weight=0.),
        # loss_obj_cls_cfg=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=panoptic_obj_bince_loss_weight),
        loss_obj_cls_cfg=dict(type='mmdet.FocalLoss', use_sigmoid=True, alpha=0.75, gamma=2.0, loss_weight=panoptic_obj_bince_loss_weight),
        loss_obj_occ_cfg=dict(type='L2Loss',
                              # type='L2MarginLoss',
                              # margin=[0.2, 0.2, 0.2],
                              reduction='mean',
                              loss_weight=panoptic_obj_l1_loss_weight),

        bg_cls_weight=0.1,
        bg_occ_weight=0.1,
        obj_occ_label_smoothing=0.0,
        obj_occ_voxel_center_noise=False,
        sync_avg_factor=True,
        embed_dims=_dim_,
        num_offsets=2197,    # 1331
        decode_ffn_cfg=dict(
            embed_dims=_dim_,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.,
            dropout_layer=None,
            add_identity=True,
            init_cfg=None,
            layer_scale_init_value=0.
        ),
        scene_assigner_cfg=dict(
            match_costs=[
                # dict(type='mmdet.ClassificationCost', weight=panoptic_scene_ce_loss_weight),
                dict(type='mmdet.FocalLossCost', weight=panoptic_scene_ce_loss_weight),
                dict(type='L2Cost', weight=panoptic_scene_l1_loss_weight),
            ]
        ),
        obj_assigner_cfg = dict(
            match_costs=[
                # dict(type='MyCrossEntropyLossCost', use_sigmoid=True, weight=panoptic_obj_bince_loss_weight),
                dict(type='MyBinaryFocalLossCost', alpha=0.75, gamma=2.0, weight=panoptic_obj_bince_loss_weight),
                dict(type='L2Cost', weight=panoptic_obj_l1_loss_weight),
            ]
        ),
    ),
    fusion_head=dict(
        type='FusionHead2Stage',
        embed_dims=_dim_,
        num_classes=17,
        lambda_merge=1.0,
        no_object_logit_value=100.0,
        obj_classes_indices=list(_base_.map_from_bbox_label_to_occ_label.values()),
        occ_l=occ_l,
        occ_w=occ_w,
        occ_h=occ_h,
        pc_range=point_cloud_range,
        majority_voting=True,
        max_voxel_distance=9,
        pred_point_labels=True
    )
)

