custom_imports = dict(
    imports=['projects.PillarNet.pillarnet'], allow_failed_imports=False)

num_epoch = 80

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.16, 0.16, 5]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
class_names = ['Car', 'Pedestrian', 'Cyclist']
tasks = [dict(num_class=3, class_names=['Car', 'Pedestrian', 'Cyclist'])]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

model = dict(
    type='PillarNet',
    # pointpillars backbone
    data_preprocessor=dict(
        type='NonVoxelizeDataPreprocessor'),
    voxel_encoder=dict(
        feat_channels=[32, ],
        in_channels=0,
        point_cloud_range=point_cloud_range,
        type='Radar7PillarNeXtFeatureNet',
        voxel_size=voxel_size,
        with_distance=False,
        use_xyz=True,       # in_channels += 3
        use_rcs=True,       # in_channels += 1
        use_vr=True,        # in_channels += 1
        use_vr_comp=True,   # in_channels += 1
        use_time=True,      # in_channels += 1
        use_elevation=True, # whether ignoring z of xyz
        ),
    backbone=dict(type="PillarResNet34S", in_channels=32),
    neck=dict(
        type="RPNV2",
        layer_nums=[5, 5],
        num_filters=256,
        in_channels=[256, 128]
    ),
    bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=tasks,
        common_heads=dict(
            # reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)), # by zrx
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[0, -25.6, -3, 51.2, 25.6, 2],
            pc_range = [0, -25.6], # added by zrx
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            # code_size=9),
            code_size=7), # by zrx
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        point_cloud_range=point_cloud_range, #added by zrx
        # grid_size=[512, 512, 1],
        grid_size=[320, 320, 1], # by zrx
        voxel_size=voxel_size,
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), # by zrx
    test_cfg=dict(
        post_center_limit_range=[0, -25.6, -3, 51.2, 25.6, 2],
        max_per_img=500,
        max_pool_nms=False,
        # min_radius=[4, 12, 10, 1, 0.85, 0.175],
        min_radius=[4, 0.85, 0.175], # car, cyclist, pedestrian by zrx
        score_threshold=0.1,
        pc_range=[0, -25.6], # need to change
        out_size_factor=4,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2))

data_root = 'data/vod/radar_5frames/'
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'waymo_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
#     classes=class_names,
#     sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=6,
#         use_dim=[0, 1, 2, 3, 4],
#         backend_args=backend_args),
#     backend_args=backend_args)

# original train_pipeline of CenterFormer
# train_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=6,
#         use_dim=5,
#         norm_intensity=True,
#         backend_args=backend_args),
#     # Add this if using `MultiFrameDeformableDecoderRPN`
#     # dict(
#     #     type='LoadPointsFromMultiSweeps',
#     #     sweeps_num=9,
#     #     load_dim=6,
#     #     use_dim=[0, 1, 2, 3, 4],
#     #     pad_empty_sweeps=True,
#     #     remove_close=True),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(type='ObjectSample', db_sampler=db_sampler),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-0.78539816, 0.78539816],
#         scale_ratio_range=[0.95, 1.05],
#         translation_std=[0.5, 0.5, 0]),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectNameFilter', classes=class_names),
#     dict(type='PointShuffle'),
#     dict(
#         type='Pack3DDetInputs',
#         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]

# kitti train_pipeline from zrx
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalScale', scale_ratio_range=[0.95, 1.05]),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        point_cloud_range=point_cloud_range,
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=point_cloud_range,
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=['points', 'gt_labels_3d','gt_bboxes_3d'],
        type='Pack3DDetInputs'),
]

# original test_pipeline of CenterFormer
# test_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=6,
#         use_dim=5,
#         norm_intensity=True,
#         backend_args=backend_args),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D'),
#             dict(
#                 type='PointsRangeFilter', point_cloud_range=point_cloud_range)
#         ]),
#     dict(
#         type='Pack3DDetInputs',
#         keys=['points'],
#         meta_keys=['box_type_3d', 'sample_idx', 'context_name', 'timestamp'])
# ]

# kitti test_pipeline from zrx
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(type='RandomFlip3D'),
    dict(
        type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(keys=['points'], type='Pack3DDetInputs'),
]

dataset_type = 'KittiDataset'

# original train_dataloader of CenterFormer
# train_dataloader = dict(
#     batch_size=4,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='waymo_infos_train.pkl',
#         data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
#         pipeline=train_pipeline,
#         modality=input_modality,
#         test_mode=False,
#         metainfo=metainfo,
#         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#         box_type_3d='LiDAR',
#         # load one frame every five frames
#         load_interval=5,
#         backend_args=backend_args))

# kitti train_dataloader from zrx
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/velodyne'),
            data_root='data/vod/radar_5frames/',
            metainfo=metainfo,
            modality=input_modality,
            pipeline=train_pipeline,
            test_mode=False,
            type=dataset_type),
        times=1,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

# original val_dataloader and test_dataloader of CenterFormer
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
#         ann_file='waymo_infos_val.pkl',
#         pipeline=test_pipeline,
#         modality=input_modality,
#         test_mode=True,
#         metainfo=metainfo,
#         box_type_3d='LiDAR',
#         backend_args=backend_args))
# test_dataloader = val_dataloader

# kitti val_dataloader and test_dataloader from zrx
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        # ann_file='kitti_infos_val.pkl',
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne'),
        data_root=data_root,
        metainfo=metainfo,
        modality=input_modality,
        pipeline=test_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

# original val_evaluator and test_evaluator of CenterFormer
# val_evaluator = dict(
#     type='WaymoMetric', waymo_bin_file='./data/waymo/waymo_format/gt.bin')
# test_evaluator = val_evaluator

# kitti val_evaluator and test_evaluator from zrx
test_evaluator = dict(
    type='KittiMetricv2',
    dataset_name='VoD',
    ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
    metric='bbox',
    format_only=True,
    submission_prefix='results/kitti-3class/kitti_results') # for submission
val_evaluator = dict(
    ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    dataset_name='VoD',
    type='KittiMetricv2')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# optimizer
lr = 0.003
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
# learning rate
step_ratio_up = 0.4   # ratio of the first phase length to the maximum training epochs
param_scheduler = [
    dict(
        type='MultiStepLR',   # call for the MultiStepLR embedded in MMEngine
        milestones=[35, 45],  # learning rate step decay when 35th and 45th epoch
        gamma=0.1,            # Factor of learning rate decay
        by_epoch=True,        # decay activated by epoch or iteration
    ),
    dict(
        # eta_min=0.85 / 0.95,           # if calling for eta_min, the momentum will change to this fixed value during the process
        eta_min_ratio=0.85 / 0.95,       # if calling for eta_min_ratio, the momentum will change to (base_value * eta_min_ratio) during the process
        begin=0,                         # beginning of the process
        end=num_epoch * step_ratio_up,   # end of the process
        T_max=num_epoch * step_ratio_up, # the length of process
        by_epoch=True,
        convert_to_iter_based=True,
        type='CosineAnnealingMomentum'),
    dict(
        # eta_min=1,                                  # if calling for eta_min, the momentum will change to this fixed value during the process
        eta_min_ratio=1,                              # if calling for eta_min_ratio, the momentum will change to (base_value * eta_min_ratio) during the process
        begin=num_epoch * step_ratio_up,              # beginning of the process
        end=num_epoch,                                # end of the process
        T_max=num_epoch - num_epoch * step_ratio_up,  # the length of process
        by_epoch=True,
        convert_to_iter_based=True,
        type='CosineAnnealingMomentum'),
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=num_epoch, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False


# define the visualizer
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),  # active Tensorboard recorder
        # dict(type='WandbVisBackend'),        # active Wandb recorder
    ])