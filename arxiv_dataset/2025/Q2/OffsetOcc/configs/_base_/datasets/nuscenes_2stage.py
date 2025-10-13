dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

occ_class_names = [
    'ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free'
]

lidarseg_label2cat = {
    0: 'ignore', 1: 'barrier', 2: 'bicycle', 3: 'bus', 4: 'car', 5: 'construction_vehicle',
    6: 'motorcycle', 7: 'pedestrian', 8: 'traffic_cone', 9: 'trailer', 10: 'truck', 11: 'driveable_surface',
    12: 'other_flat', 13: 'sidewalk', 14: 'terrain', 15: 'manmade', 16: 'vegetation'
}

lidarseg_fine2coarse_mapping = dict([(1, 0), (5, 0), (7, 0), (8, 0), (10, 0), (11, 0), (13, 0),
                                    (19, 0), (20, 0), (0, 0), (29, 0), (31, 0), (9, 1), (14, 2),
                                    (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
                                    (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
                                    (26, 13), (27, 14), (28, 15), (30, 16)])

map_from_bbox_label_to_occ_label = {
    0: 4,   # car
    1: 10,  # truck
    2: 9,   # trailer
    3: 3,   # bus
    4: 5,   # construction_vehicle
    5: 2,   # bicycle
    6: 6,   # motorcycle
    7: 7,   # pedestrian
    8: 8,   # traffic_cone
    9: 1    # barrier
}

# occupancy map size
occ_l = 200
occ_w = 200
occ_h = 16

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

class_ce_loss_weights = [10.0, 10.0, 10.0,
                         10.0, 10.0, 10.0,
                         10.0, 10.0, 10.0,
                         10.0, 10.0, 10.0,
                         10.0, 10.0, 10.0,
                         10.0, 10.0, 1.0]

occ_class_palette = [
        [0, 0, 0],          # 0 ignore
        [255, 140, 0],      # 1 barrier                     Darkorange
        [255, 61, 99],      # 2 bicycle                     Red
        [255, 247, 0],      # 3 bus                         Yellow
        [255, 158, 0],      # 4 car                         Orange
        [233, 150, 70],     # 5 construction_vehicle        Darksalmon
        [112, 128, 144],    # 6 motorcycle                  Slategrey
        [0, 0, 230],        # 7 pedestrian                  Blue
        [233, 150, 70],     # 8 traffic_cone                Darksalmon
        [0, 0, 0],          # 9 trailer                     Black
        [47, 79, 79],       # 10 truck                      Darkslategrey
        [0, 207, 191],      # 11 driveable_surface          Turquoise
        [255, 0, 0],        # 12 other_flat                 Red
        [75, 0, 75],        # 13 sidewalk                   Purple
        [165, 42, 42],      # 14 terrain                    Brown
        [128, 128, 128],    # 15 manmade                    Grey
        [0, 175, 0],        # 16 vegetation                 Green
        [0, 0, 0]           # 17 free
]

camera_display_grid = [[2, 0, 1], [5, 3, 4]]

metainfo = dict(classes=class_names, occ_class_names=occ_class_names, occ_palette=occ_class_palette, empty_class=17,
                camera_display_grid=camera_display_grid, label2cat=lidarseg_label2cat, ignore_index=0)

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval',
    pts_panoptic_mask='panoptic/v1.0-trainval',
    sweeps='',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    occ='occupancy')

backend_args = None

train_pipeline = [
    dict(
        type='mmdet3d.LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=6,
        backend_args=backend_args),
    dict(type='mmdet3d.LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadLidar2Img'),
    dict(type='Lidar2EgoBbox'),
    dict(type='LoadOccGTFromFile'),
    dict(type='MapBboxLabelsToOcc', map_to_occ_label=map_from_bbox_label_to_occ_label),
    dict(type='LoadOccGTFromFileAux'),
    dict(type='PackOccInputs', keys=['img', 'gt_occ_map', 'gt_occ_map_mask_camera', 'gt_panoptic_occ_map',
                                     'gt_bboxes_3d','gt_labels_3d', 'gt_occ_map_visible_obj_occ_grid',
                                     'gt_occ_map_visible_obj_occ_grid_start_idx', 'gt_occ_map_visible_obj_idx']),
]

# if eval lidarseg uncomment the following transforms in the eval pipeline
eval_pipeline = [
    dict(
        type='mmdet3d.LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=6,
        backend_args=backend_args),
    dict(type='mmdet3d.LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=3),
    dict(type='MyLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_seg_3d=True,
         with_panoptic_3d=True, with_attr_label=False, seg_3d_dtype='np.uint8', dataset_type='nuscenes'),
    dict(type='mmdet3d.LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='SegFine2CoarseNuscMapping', mapping=lidarseg_fine2coarse_mapping),
    dict(type='PanoSegFine2CoarseNuscMapping', mapping=lidarseg_fine2coarse_mapping),
    dict(type='LoadLidar2Img'),
    dict(type='Lidar2EgoBbox'),
    dict(type='PointCloudLidar2Ego'),
    dict(type='LoadOccGTFromFile'),
    dict(type='MapBboxLabelsToOcc', map_to_occ_label=map_from_bbox_label_to_occ_label),
    dict(type='LoadOccGTFromFileAux'),
    dict(type='PackOccInputs', keys=['img', 'points', 'gt_occ_map', 'gt_occ_map_mask_camera', 'gt_panoptic_occ_map',
                                     'gt_bboxes_3d','gt_labels_3d', 'gt_occ_map_visible_obj_occ_grid',
                                     'gt_occ_map_visible_obj_occ_grid_start_idx', 'gt_occ_map_visible_obj_idx',
                                     'pts_semantic_mask', 'pts_instance_mask', 'gt_occ_map_visible_obj_masks']),
]

test_pipeline = eval_pipeline


train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        load_type='frame_based',
        metainfo=metainfo,
        modality=input_modality,
        filter_empty_gt=False,
        with_velocity=False,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        load_type='frame_based',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        filter_empty_gt=False,
        with_velocity=False,
        load_eval_anns=True,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], use_camera_mask=True),
    dict(type='SegMetric', prefix='lidar_seg'),
    dict(type='PanopticSegMetric',  thing_class_inds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    stuff_class_inds=[11, 12, 13, 14, 15, 16], min_num_points=20, id_offset=100000000,
                                    ignore_index=0, prefix='lidar_panoseg'),
]

test_evaluator = val_evaluator
