_base_ = ["../../../configs/_base_/default_runtime.py"]
custom_imports = dict(imports=["projects.TaDe.tade"])

dataset_type = "BEVDataset"
data_root = "data/nuscenes"
train_ann_file = "nuscenes_infos_train.pkl"
val_ann_file = "nuscenes_infos_val.pkl"
camera_types = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

train_pipeline = [
    dict(
        type="LoadData",
        camera_types=camera_types,
        load_image=True,
        load_bev_map=True,
        test_mode=False,
    ),
]
val_pipeline = [
    dict(
        type="LoadData",
        camera_types=camera_types,
        load_image=True,
        load_bev_map=True,
        test_mode=True,
    ),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=10,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        data_root=data_root,
        data_prefix=dict(),
        pipeline=train_pipeline,
        test_mode=False,
        indices=None,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        data_root=data_root,
        data_prefix=dict(),
        pipeline=val_pipeline,
        test_mode=True,
        indices=None,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="BEVMetric",
    thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
)
test_evaluator = val_evaluator

model = dict(
    type="BEV",
    backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=dict(
        type="mmdet.FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
    ),
    class_weights=[
        2,  # drivable_area (0.44679)
        7,  # ped_crossing (0.02407)
        3,  # walkway (0.14491)
        6,  # carpark_area (0.02994)
        7,  # car (0.02086)
        15,  # truck (0.00477)
        23,  # trailer (0.00189)
        25,  # bus (0.00156)
        35,  # construction_vehicle (0.00084)
        91,  # bicycle (0.00012)
        73,  # motorcycle (0.00019)
        29,  # pedestrian (0.00119)
        57,  # traffic_cone (0.00031)
        24,  # barrier (0.00176)
    ],
    # class_weights_argoverse=[
    #     1.7,  # drivable_area
    #     5.2,  # vehicle
    #     22.0,  # pedestrian
    #     9.6,  # large_vehicle
    #     20.3,  # bicycle
    #     9.6,  # bus
    #     7.0,  # trailer
    #     27.5,  # motorcycle
    # ],
    vis_list=["data/nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg"],
    init_cfg=[dict(type="Xavier", layer="Conv2d")],
)

max_epochs = 50
lr = 0.001
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="Adam", lr=lr, weight_decay=0.01), clip_grad=None)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.5, begin=0, end=5, by_epoch=True),
    dict(
        type="CosineAnnealingLR",
        eta_min=lr * 1e-2,
        T_max=max_epochs - 5,
        begin=5,
        end=max_epochs,
        by_epoch=True,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()
env_cfg = dict(
    cudnn_benchmark=True,  # 是否启用 cudnn benchmark
    mp_cfg=dict(  # 多进程设置
        mp_start_method="fork",  # 使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能存在隐患。请参考 https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0,
    ),  # 关闭 opencv 的多线程以避免系统超负荷
    dist_cfg=dict(backend="nccl"),  # 分布式相关设置
)

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=max_epochs))
custom_hooks = [dict(type="EpochHook")]
find_unused_parameters = True
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)
