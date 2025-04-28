import os.path

from mmdet.apis import set_random_seed

_base_ = ["../../mmdetection/configs/retinanet/retinanet_r101_fpn_2x_coco.py"]

# set seed
seed = 42
set_random_seed(seed, deterministic=False)

# sample type
sample_type = "smartphone_sample"

experiment_name = f"retinanet_{sample_type}_with_augmentation"

# dataset settings
dataset_home = "/mnt/Enterprise/safal/AI_assisted_microscopy_system/cysts_dataset_all"
dataset_type = "CocoDataset"
data_root = os.path.join(dataset_home, sample_type)
classes = ("Crypto", "Giardia")


img_norm_cfg = dict(  # Image normalization config to normalize the input images
    mean=[
        118.9318,
        117.8760,
        118.8063,
    ],  # Mean values used to pre-training the pre-trained backbone models
    std=[
        10.9386,
        10.9824,
        11.5052,
    ],  # Standard variance used to pre-training the pre-trained backbone models
    to_rgb=True,
)

# train pipeline
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
    ),
    dict(
        type="Resize",
        img_scale=[
            (1333, 640),
            (1333, 672),
            (1333, 704),
            (1333, 736),
            (1333, 768),
            (1333, 800),
        ],
        multiscale_mode="value",
        keep_ratio=True,
    ),  # whether to keep the ratio between height and width.
    dict(
        type="RandomFlip",  # Augmentation pipeline that flip the images and their annotations
        flip_ratio=0.5,
    ),  # The ratio or probability to flip
    dict(
        type="RandomAffine",
        max_rotate_degree=15,
        max_translate_ratio=0.2,
        scaling_ratio_range=(1.0, 1.0),
        max_shear_degree=0,
        border_val=(0, 0, 0),
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(
        type="Pad", size_divisor=32  # Padding config
    ),  # The number the padded images should be divisible
    dict(
        type="DefaultFormatBundle"
    ),  # Default format bundle to gather data in the pipeline
    dict(
        type="Collect",  # Pipeline that decides which keys in the data should be passed to the detector
        keys=["img", "gt_bboxes", "gt_labels"],
    ),
]

# test pipeline
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=os.path.join(
                data_root, "fold_5", "smartphone_sample_coco_annos_train.json"
            ),
            img_prefix=os.path.join(data_root, "train"),
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(
            data_root, "fold_5", "smartphone_sample_coco_annos_val.json"
        ),
        img_prefix=os.path.join(data_root, "train"),
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(
            dataset_home,
            "smartphone_sample_test",
            "smartphone_sample_test_coco_annos.json",
        ),
        img_prefix=os.path.join(data_root, "test"),
    ),
)

# Model
# change the number of classes in roi head to match the dataset
model = dict(
    bbox_head=dict(
        num_classes=2,
    ),
)

# checkpointing and logging
checkpoint_config = dict(interval=1, max_keep_ckpts=2)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type="TensorboardLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="mmdetection_cysts",
                group=experiment_name,
                name=f"{sample_type}_retinanet_fold_5",
            ),
        ),
    ],
)

resume_from = None
auto_resume = True

work_dir = f"/mnt/Enterprise/safal/AI_assisted_microscopy_system/outputs/{sample_type}/retinanet_with_aug/fold_5"

runner = dict(type="EpochBasedRunner", max_epochs=20)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

optimizer = dict(type="AdamW", lr=3e-4, weight_decay=0.001, _delete_=True)

workflow = [("train", 1), ("val", 1)]
