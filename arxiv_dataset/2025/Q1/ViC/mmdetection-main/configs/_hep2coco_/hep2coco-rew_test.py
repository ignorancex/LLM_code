_base_ = [
    '../_base_/_hep2coco_models_/hep-retinanet_vheatk-tiny_fpn.py',
    '../_base_/_hep2coco_datasets_/hep2coco-rew_detection.py',
    '../_base_/_hep2coco_schedules_/schedule_1x_rtf.py', '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        mmt_base=0.7,
        mmt_min=0.2,
        mmt_max=1.2,
    ))

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

dataset_type = 'Hep2CocoDataset'
data_root = 'data/HEP2COCO/Nm_rew_test/'
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromHEP', backend_args=backend_args,
         bg_version='black_randn_seed', snr_db=10.0),
    dict(type='HEPLoadAnnotations', with_bbox=True, with_mmt=True),
    dict(type='Resize', scale=(960, 480), keep_ratio=True),
    dict(
        type='HEPPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_ann_files = [
    'Nm_rew__b00000001__e00100000.json',
    # 'Nm_rew__b00100001__e00200000.json',
    # 'Nm_rew__b00200001__e00300000.json',
    # 'Nm_rew__b00300001__e00400000.json',
    # 'Nm_rew__b00400001__e00500000.json',
    # 'Nm_rew__b00500001__e00600000.json',
    # 'Nm_rew__b00600001__e00700000.json',
    # 'Nm_rew__b00700001__e00800000.json',
    # 'Nm_rew__b00800001__e00900000.json',
    # 'Nm_rew__b00900001__e01000000.json',
    'Nm_rew__b01000001__e01100000.json',
    'Nm_rew__b01100001__e01200000.json',
    'Nm_rew__b01200001__e01300000.json',
    'Nm_rew__b01300001__e01400000.json',
    'Nm_rew__b01400001__e01500000.json',
    'Nm_rew__b01500001__e01600000.json',
    'Nm_rew__b01600001__e01700000.json',
    'Nm_rew__b01700001__e01769709.json',
]
test_dataset_base = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='annotations/instances_val2017.json',
    # data_prefix=dict(img='val2017/'),
    ann_file='',
    data_prefix=dict(img='./'),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args)
test_datasets = []

for test_i in range(0, len(test_ann_files)):
    temp = test_dataset_base.copy()
    temp['ann_file'] = test_ann_files[test_i]
    test_datasets.append(temp)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/dataset_wrappers.py
        type='ConcatDataset',
        datasets=test_datasets))

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + test_ann_files[0],
    outfile_prefix='./work_dirs/coco_detection/test')
