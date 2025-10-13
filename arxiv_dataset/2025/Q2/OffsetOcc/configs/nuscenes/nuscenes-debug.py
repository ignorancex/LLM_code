_base_ = ['../_base_/datasets/nuscenes.py']

data_root = 'data/nuscenes-mini/'

pts_semantic_mask = 'lidarseg/v1.0-mini'
pts_panoptic_mask = 'panoptic/v1.0-mini'

train_dataloader = _base_.train_dataloader
train_dataloader.dataset.data_root = data_root
train_dataloader.batch_size = 1
train_dataloader.dataset.ann_file = 'nuscenes_infos_val.pkl'
train_dataloader.persistent_workers = False
train_dataloader.num_workers = 0
train_dataloader.dataset.data_prefix.pts_semantic_mask = pts_semantic_mask
train_dataloader.dataset.data_prefix.pts_panoptic_mask = pts_panoptic_mask

val_dataloader = _base_.val_dataloader
val_dataloader.dataset.data_root = data_root
val_dataloader.batch_size = 1
val_dataloader.persistent_workers = False
val_dataloader.num_workers = 0
val_dataloader.dataset.data_prefix.pts_semantic_mask = pts_semantic_mask
val_dataloader.dataset.data_prefix.pts_panoptic_mask = pts_panoptic_mask

test_dataloader = _base_.test_dataloader
test_dataloader.dataset.data_root = data_root
test_dataloader.batch_size = 1
test_dataloader.persistent_workers = False
test_dataloader.num_workers = 0
test_dataloader.dataset.data_prefix.pts_semantic_mask = pts_semantic_mask
test_dataloader.dataset.data_prefix.pts_panoptic_mask = pts_panoptic_mask
