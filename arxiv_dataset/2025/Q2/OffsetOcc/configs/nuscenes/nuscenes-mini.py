_base_ = ['../_base_/datasets/nuscenes.py']

data_root = 'data/nuscenes-mini/'

pts_semantic_mask = 'lidarseg/v1.0-mini'
pts_panoptic_mask = 'panoptic/v1.0-mini'

train_dataloader = _base_.train_dataloader
train_dataloader.dataset.data_root = data_root
train_dataloader.dataset.data_prefix.pts_semantic_mask = pts_semantic_mask
train_dataloader.dataset.data_prefix.pts_panoptic_mask = pts_panoptic_mask

val_dataloader = _base_.val_dataloader
val_dataloader.dataset.data_root = data_root
val_dataloader.dataset.data_prefix.pts_semantic_mask = pts_semantic_mask
val_dataloader.dataset.data_prefix.pts_panoptic_mask = pts_panoptic_mask

test_dataloader = _base_.test_dataloader
test_dataloader.dataset.data_root = data_root
test_dataloader.dataset.data_prefix.pts_semantic_mask = pts_semantic_mask
test_dataloader.dataset.data_prefix.pts_panoptic_mask = pts_panoptic_mask