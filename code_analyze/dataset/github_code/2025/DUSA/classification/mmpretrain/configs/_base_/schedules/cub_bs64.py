# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.optim import CosineAnnealingLR, LinearLR
from torch.optim import SGD

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # fully learning rate scheduler
    dict(
        type=CosineAnnealingLR,
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=64)
