# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# learning policy
lr_config = dict(policy='step', step=4, gamma=0.5, min_lr=1e-9)
runner = dict(type='EpochRunner', max_epochs=6)