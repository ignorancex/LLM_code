# optimizer
optimizer = dict(type='Adam', lr=5e-5, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# learning policy
lr_config = dict(by_epoch=False, policy='Hood', decay_rate=5e-1, decay_steps=50000, step_start=15000)
runner = dict(type='IterRunner', max_epochs=None, max_iters=150000)