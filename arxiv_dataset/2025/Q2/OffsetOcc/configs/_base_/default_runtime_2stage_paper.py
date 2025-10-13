default_scope = 'offsetocc'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, ignore_last=False, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    visualization=dict(type='OccVisualizationHook2Stage', draw=True, interval=500)
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(   # visualize punctual loss, not running average
     window_size=1,
     by_epoch=True,
     custom_cfg=[dict(data_src='loss',
                       method_name='current')
                 ]
)

log_level = 'INFO'
load_from = None
resume = False

param_scheduler = dict(type='ExponentialLR', by_epoch=True, gamma=1.0)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        'img_neck': dict(lr_mult=1.0),
        'transformer': dict(lr_mult=1.0),
        'head': dict(lr_mult=1.0),
        'panoptic_head': dict(lr_mult=1.0),
        'fusion_head': dict(lr_mult=1.0),
    }
    ),
    clip_grad=dict(max_norm=30, norm_type=2),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


visualizer = dict(
        type='OccLocalVisualizer2Stage',
        vis_backends=[
            dict(
                type='WandbVisBackend',
                init_kwargs=dict(project='offsetocc', resume='allow', allow_val_change=True),
                watch_kwargs=dict(log='all', log_freq=100)
            ),
        ],
)

randomness = dict(seed=0)