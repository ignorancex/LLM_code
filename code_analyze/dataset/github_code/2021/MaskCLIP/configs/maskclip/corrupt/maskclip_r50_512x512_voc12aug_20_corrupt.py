_base_ = '../maskclip_r50_512x512_voc12aug_20.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            # possible choices for Corrupt `name`: gaussian_noise, shot_noise, impulse_noise
            #   speckle_noise, gaussian_blur, defocus_blur, spatter, jpeg_compression
            # possible choices for Corrupt `level`: 1, 2, 3, 4, 5
            dict(type='Corrupt', name='speckle_noise', level=1),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    test=dict(
        pipeline=test_pipeline
    ),
)