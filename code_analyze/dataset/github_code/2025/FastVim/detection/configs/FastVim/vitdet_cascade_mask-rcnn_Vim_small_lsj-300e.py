_base_ = [
    "../_base_/models/cascade-mask-rcnn_r50_fpn.py",
    "./lsj-100e_coco-instance.py",
]

custom_imports = dict(imports=["vitdet"])

norm_cfg = dict(type="LN2d", requires_grad=True)
image_size = (1024, 1024)
batch_augments = [dict(type="BatchFixedSizePad", size=image_size, pad_mask=True)]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type="MM_ViM",
        pretrained=None,
        img_size=(1024, 1024),
        drop_path_rate=0.1,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=384,
        depth=24,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_cls_token=True,
        use_middle_cls_token=True,
        out_indices=[23],
    ),
    neck=dict(
        _delete_=True,
        type="SimpleFPN",
        backbone_channel=384,
        in_channels=[96, 192, 384, 384],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg,
    ),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        type="CascadeRoIHead",
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                conv_out_channels=256,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                conv_out_channels=256,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                conv_out_channels=256,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
        ],
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type="FCNMaskHead",
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            norm_cfg=norm_cfg,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ),
)

custom_hooks = [dict(type="Fp16CompresssionHook")]
optim_wrapper = dict(type="OptimWrapper", clip_grad=dict(max_norm=35, norm_type=2))
train_dataloader = dict(batch_size=2)
