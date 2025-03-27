_base_ = './faster_rcnn_r50_fpn_1x_voc-orig.py'

model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    #pretrained='/path/to/pretrained', # Pre-trained weights can be loaded to initialize weights
    #reload_pretrained=True,
    roi_head=None, # Sometimes also using original roi head can improve performance
    condense_roi_head=dict(
        type='StandardRoIHead',
        is_condensing=True,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='OKPDBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            global_sp_len=5,
            global_ch=64,
            num_kp=16,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)))
        )
