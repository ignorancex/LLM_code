name = 'CMR'
# hyperparameter
default_config = dict(
    batch_size=16,
    num_epoch=100,
    learning_rate=1e-4,            # learning rate of Adam origin is 5 3for s
    weight_decay=0.1,             # weight decay 
    num_workers=8,    #origin is 8

    train_name = name,
    model_path = name+'.pt',
    test_vendor = 'D',
    ratio = 1,                   # 100%
    CPS_weight = 1.5,
    gpus= [0],
    ifFast = False,
    Pretrain = True,
    pretrain_file = '/home/wmaag/UROP3100/EPL_SemiDG-master/resnet50_v1c.pth',

    restore = True,
    restore_from = name+'.pt',

    # for cutmix
    cutmix_mask_prop_range = (0.25, 0.5),
    cutmix_boxmask_n_boxes = 3,
    cutmix_boxmask_fixed_aspect_ratio = True,
    cutmix_boxmask_by_size = True,
    cutmix_boxmask_outside_bounds = True,
    cutmix_boxmask_no_invert = True,

    Fourier_aug = True,
    fourier_mode = 'AS',

    # for bn
    bn_eps = 1e-5,
    bn_momentum = 0.1,
)
