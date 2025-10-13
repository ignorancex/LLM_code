import torch
import torch.nn as nn
import numpy as np
import random

g = torch.Generator()
g.manual_seed(0)

def get_coords_map(x, y):
    assert (
        x.coordinate_manager == y.coordinate_manager
    ), "X and Y are using different CoordinateManagers. Y must be derived from X through strided conv/pool/etc."
    return x.coordinate_manager.stride_map(x.coordinate_map_key, y.coordinate_map_key)


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_model(cfg):
    if cfg.arch == 'WSegPC':
        from models.WSegPC import WSegPC as Model
        model = Model(cfg=cfg)
    elif cfg.arch == 'mink_14A':
        from models.unet_3d import MinkUNet14A as Model
        model = Model(in_channels=cfg.in_channel, out_channels=cfg.classes, D=3)
    elif cfg.arch == 'mink_18A':
        from models.unet_3d import MinkUNet18A as Model
        model = Model(in_channels=cfg.in_channel, out_channels=cfg.classes, D=3)
    elif cfg.arch == 'mink_34C':
        from models.unet_3d import MinkUNet34C as Model
        model = Model(in_channels=cfg.in_channel, out_channels=cfg.classes, D=3)
    else:
        raise Exception('model architecture {} not supported yet'.format(cfg.arch))
    return model


def get_optimizer(args, model):
    optimizer_2d = None
    optimizer_3d = None
    if args.arch == 'mink_14A' or args.arch == 'mink_18A' or args.arch == 'mink_34C':
        optimizer_3d = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.arch == 'WSegPC':
        optimizer_3d = torch.optim.SGD(model.parameters(), lr=args.base_lr_3d, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception('optimizer architecture {} not supported yet'.format(args.arch))
    return optimizer_2d, optimizer_3d


def get_loss(args):
    if args.arch == 'WSegPC':
        from util.loss import WSegPC_Loss
        criterion = WSegPC_Loss(args)
    else:
        raise Exception('criterion architecture {} not supported yet'.format(args.arch))
    return criterion


def get_dataloader(args):
    if args.data_name == 'scannet_cross':
        from dataset.ScanNetCross import ScanNetCross, collation_fn, collation_fn_eval_all
        train_data = ScanNetCross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split=args.train_split, aug=args.aug, 
                                            loop=args.loop, view_num=args.view_num, mix3d=args.mix3d, 
                                            pseudo_label_3d=args.pseudo_label_3d, pseudo_dir=args.pseudo_dir, pseudo_key=args.pseudo_key, 
                                            data_root_img=args.data_root_img, data_root_sp=args.data_root_sp)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True, collate_fn=collation_fn,
                                                   worker_init_fn=worker_init_fn, persistent_workers=True,
                                                   generator=g)
        if args.evaluate:
            val_data = ScanNetCross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split=args.val_split, aug=False,
                                    eval_all=True, view_num=args.view_num, val_benchmark=True, 
                                    data_root_img=args.data_root_img, data_root_sp=args.data_root_sp)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True,
                                                     drop_last=False, collate_fn=collation_fn_eval_all, persistent_workers=True,
                                                     sampler=val_sampler)
        return train_loader, train_sampler, val_loader, val_sampler
    elif args.data_name == 's3dis_cross':
        from dataset.S3DIS_Cross import S3DIS_Cross, collation_fn, collation_fn_eval_all
        train_data = S3DIS_Cross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split=args.train_split, aug=args.aug, 
                                            loop=args.loop, view_num=args.view_num, mix3d=args.mix3d, 
                                            pseudo_label_3d=args.pseudo_label_3d, pseudo_dir=args.pseudo_dir, pseudo_key=args.pseudo_key, 
                                            data_root_img=args.data_root_img, data_root_sp=args.data_root_sp)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True, collate_fn=collation_fn,
                                                   worker_init_fn=worker_init_fn, persistent_workers=True,
                                                   generator=g)
        if args.evaluate:
            val_data = S3DIS_Cross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split=args.val_split, aug=False,
                                    eval_all=True, view_num=args.view_num, val_benchmark=True, 
                                    data_root_img=args.data_root_img, data_root_sp=args.data_root_sp)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True,
                                                     drop_last=False, collate_fn=collation_fn_eval_all, persistent_workers=True,
                                                     sampler=val_sampler)
        return train_loader, train_sampler, val_loader, val_sampler
    else:
        raise Exception('Dataset {} not supported yet'.format(args.data_name))
    
    