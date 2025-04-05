# encoding: utf-8

import _init_paths

import fire
import time
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from poc_dataset import BaseDatasetVoxel
from mylib.loss import soft_cross_entropy_loss
from mylib.utils import MultiAverageMeter, save_model, log_results, to_var, set_seed, \
        to_device, initialize, categorical_to_one_hot, copy_file_backup, redirect_stdout
from poc_config import POCVoxelConfig as cfg
from poc_config import POCVoxelEnv as env

from unet import UNet
from acsconv.models import ACSUNet
from acsconv.converters import ACSConverter, Conv3dConverter, Conv2_5dConverter

from mylib.metrics import cal_batch_iou, cal_batch_dice
from mylib.loss import soft_dice_loss

def main(save_path=cfg.save, 
         n_epochs=cfg.n_epochs, 
         seed=cfg.seed
         ):
    if seed is not None:
        set_seed(cfg.seed)
    cudnn.benchmark = True

    os.makedirs(save_path)
    copy_file_backup(save_path)
    redirect_stdout(save_path)
    # Datasets
    train_data = env.data_train
    test_data = env.data_test
    shape_cp = env.shape_checkpoint

    train_set = BaseDatasetVoxel(train_data, cfg.train_samples)
    valid_set = None
    test_set = BaseDatasetVoxel(test_data, 200)

    # # Models

    model = UNet(6)
    if cfg.conv == 'Conv3D':
        model = Conv3dConverter(model)
        initialize(model.modules())
    elif cfg.conv == 'Conv2_5D':
        if cfg.pretrained:
            shape_cp = torch.load(shape_cp)
            shape_cp.popitem()
            shape_cp.popitem()
            incompatible_keys = model.load_state_dict(shape_cp, strict=False)
            print('load shape pretrained weights\n', incompatible_keys)
        model = Conv2_5dConverter(model)
    elif cfg.conv == 'ACSConv':
        # You can use either the naive ``ACSUNet`` or the ``ACSConverter(model)``
        model = ACSConverter(model)
        # model = ACSUNet(6)
        if cfg.pretrained:
            shape_cp = torch.load(shape_cp)
            shape_cp.popitem()
            shape_cp.popitem()
            incompatible_keys = model.load_state_dict(shape_cp, strict=False)
            print('load shape pretrained weights\n', incompatible_keys)
    else:
        raise ValueError('not valid conv')
    
    print(model)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))
    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save_path, n_epochs=n_epochs)

    print('Done!')



def train(model, train_set, test_set, save, valid_set, n_epochs):


    # Data loaders
    train_loader = DataLoader(train_set, batch_size=cfg.train_batch_size, shuffle=True,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.test_batch_size, shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    if valid_set is None:
        valid_loader = None
    else:
        valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    # Model on cuda
    model = to_device(model)

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones,
                                                     gamma=cfg.gamma)

    # Start log
    logs = ['loss', 'iou', 'dice'] + ['iou{}'.format(i) for i in range(6)]+['dice{}'.format(i) for i in range(6)]
    train_logs = ['train_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+test_logs, 0)
    with open(os.path.join(save, 'logs.csv'), 'w') as f:
        f.write('epoch,')
        for key in log_dict.keys():
            f.write(key+',')
        f.write('\n')
    writer = SummaryWriter(log_dir=os.path.join(save, 'Tensorboard_Results'))

    # Train model
    best_dice = 0

    for epoch in range(n_epochs):
        os.makedirs(os.path.join(cfg.save, 'epoch_{}'.format(epoch)))
        train_meters = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            writer=writer
        )
        # if (epoch+1)%5==0:
        test_meters = test_epoch(
            model=model_wrapper,
            loader=test_loader,
            epoch=epoch,
            is_test=True,
            writer = writer
        )
        scheduler.step()

        # Log results
        for i, key in enumerate(train_logs):
            log_dict[key] = train_meters[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_meters[i]

        log_results(save, epoch, log_dict, writer=writer)

        if cfg.save_all:
            torch.save(model.state_dict(), os.path.join(save, 'epoch_{}'.format(epoch), 'model.dat'))

        if log_dict['test_dice'] > best_dice:
            torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
            best_dice = log_dict['test_dice']
            print('New best dice: %.4f' % log_dict['test_dice'])
        else:
            print('Current best dice: %.4f' % best_dice)
    writer.close()

    with open(os.path.join(save, 'logs.csv'), 'a') as f:
        f.write(',,,,best dice,%0.5f\n' % (best_dice))
    # Final test of the best model on test set
    print('best dice: ', best_dice)

iteration = 0

def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1, writer=None):
    meters = MultiAverageMeter()
    # Model on train mode
    model.train()
    global iteration
    end = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        # Create vaiables
        x = to_var(x)
        y = to_var(y)
        # compute output
        pred_logit = model(x)
        loss = soft_dice_loss(pred_logit, y, smooth=1e-2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y = y.long()

        batch_size = y.size(0)
        iou = cal_batch_iou(pred_logit, y)
        dice = cal_batch_dice(pred_logit, y)

        logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                            [iou[i].item() for i in range(len(iou))]+ \
                            [dice[i].item() for i in range(len(dice))]+ \
                            [time.time() - end]
        meters.update(logs, batch_size)   
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        with open(os.path.join(cfg.save, 'loss_logs.csv'), 'a') as f:
            f.write('%09d,%0.6f,\n'%((iteration + 1),loss.item(),))
        iteration += 1


        # measure elapsed time
        end = time.time()
        # print stats
        print_freq = 2 // meters.val[-1] + 1
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                'IOU %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                'DICE %.4f (%.4f)' % (meters.val[2], meters.avg[2]),
            ])
            print(res)

    return meters.avg[:-1]


def test_epoch(model, loader, epoch, print_freq=1, is_test=True, writer=None):
    meters = MultiAverageMeter()
    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = to_var(x)
            y = to_var(y)
            pred_logit = model(x)
            loss = soft_dice_loss(pred_logit, y, smooth=1e-2)
            y = y.long()
            batch_size = y.size(0)
            iou = cal_batch_iou(pred_logit, y)
            dice = cal_batch_dice(pred_logit, y)

            logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                                [iou[i].item() for i in range(len(iou))]+ \
                                [dice[i].item() for i in range(len(dice))]+ \
                                [time.time() - end]
            meters.update(logs, batch_size)

            end = time.time()

            print_freq = 2 // meters.val[-1] + 1
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                    'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                    'IOU %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                    'DICE %.4f (%.4f)' % (meters.val[2], meters.avg[2]),
                ])
                print(res)

    return meters.avg[:-1]

if __name__ == '__main__':
    fire.Fire(main)

