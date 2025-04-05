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
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from mylib.sync_batchnorm import DataParallelWithCallback
from lidc_dataset import LIDCSegDataset
from mylib.utils import MultiAverageMeter, save_model, log_results, to_var, set_seed, \
        to_device, initialize, categorical_to_one_hot, copy_file_backup, redirect_stdout, \
        model_to_syncbn
from mylib.metrics import cal_batch_iou, cal_batch_dice
from mylib.loss import soft_dice_loss

from lidc_config import LIDCSegConfig as cfg
from lidc_config import LIDCEnv as env

from resnet import FCNResNet
from densenet import FCNDenseNet
from vgg import FCNVGG
from acsconv.converters import ACSConverter, SoftACSConverter, Conv3dConverter, Conv2_5dConverter
from load_pretrained_weights_funcs import load_mednet_pretrained_weights, load_video_pretrained_weights


def main(save_path=cfg.save, 
         n_epochs=cfg.n_epochs, 
         seed=cfg.seed
         ):
    # set seed
    if seed is not None:
        set_seed(cfg.seed)
    cudnn.benchmark = True
    # back up your code
    os.makedirs(save_path)
    copy_file_backup(save_path)
    redirect_stdout(save_path)

    # Datasets
    train_set = LIDCSegDataset(crop_size=48, move=5, data_path=env.data, train=True)
    valid_set = None
    test_set = LIDCSegDataset(crop_size=48, move=5, data_path=env.data, train=False)

    # Define model
    model_dict = {'resnet18': FCNResNet, 'vgg16': FCNVGG, 'densenet121': FCNDenseNet}
    model = model_dict[cfg.backbone](pretrained=cfg.pretrained, num_classes=2, backbone=cfg.backbone)

    # convert to counterparts and load pretrained weights according to various convolution
    if cfg.conv=='ACSConv':
        model  = model_to_syncbn(ACSConverter(model))
    if cfg.conv=='Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if cfg.conv=='Conv3d':
        if cfg.pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
            if cfg.pretrained_3d == 'video':
                model = load_video_pretrained_weights(model, env.video_resnet18_pretrain_path)
            elif cfg.pretrained_3d == 'mednet':
                model = load_mednet_pretrained_weights(model, env.mednet_resnet18_pretrain_path)
    print(model)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))
    # train and test the model
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save_path, n_epochs=n_epochs)

    print('Done!')



def train(model, train_set, test_set, save, valid_set, n_epochs):
    '''
    Main training function
    '''
    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
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
        if cfg.use_syncbn:
            print('Using sync-bn')
            model_wrapper = DataParallelWithCallback(model).cuda()
        else:
            model_wrapper = torch.nn.DataParallel(model).cuda()
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones,
                                                     gamma=cfg.gamma)
    # Start logging
    logs = ['loss', 'iou', 'dice', 'iou0', 'iou1', 'dice0', 'dice1', 'dice_global']
    train_logs = ['train_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]

    log_dict = OrderedDict.fromkeys(train_logs+test_logs, 0)
    with open(os.path.join(save, 'logs.csv'), 'w') as f:
        f.write('epoch,')
        for key in log_dict.keys():
            f.write(key+',')
        f.write('\n')
    with open(os.path.join(save, 'loss_logs.csv'), 'w') as f:
        f.write('iter,train_loss,\n')
    writer = SummaryWriter(log_dir=os.path.join(save, 'Tensorboard_Results'))

    # train and test the model
    best_dice_global = 0
    global iteration
    iteration = 0
    for epoch in range(n_epochs):
        os.makedirs(os.path.join(cfg.save, 'epoch_{}'.format(epoch)))
        print('learning rate: ', scheduler.get_lr())
        # train epoch
        train_meters = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            writer=writer
        )
        # test epoch
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
        # save model checkpoint
        if cfg.save_all:
            torch.save(model.state_dict(), os.path.join(save, 'epoch_{}'.format(epoch), 'model.dat'))

        if log_dict['test_dice_global'] > best_dice_global:
            torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
            best_dice_global = log_dict['test_dice_global']
            print('New best global dice: %.4f' % log_dict['test_dice_global'])
        else:
            print('Current best global dice: %.4f' % best_dice_global)
    # end 
    writer.close()
    with open(os.path.join(save, 'logs.csv'), 'a') as f:
        f.write(',,,,best global dice,%0.5f\n' % (best_dice_global))
    print('best global dice: ', best_dice_global)


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1, writer=None):
    '''
    One training epoch
    '''
    meters = MultiAverageMeter()
    # Model on train mode
    model.train()
    global iteration
    intersection = 0
    union = 0
    end = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        x = to_var(x)
        y = to_var(y)
        # forward and backward
        pred_logit = model(x)
        y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)

        loss = soft_dice_loss(pred_logit, y_one_hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calculate metrics
        pred_classes = pred_logit.argmax(1)
        intersection += ((pred_classes==1) * (y[:,0]==1)).sum().item()
        union += ((pred_classes==1).sum() + y[:,0].sum()).item()
        batch_size = y.size(0)

        iou = cal_batch_iou(pred_logit, y_one_hot)
        dice = cal_batch_dice(pred_logit, y_one_hot)
        # log
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        with open(os.path.join(cfg.save, 'loss_logs.csv'), 'a') as f:
            f.write('%09d,%0.6f,\n'%((iteration + 1),loss.item(),))
        iteration += 1

        logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                            [iou[i].item() for i in range(len(iou))]+ \
                            [dice[i].item() for i in range(len(dice))]+ \
                            [time.time() - end]
        meters.update(logs, batch_size)   
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
    dice_global = 2. * intersection / union
    return meters.avg[:-1] + [dice_global]


def test_epoch(model, loader, epoch, print_freq=1, is_test=True, writer=None):
    '''
    One test epoch
    '''
    meters = MultiAverageMeter()
    model.eval()
    intersection = 0
    union = 0
    end = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = to_var(x)
            y = to_var(y)
            # forward
            pred_logit = model(x)
            # calculate metrics
            y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)
            pred_classes = pred_logit.argmax(1)
            intersection += ((pred_classes==1) * (y[:,0]==1)).sum().item()
            union += ((pred_classes==1).sum() + y[:,0].sum()).item()

            loss = soft_dice_loss(pred_logit, y_one_hot)
            batch_size = y.size(0)

            
            iou = cal_batch_iou(pred_logit, y_one_hot)
            dice = cal_batch_dice(pred_logit, y_one_hot)

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
    dice_global = 2. * intersection / union

    return meters.avg[:-1] + [dice_global]

if __name__ == '__main__':
    fire.Fire(main)
