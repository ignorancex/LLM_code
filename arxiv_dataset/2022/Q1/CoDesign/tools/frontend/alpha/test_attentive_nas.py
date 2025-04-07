# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
from datetime import date

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import alpha.models as models
from alpha.utils.config import setup
from alpha.utils.flops_counter import count_net_flops_and_params
import alpha.utils.comm as comm
import alpha.utils.saver as saver

from alpha.data.data_loader import build_data_loader
from alpha.utils.progress import AverageMeter, ProgressMeter, accuracy
import argparse

#parser = argparse.ArgumentParser(description='Test AttentiveNas Models')
#parser.add_argument('--config-file', default='./configs/eval_attentive_nas_models.yml')
#parser.add_argument('--model', default='a0', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
#parser.add_argument('--gpu', default=0, type=int, help='gpu id')
#
#run_args = parser.parse_args()

model_config = pickle.load(open("./alpha/models/config.pickle", "rb"))

if __name__ == '__main__':
    args = setup(run_args.config_file)
    args.model = run_args.model
    args.gpu = run_args.gpu

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.__dict__['active_subnet'] = args.__dict__['pareto_models'][args.model]
    print("hello", args.active_subnet)

    for config in model_config:
        #train_loader, val_loader, train_sampler = build_data_loader(args)
    
        ## init static attentivenas model with weights inherited from the supernet 
        model = models.model_factory.create_model(args, config)
        print(model)
    
        #model.to(args.gpu)
        model.to(torch.device('cpu'))
        model.eval()
    
    # bn running stats calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    # please consider trying a different random seed if you see a small accuracy drop
    with torch.no_grad():
        model.reset_running_stats_for_calibration()
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= args.post_bn_calibration_batch_num:
                break
            images = images.cuda(args.gpu, non_blocking=True)
            model(images)  #forward only

    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss().cuda()

        from evaluate.imagenet_eval import validate_one_subnet
        acc1, acc5, loss, flops, params = validate_one_subnet(val_loader, model, criterion, args)
        print(acc1, acc5, flops, params)


def get_model(config):
    args = setup("./alpha/configs/eval_attentive_nas_models.yml")
    args.model = "a0"
    args.gpu = 0
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    args.__dict__['active_subnet'] = args.__dict__['pareto_models'][args.model]
    print("hello", args.active_subnet)
    print("hello", config)
    model = models.model_factory.create_model(args, config)
    #print(model)
    
    #model.to(args.gpu)
    model.to(torch.device('cpu'))
    return model
    model.eval()
        
    # bn running stats calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    # please consider trying a different random seed if you see a small accuracy drop
    with torch.no_grad():
        model.reset_running_stats_for_calibration()
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= args.post_bn_calibration_batch_num:
                break
            images = images.cuda(args.gpu, non_blocking=True)
            model(images)  #forward only
            
    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss().cuda()
        
        from evaluate.imagenet_eval import validate_one_subnet
        acc1, acc5, loss, flops, params = validate_one_subnet(val_loader, model, criterion, args)
        print(acc1, acc5, flops, params)