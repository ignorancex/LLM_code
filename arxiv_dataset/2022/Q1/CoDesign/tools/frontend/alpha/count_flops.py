#!/usr/bin/env python3

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

import models as models
from utils.config import setup
from utils.flops_counter import count_net_flops_and_params
import utils.comm as comm
import utils.saver as saver

from data.data_loader import build_data_loader
from utils.progress import AverageMeter, ProgressMeter, accuracy
import argparse

#parser = argparse.ArgumentParser(description='Test AttentiveNas Models')
#parser.add_argument('--config-file', default='./configs/eval_attentive_nas_models.yml')
#parser.add_argument('--model', default='a0', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
#parser.add_argument('--gpu', default=0, type=int, help='gpu id')
#
#run_args = parser.parse_args()

model_config = pickle.load(open("./models/config.pickle", "rb"))

if __name__ == '__main__':
	args = setup("./configs/eval_attentive_nas_models.yml")
	args.model = "a0"
	args.gpu = 0
	
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	
	args.__dict__['active_subnet'] = args.__dict__['pareto_models'][args.model]
	print("hello", args.active_subnet)
	
	flops = []
	params = []
	for config in model_config:
		#train_loader, val_loader, train_sampler = build_data_loader(args)
		
		## init static attentivenas model with weights inherited from the supernet 
		model = models.model_factory.create_model(args, config)
		#print(model)
		
		#model.to(args.gpu)
		model.to(torch.device('cpu'))
		f, p = count_net_flops_and_params(model, data_shape=(1, 3, config["resolution"], config["resolution"]))
		flops.append(f)
		params.append(p)
		print(len(flops), f, p)
		if len(flops) % 500 == 0:
			pickle.dump(flops, open("flops.pickle", "wb"))
			pickle.dump(params, open("params.pickle", "wb"))
	print(len(flops), len(params))
	#print(flops)
	#print(params)
	pickle.dump(flops, open("flops.pickle", "wb"))
	pickle.dump(params, open("params.pickle", "wb"))