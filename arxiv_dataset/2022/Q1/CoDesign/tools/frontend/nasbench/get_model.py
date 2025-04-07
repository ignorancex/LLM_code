#!/usr/bin/env python3

import os
import sys
import glob
import numpy as np
import torch
import logging
import argparse

import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from nasbench.darts.model import NetworkCIFAR as Network
import nasbench.darts.genotypes as genotypes
from nasbench.darts.genotypes import get_genotype_from_arch 
import nasbench.darts.utils
from nasbench.darts.utils import flops_counter

import pickle

#def get_model():
#	print("hello")
#	log_format = '%(asctime)s %(message)s'
#	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#			format=log_format, datefmt='%m/%d %I:%M:%S %p')
#	
#	CIFAR_CLASSES = 10
#	
#	model_arch = pickle.load(open("./nasbench/new/model_arch_depth.pickle", "rb"))
#	
#	if os.path.exists("./nasbench/new/flops_arr_depth.pickle"):
#		flops_arr = pickle.load(open("./nasbench/new/flops_arr_depth.pickle", "rb"))
#		print(len(flops_arr))
#	else:
#		flops_arr = []
#	all_model = []
#	
#	index = 0
#	#flops_arr = []
#	for i in range(len(flops_arr), len(model_arch)):
#		if index < 100000:
#			arch = model_arch[i]
#			genotype = get_genotype_from_arch(arch)
#			#genotype = eval("genotypes.%s" % 'DARTS')
#			#print(genotype)
#			print(i)
#		#	for geno in model_genotype:
#		#		genotype = eval("genotypes.%s" % 'DARTS')
#			model = Network(36, CIFAR_CLASSES, 20, True, genotype)
#			#print(model)
#			flops, params = flops_counter(model, (1, 3, 32, 32))
#			flops_arr.append(flops)
#			#print('DARTS', genotype)
#			#print(model)
#			#utils.load(model, './cifar10_model.pt')
#			#torch.save(model, './model/0.pt')
#			
#			#model2 = torch.load('./model/0.pt')
#			all_model.append(model)
#			index += 1
#			
#			if index % 100 == 0 or index == len(model_arch)-1:
#				pickle.dump(flops_arr, open("./nasbench/new/flops_arr_depth.pickle", "wb"))
#				print(flops_arr)
#				print(len(flops_arr))
#				
#	print(all_model)
#	return all_model
#
#
#get_model()


def get_model():
	print("hello")
	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
			format=log_format, datefmt='%m/%d %I:%M:%S %p')
	
	CIFAR_CLASSES = 10
	
	model_arch = pickle.load(open("./nasbench/new/model_arch_depth.pickle", "rb"))
	select_index = pickle.load(open("./nasbench/new/model_index_depth.pickle", "rb"))
	depth = pickle.load(open("./nasbench/new/depth.pickle", "rb"))
	print(len(select_index))
	all_model = []
	
	for i in range(400):
		index = select_index[i]
		arch = model_arch[index]
		genotype = get_genotype_from_arch(arch)
#		print(arch, genotype)
#		from collections import namedtuple
#		Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
#		DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
#		genotype = DARTS_V2
		print(i)
		#print(arch, genotype)
		model = Network(36, CIFAR_CLASSES, 20, False, genotype)
#		print(model)
#		print(h)
		all_model.append(model)
				
	return all_model