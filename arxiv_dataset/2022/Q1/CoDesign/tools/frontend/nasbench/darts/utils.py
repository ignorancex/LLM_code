#!/usr/bin/env python3

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):
	
	def __init__(self):
		self.reset()
		
	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0
		
	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt
		
		
def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)
	
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res


class Cutout(object):
		def __init__(self, length):
				self.length = length
			
		def __call__(self, img):
				h, w = img.size(1), img.size(2)
				mask = np.ones((h, w), np.float32)
				y = np.random.randint(h)
				x = np.random.randint(w)
			
				y1 = np.clip(y - self.length // 2, 0, h)
				y2 = np.clip(y + self.length // 2, 0, h)
				x1 = np.clip(x - self.length // 2, 0, w)
				x2 = np.clip(x + self.length // 2, 0, w)
			
				mask[y1: y2, x1: x2] = 0.
				mask = torch.from_numpy(mask)
				mask = mask.expand_as(img)
				img *= mask
				return img
	
	
def _data_transforms_cifar10(args):
	CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
	CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
	
	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	if args.cutout:
		train_transform.transforms.append(Cutout(args.cutout_length))
		
	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
		])
	return train_transform, valid_transform


def count_parameters_in_MB(model):
	return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
	filename = os.path.join(save, 'checkpoint.pth.tar')
	torch.save(state, filename)
	if is_best:
		best_filename = os.path.join(save, 'model_best.pth.tar')
		shutil.copyfile(filename, best_filename)
		
		
def save(model, model_path):
	torch.save(model.state_dict(), model_path)
	
	
def load(model, model_path):
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	
	
def drop_path(x, drop_prob):
	if drop_prob > 0.:
		keep_prob = 1.-drop_prob
		mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
		x.div_(keep_prob)
		x.mul_(mask)
	return x


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.mkdir(path)
	print('Experiment dir : {}'.format(path))
	
	if scripts_to_save is not None:
		os.mkdir(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)
			
			
			
import json
import logging
import math
import os
import pprint
import random
import sys

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def reset_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	
	
def prepare_logger(args):
	time_format = "%m/%d %H:%M:%S"
	fmt = "[%(asctime)s] %(levelname)s (%(name)s) %(message)s"
	formatter = logging.Formatter(fmt, time_format)
	logger = logging.getLogger()
	if logger.hasHandlers():
		logger.handlers.clear()
		
	def add_stdout_handler(logger):
		handler = logging.StreamHandler(sys.stdout)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		
	main_worker = not hasattr(args, "local_rank") or args.local_rank == 0
	if hasattr(args, "debug") and args.debug:
		# Debug log doesn't save
		add_stdout_handler(logger)
		logger.setLevel(logging.DEBUG)
	elif main_worker:
		# Process with local_rank > 0 will not produce any log
		add_stdout_handler(logger)
		logger.setLevel(logging.INFO)
		log_file = os.path.join(args.output_dir, "stdout.log")
		os.makedirs(args.output_dir, exist_ok=True)
		handler = logging.FileHandler(log_file, mode="a")
		handler.setFormatter(formatter)
		logger.addHandler(handler)
	else:
		logger.setLevel(logging.ERROR)
	logger.info("ARGPARSE: %s", json.dumps(vars(args)))
	logger.debug(pprint.pformat(vars(args)))
	if main_worker:
		with open(os.path.join(args.output_dir, "config.json"), "w") as f:
			json.dump(vars(args), f, sort_keys=True, indent=2)
			
	return logger


def prepare_split(args):
	pass
	
	
def prepare_experiment(args):
	reset_seed(args.seed)
	
	args.tb_dir = os.path.join(args.output_dir, "tb")
	os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
	
	prepare_split(args)
	prepare_distributed(args)
	logger = prepare_logger(args)
	return logger


def prepare_distributed(args):
	if not hasattr(args, "distributed"):
		args.distributed = False
		return
	if args.distributed:
		args.rank = int(os.environ.get("RANK", 0))
		# to be compatible with single worker mode
		if "WORLD_SIZE" not in os.environ:
			os.environ["WORLD_SIZE"] = "1"
		world_size = int(os.environ["WORLD_SIZE"])
		
		master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
		master_port = os.environ.get("MASTER_PORT", "54321")
		
		assert 0 <= args.local_rank < torch.cuda.device_count()
		torch.cuda.set_device(args.local_rank)
		args.world_size = world_size
		args.master_addr = master_addr
		args.master_port = master_port
		torch.distributed.init_process_group(backend="nccl", init_method="tcp://{}:{}".format(master_addr, master_port),
											world_size=world_size, rank=args.rank)
		args.is_worker_main = args.rank == 0
		args.is_worker_logging = args.local_rank == 0
	else:
		if "RANK" in os.environ:
			logger.warning("Rank is found in environment variables. Did you forget to set distributed?")
		args.is_worker_main = True
		args.is_worker_logging = True
		args.world_size = 1
		args.rank = args.local_rank = 0
		
		
def count_convNd(m, _, y):
	cin = m.in_channels
	kernel_ops = m.weight.size()[2] * m.weight.size()[3]
	ops_per_element = kernel_ops
	output_elements = y.nelement()
	total_ops = cin * output_elements * ops_per_element // m.groups  # cout x oW x oH
	m.total_ops = torch.Tensor([int(total_ops)])
	m.module_used = torch.tensor([1])
	
	
def count_linear(m, _, __):
	total_ops = m.in_features * m.out_features
	m.total_ops = torch.Tensor([int(total_ops)])
	m.module_used = torch.tensor([1])
	
	
def count_naive(m, _, __):
	m.module_used = torch.tensor([1])
	
	
register_hooks = {
	nn.Conv1d: count_convNd,
	nn.Conv2d: count_convNd,
	nn.Conv3d: count_convNd,
	nn.Linear: count_linear,
}


def flops_counter(model, input_size):
	handler_collection = []
	logger = logging.getLogger(__name__)
	
	def add_hooks(m_):
		if len(list(m_.children())) > 0:
			return
		
		m_.register_buffer('total_ops', torch.zeros(1))
		m_.register_buffer('total_params', torch.zeros(1))
		m_.register_buffer('module_used', torch.zeros(1))
		
		for p in m_.parameters():
			m_.total_params += torch.Tensor([p.numel()])
			
		m_type = type(m_)
		fn = register_hooks.get(m_type, count_naive)
		
		if fn is not None:
			_handler = m_.register_forward_hook(fn)
			handler_collection.append(_handler)
			
	def remove_buffer(m_):
		if len(list(m_.children())) > 0:
			return
		
		del m_.total_ops, m_.total_params, m_.module_used
		
	original_device = next(model.parameters()).device
	training = model.training
	
	model.eval()
	model.apply(add_hooks)
	
	assert isinstance(input_size, tuple)
	if torch.is_tensor(input_size[0]):
		x = (t.to(original_device) for t in input_size)
	else:
		x = (torch.zeros(input_size).to(original_device), )
	with torch.no_grad():
		model(*x)
		
	total_ops = 0
	total_params = 0
	for name, m in model.named_modules():
		if len(list(m.children())) > 0:  # skip for non-leaf module
			continue
		if not m.module_used:
			continue
		total_ops += m.total_ops
		total_params += m.total_params
		logger.debug("%s: %.2f %.2f", name, m.total_ops.item(), m.total_params.item())
	
	total_ops = total_ops.item()
	total_params = total_params.item()
	
	model.train(training).to(original_device)
	for handler in handler_collection:
		handler.remove()
	model.apply(remove_buffer)
	
	return total_ops, total_params