import sys
sys.path.append("..")
import numpy as np
import torch
import torch
import torch
import random
import argparse


def get_run_type_name(args): 
	name = "_"
	if args.add_caption: 
		name += "caption_"

	return name[:-1] 

def worker_init_fn(worker_id):                                                                                                                                
	seed = 0                
																																
	torch.manual_seed(seed)                                                                                                                                   
	torch.cuda.manual_seed(seed)                                                                                                                              
	torch.cuda.manual_seed_all(seed)                                                                                          
	np.random.seed(seed)                                                                                                             
	random.seed(seed)                                                                                                       
	torch.manual_seed(seed)                                                                                                                                   
	return

def get_out_of_domain_logos(logos_dir):
	with open(f"{logos_dir}/out_of_domain.txt", "r") as f:
		logos = f.readlines()[0]
		logos = logos.split(",")
	logos = [logo.strip() + ".jpg" for logo in logos]
	return logos



def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
