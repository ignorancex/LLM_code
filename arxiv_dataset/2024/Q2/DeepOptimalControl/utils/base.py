import argparse
import json
import random
from datetime import datetime

import numpy as np
import torch


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),
        'args': str(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Generation
    data_type_list = ['train', 'valid', 'test']
    bvp_type_list = ['no_march', 'space_march_fw']
    parser.add_argument('--data_type', type=str, choices=data_type_list, default='train')
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument('--time_step', type=int, default=100)
    parser.add_argument('--bvp_type', type=str, choices=bvp_type_list, default='space_march_fw')
    parser.add_argument('--k_try', type=int, default=30)
    parser.add_argument('--num_processors', type=int, default=1)

    # Problem Setting
    parser.add_argument('--data_train', type=str, default='data_fix_train.mat')
    parser.add_argument('--scaling', type=str, default='data_fix_scaling.mat')
    parser.add_argument('--data_valid', type=str, default='data_fix_valid.mat')
    parser.add_argument('--experiment_name', default='quadrotor_fix', type=str)
    parser.add_argument('--problem_id', default="1_20", type=str) 
    parser.add_argument('--X0_scale', default=10, type=int)
    parser.add_argument('--T', default=16, type=int)
    parser.add_argument('--x_dim', default=12, type=int)
    parser.add_argument('--u_dim', default=4, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)

    # Training
    parser.add_argument('--num_epochs', default=1000, type=int)     # sl
    parser.add_argument('--num_iters', default=3000, type=int)      # direct
    parser.add_argument('--pre_epochs', default=1000, type=int)     # finetune
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.01, type=float) 
    parser.add_argument('--adjoint', default=False, action='store_true')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--adaptive_type', type=str, default='None')

    # Misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_ckpt', default=False, action='store_true')
    parser.add_argument('--use_gpu', default=False, action='store_true')

    args = parser.parse_args()

    # CUDA support
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    return args
