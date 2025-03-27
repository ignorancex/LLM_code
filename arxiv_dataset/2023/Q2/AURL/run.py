import random
import sys
import argparse
import warnings

import torch

from src.utilities.config_new import global_config as cfg
cfg.set_seed()
from src.utilities.config_new import parser
parser.add_argument('--mode', type=str, default="rl")
args = parser.parse_args()

for k, v in vars(args).items():
    if hasattr(cfg, k) and v:
        setattr(cfg, k, v)

from pretrain_user import pretrain_user
from supervised_learning import train_manager
from train_one_to_many import train

warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True



if __name__ == '__main__':
    mode = args.mode
    cfg.init_logging_handler(mode)


    if mode == 'sl_user':
        pretrain_user(pre_nlu=False, pre_policy=False)
    elif mode == 'sl_sys':
        train_manager()
    else:
        train()



