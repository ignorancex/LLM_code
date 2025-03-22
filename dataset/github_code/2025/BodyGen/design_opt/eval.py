import argparse
import os
import sys
sys.path.append(os.getcwd())

import yaml
from omegaconf import OmegaConf

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent

project_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str)
parser.add_argument('--epoch', default='best')
parser.add_argument('--save_video', action='store_true', default=False)
parser.add_argument('--pause_design', action='store_true', default=False)
args = parser.parse_args()

train_dir = os.path.join(args.train_dir, "0")

train_config_path = os.path.join(train_dir, ".hydra", "config.yaml")

FLAGS = yaml.safe_load(open(train_config_path, 'r'))
FLAGS = OmegaConf.create(FLAGS)

cfg = Config(FLAGS, project_path, base_dir=train_dir)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch

"""create agent"""
agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device, seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)

agent.visualize_agent(num_episode=4, save_video=args.save_video, pause_design=args.pause_design)