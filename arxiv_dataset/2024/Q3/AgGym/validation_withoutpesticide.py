from sqlite3 import Timestamp
from tokenize import Double
from modules import env_modules_pestval
from utils import general_utils as gu
from configobj import ConfigObj
from modules import threat_modules_pestval as tm
import pfrl
import torch
from torch import nn
import gym
import numpy as np
from einops import rearrange
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import time
import os
import random
import argparse
import pdb

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--seed', type=int, default=123, help='Description for foo argument')
args = parser.parse_args()
cwd = Path.cwd()
now = datetime.now()
now_str = now.strftime("%d-%m-%y_%H-%M-%S")
(cwd / 'results' / now_str).mkdir(parents=True, exist_ok=True)
(cwd / 'results' / now_str / 'data').mkdir(parents=True, exist_ok=True)
(cwd / 'results' / now_str / 'eval').mkdir(parents=True, exist_ok=True)
(cwd / 'results' / now_str / 'agent_weights').mkdir(parents=True, exist_ok=True)
gu.seed_everything(args.seed)
config = ConfigObj('training_config.ini')
config['general']['result_path'] = str(cwd / 'results' / now_str)
logging.basicConfig(filename=config['general']['result_path'] + '/train.log', 
                    level=eval(f"logging.{config['general']['level']}"), 
                    filemode='w', 
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info("Training")
logging.info(f"Seed: {args.seed}")
branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
branch = branch.decode('utf-8').replace('\n','')
commit = subprocess.check_output(["git", "rev-parse", "HEAD"])
commit = commit.decode('utf-8').replace('\n','')
logging.info(f"Github Branch: {branch}")
logging.info(f"Github Commit: {commit}")
gu.print_to_logs(config)
env = env_modules_pestval.cartesian_grid()
gu.set_as_attr(env, config)
env.init()
env.rl_init()

obs_size = env.state_space
n_actions = env.action_space.n

env.gpu = int(env.gpu)
print(f"Model: {env.model}, GPU: {env.gpu}")


total_step = 0
training_R = []
for ep in range(1):
    obs = env.reset()
    if env.sim_mode == 'growthseason':
        end_condition = env.gs_end
        # end_condition = 5
    elif env.sim_mode == 'survival':
        end_condition = env.max_timestep
    # training_R_ep=[]
    i = 0
    # R = 0
    # agent_list = []
    env_list = []
    print("episod")
    while True:
        print(total_step)
        # agent_start = time.time()
        # action = agent.act(obs.astype(np.float32))
        # action = agent.act(obs.item())
        # agent_end = time.time()
        # agent_list.append(agent_end - agent_start)
        env_start = time.time()
        # obs, reward, done = env.step(action)
        obs, reward, done = env.step(0)
        env_end = time.time()
        env_list.append(env_end - env_start)
        # R += reward
        reset = env.timestep == end_condition
        print(type(obs[0]), type(reward))
        print("dead_counts={}, infect_counts={}, total_step={}".format(env.dead_counts[-1], env.infect_counts[-1], total_step))
        total_yield=env.state_space+np.sum(np.array(env.Degraded_list))
        print("total_yield={}".format(total_yield))
        print("Degraded_list={}".format(np.sum(np.array(env.Degraded_list))))
        print(f"Number of elements in the Degraded List are {len(env.Degraded_list)}")
        print("Degraded_list={}".format(np.array(env.Degraded_list)))
        # agent.observe(obs.astype(np.float32), reward.astype(np.float32), done, reset)
        # logging.debug(f"Step: {i}, Action: {action} Reward: {reward}, Done: {done}, {env.gs_end}")
        print(f'Timestep is {env.timestep}')
        i += 1
        total_step += 1
        if done or reset:
            break

