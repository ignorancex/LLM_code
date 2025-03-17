import copy
import numpy as np
import os
import sys
import random
import time

from utils import general_utils as gu
from configobj import ConfigObj
from modules import env_modules
from pathlib import Path

EMPTY = 0.0
DEAD = 1.0
INFECTED = 2.0
ALIVE = 3.0

def test_plot_state():
    gu.seed_everything(1337)
    size = (3,3)
    cwd = Path.cwd()
    config = ConfigObj('training_config.ini')
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'plot_state')
    config['env']['plot_size'] = size
    config['env']['reward'] = 'plot_state'
    config['env']['sim_mode'] = 'growthseason'
    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    env.reset()
    env.plot_state[1,1] = INFECTED
    env.plot_state[2,2] = DEAD
    env.threat.infect_list = {(1, 1):0}
    obs, reward, done = env.step(0)
    
    assert reward == -4

def test_expected_yield():
    gu.seed_everything(1337)
    size = (3,3)
    cwd = Path.cwd()
    config = ConfigObj('training_config.ini')
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'expected_yield')
    config['env']['plot_size'] = size
    config['env']['reward'] = 'expected_yield'
    config['env']['sim_mode'] = 'growthseason'
    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    env.reset()
    env.plot_state[1,1] = INFECTED
    env.plot_state[2,2] = INFECTED
    env.plot_state[1,0] = INFECTED
    env.threat.infect_list = {(1, 1):6, (2,2):2, (1,0):8}
    obs, reward, done = env.step(0)
    print(env.threat.infect_list)
    
    assert reward == -1.73

def test_multi_reward():
    gu.seed_everything(1337)
    size = (3,3)
    cwd = Path.cwd()
    config = ConfigObj('training_config.ini')
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'multi_reward')
    config['env']['plot_size'] = size
    config['env']['reward'] = 'expected_yield + exp_yield_pesticide'
    config['env']['sim_mode'] = 'growthseason'
    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    env.reset()
    env.plot_state[1,1] = INFECTED
    env.plot_state[2,2] = INFECTED
    env.plot_state[1,0] = INFECTED

    print(f"env.threat_list: {len(env.threat.infect_list)}, {env.threat.infect_list}")
    env.threat.infect_list = {(1, 1):6, (2,2):2, (1,0):8}
    obs, reward, done = env.step(2)

    print(f"env.threat_list: {len(env.threat.infect_list)}, {env.threat.infect_list}")
    print(env.threat.infect_list)
    print(reward)
    obs, reward, done = env.step(2)

    print(f"env.threat_list: {len(env.threat.infect_list)}, {env.threat.infect_list}")
    print(env.threat.infect_list)
    print(reward)
    obs, reward, done = env.step(2)

    print(f"env.threat_list: {len(env.threat.infect_list)}, {env.threat.infect_list}")
    print(env.threat.infect_list)
    print(reward)

    assert reward == -2.55

def test_done_reward_terminal():
    gu.seed_everything(1337)
    size = (3,3)
    cwd = Path.cwd()
    config = ConfigObj('training_config.ini')
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'done_reward_terminal')
    config['env']['plot_size'] = size
    config['env']['reward'] = 'done_reward'
    config['env']['sim_mode'] = 'growthseason'
    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    env.reset()
    env.plot_state[:,:] = DEAD

    print(f"env.threat_list: {len(env.threat.infect_list)}, {env.threat.infect_list}")
    env.threat.infect_list = {(0,0):0, (0,1):0, (0,2):0, (1,0):0, (1,1):0, (1,2):0, (2,0):0, (2,1):0, (2,2):0}
    obs, reward, done = env.step(1)

    assert reward == -1000

def test_done_reward_healthy():
    gu.seed_everything(1337)
    size = (3,3)
    cwd = Path.cwd()
    config = ConfigObj('training_config.ini')
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'done_reward_healthy')
    config['env']['plot_size'] = size
    config['env']['reward'] = 'done_reward'
    config['env']['sim_mode'] = 'growthseason'
    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    env.reset()
    env.plot_state[:,:] = ALIVE
    env.timestep = env.gs_end - 1

    print(f"env.threat_list: {len(env.threat.infect_list)}, {env.threat.infect_list}")
    obs, reward, done = env.step(1)

    assert reward == 1000

def test_done_reward_survival():
    gu.seed_everything(1337)
    size = (3,3)
    cwd = Path.cwd()
    config = ConfigObj('training_config.ini')
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'done_reward_survival')
    config['training']['max_timestep'] = 0
    config['env']['plot_size'] = size
    config['env']['reward'] = 'done_reward'
    config['env']['sim_mode'] = 'survival'
    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    env.reset()
    env.plot_state[:,:] = ALIVE

    print(f"env.threat_list: {len(env.threat.infect_list)}, {env.threat.infect_list}")
    obs, reward, done = env.step(1)

    assert reward == 1000