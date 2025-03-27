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

def test_min_steps():
    gu.seed_everything(1337)
    size = (10,10)
    config = ConfigObj('training_config.ini')
    config['env']['plot_size'] = size
    cwd = Path.cwd()
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'min_steps')
    config['env']['mode'] = 'train'
    config['env']['action_dim'] = ['0.', '0.5', '0.7', '0.9']
    config['env']['growth_stage_title'] = ['Planting', 'VE', 'VC', 'V1', 'V2', 'V3', 'V4', 'V5', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    config['env']['growth_stage_days'] = ['10', '5', '5', '5', '5', '5', '5', '3', '3', '10', '9', '9', '15', '18', '9']
    config['env']['sim_mode'] = 'growthseason'

    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    for ep in range(1):
        obs = env.reset()
        while True:
            obs, reward, done = env.step(0)
            reset = env.timestep == env.gs_end
            if done or reset:
                break
    
    assert env.timestep == 74

def test_max_steps():
    gu.seed_everything(1337)
    size = (10,10)
    config = ConfigObj('training_config.ini')
    config['env']['plot_size'] = size
    cwd = Path.cwd()
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'max_steps')
    config['env']['mode'] = 'train'
    config['env']['action_dim'] = ['0.', '0.5', '0.7', '0.9']
    config['env']['growth_stage_title'] = ['Planting', 'VE', 'VC', 'V1', 'V2', 'V3', 'V4', 'V5', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    config['env']['growth_stage_days'] = ['10', '5', '5', '5', '5', '5', '5', '3', '3', '10', '9', '9', '15', '18', '9']
    config['env']['sim_mode'] = 'growthseason'

    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    for ep in range(1):
        obs = env.reset()
        while True:
            obs, reward, done = env.step(3)
            reset = env.timestep == env.gs_end
            if done or reset:
                break
    
    assert env.timestep == 116

def test_growth_stage_min():
    gu.seed_everything(1337)
    size = (10,10)
    config = ConfigObj('training_config.ini')
    config['env']['plot_size'] = size
    cwd = Path.cwd()
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'growth_stage_min')
    config['env']['mode'] = 'train'
    config['env']['action_dim'] = ['0.', '0.5', '0.7', '0.9']
    config['env']['growth_stage_title'] = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    config['env']['growth_stage_days'] = ['3', '10', '9', '9', '15', '18', '9']
    config['env']['sim_mode'] = 'growthseason'
    config['env']['reward'] = 'expected_yield + exp_yield_pesticide + done_reward'

    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    for ep in range(1):
        obs = env.reset()
        while True:
            obs, reward, done = env.step(0)
            reset = env.timestep == env.gs_end
            if done or reset:
                break
    
    assert env.timestep == 25 and env.gs_title == 'R4'

def test_growth_stage_max():
    gu.seed_everything(1337)
    size = (10,10)
    config = ConfigObj('training_config.ini')
    config['env']['plot_size'] = size
    cwd = Path.cwd()
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'growth_stage_max')
    config['env']['mode'] = 'train'
    config['env']['action_dim'] = ['0.', '0.5', '0.7', '0.9']
    config['env']['growth_stage_title'] = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    config['env']['growth_stage_days'] = ['3', '10', '9', '9', '15', '18', '9']
    config['env']['sim_mode'] = 'growthseason'
    config['env']['reward'] = 'expected_yield + exp_yield_pesticide + done_reward'

    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    for ep in range(1):
        obs = env.reset()
        while True:
            obs, reward, done = env.step(3)
            reset = env.timestep == env.gs_end
            if done or reset:
                break
    
    assert env.timestep == 73 and env.gs_title == 'R7'

def test_survival_min():
    gu.seed_everything(1337)
    size = (10,10)
    config = ConfigObj('training_config.ini')
    config['env']['plot_size'] = size
    cwd = Path.cwd()
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'survival_min')
    config['env']['mode'] = 'train'
    config['env']['action_dim'] = ['0.', '0.5', '0.7', '0.9']
    config['env']['growth_stage_title'] = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    config['env']['growth_stage_days'] = ['3', '10', '9', '9', '15', '18', '9']
    config['env']['sim_mode'] = 'survival'
    config['env']['reward'] = 'expected_yield + exp_yield_pesticide + done_reward'
    config['training']['max_timestep'] = 200

    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    for ep in range(1):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            obs, reward, done = env.step(0)
            R += reward
            t += 1
            reset = t == env.max_timestep
            if done or reset:
                break
    
    R = np.round(R, 2)

    assert t == 39 and R == 3027.44

def test_survival_max():
    gu.seed_everything(1337)
    size = (10,10)
    config = ConfigObj('training_config.ini')
    config['env']['plot_size'] = size
    cwd = Path.cwd()
    config['general']['result_path'] = str(cwd / 'tests' / 'plots' / 'survival_max')
    config['env']['mode'] = 'train'
    config['env']['action_dim'] = ['0.', '0.5', '0.7', '0.9']
    config['env']['growth_stage_title'] = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    config['env']['growth_stage_days'] = ['3', '10', '9', '9', '15', '18', '9']
    config['env']['sim_mode'] = 'survival'
    config['env']['reward'] = 'expected_yield + exp_yield_pesticide + done_reward'
    config['training']['max_timestep'] = 200

    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    for ep in range(1):
        obs = env.reset()
        R = 0
        while True:
            obs, reward, done = env.step(1)
            R += reward
            reset = env.timestep == env.max_timestep
            if done or reset:
                break

    R = np.round(R, 2)
    
    assert env.timestep == 200 and R == 18088.68