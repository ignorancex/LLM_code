import copy
import numpy as np
import os
import sys
import random
import time

from utils import general_utils as gu
from configobj import ConfigObj
from modules import env_modules

EMPTY = 0.0
DEAD = 1.0
INFECTED = 2.0
ALIVE = 3.0

def test_2d_array_to_flatten():
    gu.seed_everything(1337)
    size = (3,3)
    config = ConfigObj('training_config.ini')
    config['env']['plot_size'] = size
    config['env']['reward'] = 'plot_state'
    config['env']['state'] = 'flattener'
    config['env']['sim_mode'] = 'growthseason'
    env = env_modules.insectEnv()
    gu.set_as_attr(env, config)
    env.init()

    env.reset()
    env.plot_state[1,1] = INFECTED
    env.plot_state[2,2] = DEAD
    env.threat.infect_list = {(1, 1):0}
    obs, reward, done = env.step(0)
    
    assert obs.all() == (env.plot_state.reshape(9)/3).all()

