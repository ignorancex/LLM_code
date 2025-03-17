from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import random
import numpy as np

from xoa.commons.logger import *
from .div import DiversificationStrategy     

class EpsilonGreedyStrategy(DiversificationStrategy):
    def __init__(self, num_arms, values, counts, 
        title="", 
        init_eps=1.0,
        decay_factor=5,
        reward_scaling=None, 
        log_scale_decay=False):

        name = "EG"
        if reward_scaling == 'LOG_ERR':
            name = name + '_LE'

        if title != "":
            name = name + "_" + title

        self.initial_epsilon = init_eps
        super(EpsilonGreedyStrategy, self).__init__(name, num_arms, values, counts, init_eps)
        
        self.decay_factor = decay_factor
        self.log_scale_decay = log_scale_decay
        self.reward_scaling = reward_scaling

    def next(self, step):
        ran_num = np.random.random_sample()
        idx = None

        decay_steps = self.num_arms * self.decay_factor
        # linear scale decreasing
        self.epsilon = self.initial_epsilon - (step // decay_steps) * 0.1
            
        # force to keep epsilon greater than 0.1
        if self.epsilon < 0.1:
            self.epsilon = 0.1

        max_val = np.max(self.values)
        max_idxs = np.where(self.values == max_val)[0]

        if ran_num < 1 - self.epsilon:
            idx = np.random.choice(max_idxs, 1)[0]
        else:
            if len(max_idxs) == len(self.values):
                idx = np.random.choice(max_idxs, 1)[0]
            else:
                temp = np.arange(len(self.values))
                temp = np.setdiff1d(temp, max_idxs)
                idx = np.random.choice(temp, 1)[0]
        
        idx = np.asscalar(idx)
                            
        return idx

    def update(self, arm_index, curr_acc, opt):
        acc = self.scale_acc(curr_acc)
        reward = (acc - self.values[arm_index])
        if self.counts[arm_index] > 0:  
            reward = reward / self.counts[arm_index]                
        self.counts[arm_index] += 1
        self.values[arm_index] += reward
    
    def scale_acc(self, acc):
        if self.reward_scaling is None:
            return acc
        elif self.reward_scaling == "LOG_ERR":
            
            # XXX: truncate extrapolated estimation
            if acc < 0:
                acc = 0.00001
            if acc > 1.0:
                acc = 0.99999

            # scaling log with error
            err = 1.0 - acc
            abs_log_err = math.fabs(math.log10(err))
            return abs_log_err
        else:
            raise TypeError('unsupported reward scaling:{}'.format(self.reward_scaling))


class GreedyTimeStrategy(DiversificationStrategy):
    
    def __init__(self, num_arms, values, counts, 
                title="",
                time_unit='H',
                reward_scaling=None):
        
        name = "GT" + time_unit

        if reward_scaling == 'LOG_ERR':
            name = name + '_LE'

        if title != "":
            name = name + "_" + title

        super(GreedyTimeStrategy, self).__init__(name, num_arms, values, counts, 1.0)
        self.reward_scaling = reward_scaling

        self.time_unit = time_unit       
        self.start_time = time.time()
        self.cum_exec_time = 0

    def get_elapsed_time(self):
        if self.cum_exec_time == 0:
            elapsed = time.time() - self.start_time
        else:
            elapsed = self.cum_exec_time

        if self.time_unit == 'H':
            elapsed = math.ceil(elapsed / (60 * 60))
        elif self.time_unit == 'M':
            elapsed = math.ceil(elapsed / 60)
        elif self.time_unit == 'S':
            elapsed = math.ceil(elapsed)        
        else:
            warn('unsupported time unit: {}'.format(self.time_unit))
        return elapsed
    
    def next(self, step):
        ran_num = np.random.random_sample()
        idx = None

        # time dependent epsilon
        t = self.get_elapsed_time()         
        self.epsilon = 1 / math.sqrt(t + 1)
            
        # force to keep epsilon greater than 0.1
        if self.epsilon < 0.1:
            self.epsilon = 0.1

        max_val = np.max(self.values)
        max_idxs = np.where(self.values == max_val)[0]

        if ran_num < 1 - self.epsilon:
            idx = np.random.choice(max_idxs, 1)[0]
        else:
            if len(max_idxs) == len(self.values):
                idx = np.random.choice(max_idxs, 1)[0]
            else:
                temp = np.arange(len(self.values))
                temp = np.setdiff1d(temp, max_idxs)
                idx = np.random.choice(temp, 1)[0]
        
        idx = np.asscalar(idx)
        return idx

    def update(self, arm_index, curr_acc, opt):

        if 'exec_time' in opt:            
            self.cum_exec_time += opt['exec_time']        

        acc = self.scale_acc(curr_acc)
        reward = (acc - self.values[arm_index])
        if self.counts[arm_index] > 0:  
            reward = reward / self.counts[arm_index]                
        self.counts[arm_index] += 1
        self.values[arm_index] += reward
    
    def scale_acc(self, acc):
        if self.reward_scaling is None:
            return acc
        elif self.reward_scaling == "LOG_ERR":
            
            # XXX: truncate extrapolated estimation
            if acc < 0:
                acc = 0.00001
            if acc > 1.0:
                acc = 0.99999

            # scaling log with error
            err = 1.0 - acc
            abs_log_err = math.fabs(math.log10(err))
            return abs_log_err
        else:
            raise TypeError('unsupported reward scaling:{}'.format(self.reward_scaling))
   