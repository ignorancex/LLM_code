from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from xoa.commons import *
from optimizers.arms.strategies import *


class ArmSelector(object):
    def __init__(self, spec, config, space, choosers):
        
        self.num_skip = 0
        self.configure_arms(config)                  
        self.reset()                             
        self.configure_strategy(spec, config, space, choosers)

    def reset(self):
        self.cur_arm_index = 0
        self.values = []
        self.counts = []

        for arm in range(self.num_arms):
            self.values.append(0.0)
            self.counts.append(0)  

    def get_stats(self):
        return self.values, self.counts

    def update_reward(self, step, cur_acc, optional):
        '''reward update'''
        if step >= self.num_skip:
            self.strategy.update(self.cur_arm_index, cur_acc, optional)
            
            debug("Arm selection ratio: {}, Current epsilon: {}".format(
                [round(v, 2) for v in self.values], 
                self.strategy.get_random_ratio()))                 

    def get_arm(self, step):
        if step < self.num_skip:
            next_index = np.random.randint(0, self.num_arms)
            debug('The arm will be selected uniformly at random before {} steps'.format(self.num_skip)) 
        else:
            next_index = self.strategy.next(step)
        arm = self.arms[next_index]
        self.cur_arm_index = next_index
        
        return arm['model'], arm['acq_func']

    def get_other_arms(self):
        others = []
        for i in range(self.num_arms):
            if i != self.cur_arm_index:
                arm = self.arms[i]
                others.append(arm)
        
        return others

    def configure_arms(self, config):
        if 'arms' in config:
            self.arms = config['arms']
        else:
            # XXX: default diversified models
            self.arms = [
                {
                    "model": "RF-HLE",
                    "acq_func": "EI"
                },
                {
                    "model": "RF-HLE",
                    "acq_func": "PI"
                },
                {
                    "model": "RF-HLE",
                    "acq_func": "UCB"
                },
                {
                    "model": "GP-HLE",
                    "acq_func": "EI"
                },
                {
                    "model": "GP-HLE",
                    "acq_func": "PI"
                },
                {
                    "model": "GP-HLE",
                    "acq_func": "UCB"
                },        
                {
                    "model": "TPE",
                    "acq_func": "EI"
                }
            ]
        self.num_arms = len(self.arms)

    def configure_strategy(self, spec, config, space, choosers):        
        
        self.spec = spec
        
        # Configure model diversification strategy
        title = ''
        if title in config: 
            title = config['title'].replace(" ","_")
        
        if spec == 'SEQ':
            self.strategy = SequentialStrategy(self.num_arms, self.values, self.counts, title)
        elif spec == 'SKO':
            num_iters_per_round = self.num_arms * 10
            self.strategy = SequentialKnockOutStrategy(self.num_arms, self.values, self.counts,
                                            num_iters_per_round, title)
        elif spec == 'RANDOM':
            self.strategy = RandomStrategy(self.num_arms, self.values, self.counts, title)
        elif spec == 'HEDGE':
            eta = 1
            if 'eta' in config.keys():
                eta = config['eta']
            self.strategy = ClassicHedgeStrategy(self.arms, eta, self.values, self.counts, 
                                        title)
        elif spec == 'BO-HEDGE':
            eta = 0.1
            if 'eta' in config.keys():
                eta = config['eta']
            self.strategy = BayesianHedgeStrategy(self.arms, eta, 
                                        self.values, self.counts, 
                                        space, choosers, 
                                        title=title)
        elif spec == 'BO-HEDGE-T':
            eta = 0.1
            if 'eta' in config.keys():
                eta = config['eta']
            self.strategy = BayesianHedgeStrategy(self.arms, eta, 
                                        self.values, self.counts, 
                                        space, choosers, 
                                        title=title,
                                        unbiased_estimation=True)
        elif spec == 'BO-HEDGE-LE':
            eta = 0.1
            if 'eta' in config.keys():
                eta = config['eta']
            self.strategy = BayesianHedgeStrategy(self.arms, eta, 
                                        self.values, self.counts, 
                                        space, choosers, 
                                        title=title,
                                        reward_scaling="LOG_ERR")
        elif spec == 'BO-HEDGE-LET':
            eta = 0.1
            if 'eta' in config.keys():
                eta = config['eta']
            self.strategy = BayesianHedgeStrategy(self.arms, eta, 
                                        self.values, self.counts, 
                                        space, choosers, 
                                        title=title,
                                        unbiased_estimation=True,
                                        reward_scaling="LOG_ERR")
        elif spec == 'DISP':
            self.strategy = DisparityCheckStrategy(self.arms,
                                        self.values, self.counts, 
                                        choosers, 
                                        title=title)
        elif spec == 'CVS':
            self.strategy = CVSelectionStrategy(self.arms,
                                        self.values, self.counts, 
                                        choosers, 
                                        title=title)
        elif spec == 'CVD':
            self.strategy = CVDropStrategy(self.arms,
                                        self.values, self.counts, 
                                        choosers, 
                                        title=title)
        elif spec == 'EG':
            init_eps = 1.0
            decay_factor = 5
            
            self.num_skip = 2 #XXX: avoiding selection bias by first two steps  

            if 'init_eps' in config.keys():
                init_eps = config['init_eps']
            if 'decay_factor' in config.keys():
                decay_factor = config['decay_factor']

            self.strategy = EpsilonGreedyStrategy(self.num_arms, self.values, self.counts, 
                        title,
                        init_eps=init_eps, 
                        decay_factor=decay_factor)
        elif spec == 'EG-LE':
            init_eps = 1.0
            decay_factor = 5            
            reward_scaling = 'LOG_ERR'
            
            self.num_skip = 2 #XXX: avoiding first two selection bias  

            if 'init_eps' in config.keys():
                init_eps = config['init_eps']
            if 'decay_factor' in config.keys():
                decay_factor = config['decay_factor']

            self.strategy = EpsilonGreedyStrategy(self.num_arms, self.values, self.counts, 
                        title,
                        init_eps=init_eps,
                        decay_factor=decay_factor, 
                        reward_scaling=reward_scaling)
        elif spec == 'GT':
            time_unit = 'H'            
            
            self.num_skip = 2 #XXX: avoiding first two selection bias  

            if 'time_unit' in config.keys():
                time_unit = config['time_unit']

            self.strategy = GreedyTimeStrategy(self.num_arms, self.values, self.counts, 
                        title,
                        time_unit=time_unit)

        elif spec == 'GT-LE':
            time_unit = 'H'
            reward_scaling = 'LOG_ERR'
            
            self.num_skip = 2 #XXX: avoiding first two selection bias  

            if 'time_unit' in config.keys():
                time_unit = config['time_unit']

            self.strategy = GreedyTimeStrategy(self.num_arms, self.values, self.counts, 
                        title,
                        time_unit=time_unit, 
                        reward_scaling=reward_scaling)                                                     
        else:
            raise ValueError("No {} strategy available!".format(spec))                        
        
