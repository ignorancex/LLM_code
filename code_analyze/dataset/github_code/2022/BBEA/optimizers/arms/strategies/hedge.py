from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import random
import copy as cp

from xoa.commons.logger import *

import numpy as np


from .div import DiversificationStrategy

MAX_ERROR = 100.0

class ClassicHedgeStrategy(DiversificationStrategy):
    '''
    algorithm code referred from:
    https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/hedge/hedge.py
    '''
    def __init__(self, arms, temperature, values, counts, title=""):
        name = "HEDGE"
        if title != "":
            name = name + "_" + title
        
        super(ClassicHedgeStrategy, self).__init__(name, num_arms, values, counts, 0.0)
        self.temperature = temperature

    def categorical_draw(self, probs):
        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            prob = probs[i]
            cum_prob += prob
            if cum_prob > z:
                return i
        raise ValueError("unrealistic status.")

    def next(self, step):
        z = sum([math.exp(v * self.temperature) for v in self.values])
        probs = [math.exp(v * self.temperature) / z for v in self.values]
        return self.categorical_draw(probs)

    def update(self, chosen_arm, curr_acc, opt):
        reward = curr_acc # TODO:reward design required              
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1                
        value = self.values[chosen_arm]

        self.values[chosen_arm] = value + reward                


class BayesianHedgeStrategy(DiversificationStrategy):
    ''' An extension of GP-Hedge algorthm'''
    def __init__(self, arms, temperature, values, counts, 
                    s_space, choosers, 
                    title="", 
                    unbiased_estimation=False,
                    reward_scaling=None):
        name = "BO-HEDGE"
        if title != "":
            name = name + "_" + title
        
        super(BayesianHedgeStrategy, self).__init__(name, len(arms), values, counts, 0.0)
        
        self.temperature = temperature
        self.search_space = s_space
        self.choosers = choosers
        self.arms = arms
        self.nominees = None
        self.unbiased_estimation = unbiased_estimation
        self.reward_scaling = reward_scaling

    def categorical_draw(self, probs):
        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            prob = probs[i]
            cum_prob += prob
            if cum_prob > z:
                return i
        raise ValueError("unrealistic status.")

    def nominate(self):       
        all_nominees = []
        
        for arm in self.arms:
            
            optimizer = arm['model']
            aquisition_func = arm['acq_func']
            chooser = self.choosers[optimizer]
            # default error value.
            est_value = MAX_ERROR
            test_error = MAX_ERROR
            
            next_index, estimate = chooser.next(aquisition_func) 
            if estimate is not None:
                i = estimate['means'].index(next_index)
                est_value = estimate['means'][i]

                try:            
                    test_error = self.search_space.get_errors(next_index)
                except Exception as ex:
                    warn('No record in the search history: {}'.format(next_index))

            all_nominees.append({
                "optimizer": optimizer,
                "aquisition_func" : aquisition_func,
                "best_index" : next_index,
                "true_err" : test_error,
                "est_err" : est_value
            })
        return all_nominees

    def next(self, step):
        self.nominees = self.nominate()
        arm_index = 0
        try:       
            z = sum([math.exp(v * self.temperature) for v in self.values])
            probs = [round(math.exp(v * self.temperature) / z, 3) for v in self.values]
            debug('probability:{}'.format(probs))
            arm_index = self.categorical_draw(probs)
        except Exception as ex:
            warn("Exception on hedge: {}".format(ex))            
            arm_index = random.randrange(len(probs))
            debug("Uniform random select: {}".format(arm_index))
        
        return arm_index

    def update(self, arm_index, curr_acc, opt=None):
        
        self.counts[arm_index] = self.counts[arm_index] + 1 

        if self.unbiased_estimation is False:
            for n in range(len(self.nominees)):
                selected_nominee = self.nominees[n]
                est_err = selected_nominee['est_err']
                value = self.values[n]
                self.values[n] = value + self.scale_reward(est_err)
        else:
            for n in range(len(self.nominees)):
                selected_nominee = self.nominees[n]
                err = selected_nominee['true_err']
                value = self.values[n]
                self.values[n] = value + self.scale_reward(err)            
    
    def scale_reward(self, err):
        if self.reward_scaling is None:
            return -1.0 * err
        elif self.reward_scaling == "LOG_ERR":


            # scaling log with error

            abs_log_err = math.fabs(math.log10(err))
            return -1.0 * abs_log_err
        else:
            raise TypeError('unsupported reward scaling:{}'.format(self.reward_scaling))
  
