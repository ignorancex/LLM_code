import random
import copy as cp

from xoa.commons.logger import *

import numpy as np
from itertools import combinations
from scipy.stats import rankdata
from scipy.stats import kendalltau

from .div import DiversificationStrategy


class DisparityCheckStrategy(DiversificationStrategy):
    
    def __init__(self, arms, 
                values, counts, 
                choosers, 
                title="",
                check_iters=0):
        
        if check_iters == 0:
            check_iters = len(arms)

        name = "DISP-CHECK{}".format(check_iters)
        if title != "":
            name = name + "_" + title
        
        super(DisparityCheckStrategy, self).__init__(name, len(arms), values, counts, 0.0)
        self.arms = arms
        self.valid_arm_indices = [ i for i in range(len(arms))]
        self.check_iters = check_iters
        self.choosers = choosers

    def nominate(self, top_k=None):       
        all_nominees = []
        
        for i in range(len(self.arms)):
            arm = self.arms[i]
            optimizer = arm['model']
            aquisition_func = arm['acq_func']
            chooser = self.choosers[optimizer]
            
            _, estimate = chooser.next(aquisition_func) 
            if estimate is not None and 'acq_funcs' in estimate:
                cand_indices = estimate['candidates']
                ranks = rankdata(estimate['acq_funcs'], method='min')
                
                if top_k and top_k < len(cand_indices):
                    # TODO: consider only top-k ranks
                    pass

                all_nominees.append({
                    "arm_index": i,
                    "name": '{}-{}'.format(optimizer, aquisition_func),
                    "candidates": cand_indices,
                    "ranks" : ranks
                })
        
        return all_nominees

    def measure_correlation(self, nominees1, nominees2, metric='kendall', alpha=0.05):
        coef = 0.0 # default coefficient
        if metric != 'kendall':
            raise NotImplementedError('correlation metric not supported: {}'.format(metric))
        else:
            ranks1 = nominees1["ranks"]
            ranks2 = nominees2["ranks"]
            coef, p = kendalltau(ranks1, ranks2)
            if p < alpha:
                #debug('{} and {} are correlated with the value {}'.format(
                #    nominees1['name'], nominees2['name'], coef))
                return coef
        return coef

    def next(self, step):
            
        if step >= self.check_iters:
            s_t = time.time()

            nominees = self.nominate()
            if len(nominees) > 1:
                self.valid_arm_indices = self.configure_arms(nominees)
                remained_arms = []
                for i in self.valid_arm_indices:
                    arm = self.arms[i]
                    remained_arms.append('{}-{}'.format(arm['model'], arm['acq_func']))
                
                self.check_iters = step + len(remained_arms)
                dur = time.time() - s_t
                debug('Diversity measure takes {:.2f} sec: {}'.format(dur, remained_arms))
                 

        i = step % len(self.valid_arm_indices)
        arm_index = self.valid_arm_indices[i]

        return arm_index

    def configure_arms(self, nominees):
        valid_arm_indices = [ i for i in range(len(self.arms)) ]
        sim_combi = []
        disp_combi = []
        combi = list(combinations(nominees, 2))
        for c in combi:
            coef = self.measure_correlation(c[0], c[1])
            if coef >= 0.65:
                debug('{} & {} show strong agreement({:0.4f}).'.format(c[0]['name'], c[1]['name'], coef))
                sim_combi.append(c)
                                        
            elif coef <= -0.65:
                debug('{} & {} show strong disagreement({:0.4f}).'.format(c[0]['name'], c[1]['name'], coef))
                disp_combi.append(c)
                
        # drop a item randomly in the similar combinations
        for s in sim_combi:
            c = random.randint(0, 1)                
            k = s[c]["arm_index"]
            if len(valid_arm_indices) > 1:
                if k in valid_arm_indices:
                    valid_arm_indices.remove(k)
                    #debug('Arm# {} removed'.format(k))

        # append disagreed arms to the list
        for d in disp_combi:
            for j in range(2):
                a = d[j]["arm_index"]
                if not a in valid_arm_indices:
                    valid_arm_indices.append(a)
                    #debug('Arm# {} appended'.format(a))
        #debug('The remained arms: {}'.format(valid_arm_indices))
        return valid_arm_indices    

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] += 1
        for i in range(len(self.values)):
            self.values[i] = float(self.counts[i]) / float(sum(self.counts))