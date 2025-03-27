from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np


from xoa.commons.logger import *
from .div import DiversificationStrategy


class RandomStrategy(DiversificationStrategy):
    def __init__(self, num_arms, values, counts, title):
        name = "RANDOM_" + title
        super(RandomStrategy, self).__init__(name, num_arms, values, counts, 1.0)

    def next(self, step):
        idx = np.random.randint(0, self.num_arms)
        
        return idx

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] += 1
        for i in range(len(self.values)):
            self.values[i] = float(self.counts[i]) / float(sum(self.counts))  
