from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from xoa.commons.logger import *


class DiversificationStrategy(object):
    def __init__(self, name, num_arms, values, counts, epsilon=0.0):
        self.name = name
        self.num_arms = num_arms
        self.values = values
        self.counts = counts

        self.epsilon = epsilon           
        
    def get_random_ratio(self):
        return self.epsilon

    def next(self, step):
        raise NotImplementedError("This method should be overloaded properly.")

    def update(self, arm_index, curr_acc, opt=None):
        raise NotImplementedError("This method should be overloaded properly.")

