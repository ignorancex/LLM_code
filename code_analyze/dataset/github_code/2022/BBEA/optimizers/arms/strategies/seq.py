from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from xoa.commons.logger import *
from .div import DiversificationStrategy

import numpy as np
from scipy.stats import rankdata


class SequentialStrategy(DiversificationStrategy):
    def __init__(self, num_arms, values, counts, title):
        name = "SEQ_" + title
        super(SequentialStrategy, self).__init__(name, num_arms, values, counts)

    def next(self, step):
        idx = step % self.num_arms
        
        return idx

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] += 1
        for i in range(len(self.values)):
            self.values[i] = float(self.counts[i]) / float(sum(self.counts))

class SequentialKnockOutStrategy(DiversificationStrategy):
    def __init__(self, num_arms, values, counts, iters_round, title):
        name = "SKO_" + title
        self.iters_round = iters_round                

        super(SequentialKnockOutStrategy, self).__init__(name, num_arms, values, counts)

        self.remain_arms = [ i for i in range(num_arms) ]
        self.min_remains = 2 # XXX: minimum remains
        
        init_prob = 1.0 / float(num_arms)
        for i in range(len(self.values)):
            self.values[i] = init_prob
 
        self.reset()

    def reset(self):
        self.prev_arm_values = [ 0.0 for i in range(self.num_arms) ]
        self.rank_sums = np.array([ 0.0 for i in range(self.num_arms)])
        self.cur_index = 0            

    def next(self, step):
        idx = 0
        if step != 0 and step % self.iters_round == 0:
            # remove the worst performed arm            
            try:
                i_worst = np.argmax(self.rank_sums)

                if len(self.remain_arms) > self.min_remains:
                    info('Arm #{} will be eliminated in {}'.format(i_worst, self.remain_arms) )
                    self.remain_arms.remove(i_worst)
                    prob = 1.0 / float(len(self.remain_arms))
                    for i in range(len(self.values)):
                        if i in self.remain_arms:
                            self.values[i] = prob
                        else:
                            self.values[i] = 0.0
                else:
                    debug('the number of remained arms is {}.'.format(len(self.remain_arms)))
            except:
                warn("no {} in {}".format(i_worst, self.remain_arms))
            finally:
                self.reset()
        else:
            if self.cur_index < len(self.remain_arms):
                idx = self.remain_arms[self.cur_index]                
                self.cur_index += 1
            else:
                self.cur_index = 0
                idx = self.remain_arms[self.cur_index]            

        return idx

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] = self.counts[arm_index] + 1
        self.prev_arm_values[arm_index] = curr_acc

        # calculate rank
        cur_ranks = rankdata(self.prev_arm_values)
        self.rank_sums += cur_ranks

