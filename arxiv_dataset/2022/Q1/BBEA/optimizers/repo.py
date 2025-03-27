from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from xoa.commons.logger import *


class ResultsRepository(object):
    def __init__(self, goal_metric, report_type='test'):
       
        self.result = {}
        self.goal_metric = goal_metric

        self.result['error'] = []
        if goal_metric == 'accuracy':
            self.result['accuracy'] = []
        #self.result['metrics'] = []

        self.report_type = report_type
        
        # list of the estimated execution time of a candidate
        #self.result['est_exec_time'] = []
        self.result['exec_time'] = []
        self.result['opt_time'] = []

        self.result['model_idx'] = []  # for tracing selected candidates
        self.result['select_trace'] = []  # for tracing arm selection
        self.result['train_epoch'] = []

        self.result['num_duplicates'] = []
        #self.result['force_terminate'] = False # whether it found a goal or not

        #debug("result initialized")

    def append(self, eval_result):
        if not 'test_error' in eval_result and not 'valid_error' in eval_result:
            warn("Invalid evaluation result: {}".format(eval_result))
            return

        for k in eval_result.keys():
            v = eval_result[k]

            if k in self.result:
                self.result[k].append(v)
            elif self.report_type == 'test' and k == 'test_error':
                self.result['error'].append(v)
            elif self.report_type == 'validation' and k == 'valid_error':
                self.result['error'].append(v)
        
        if not 'test_accuracy' in eval_result and self.goal_metric == "accuracy":

            if self.report_type == 'test' and 'test_error' in eval_result:
                test_acc = 1.0 - eval_result['test_error'] 
                self.result['accuracy'].append(test_acc)                
            elif self.report_type == 'validation' and 'valid_error' in eval_result:
                valid_acc = 1.0 - eval_result['valid_error'] 
                self.result['accuracy'].append(valid_acc)                

    def count_duplicates(self, shelves):
        selects = []
        for s in shelves:
            selects.append(s['model_idx'])
        num_duplicate = len(selects) - len(set(selects))

        self.result['num_duplicates'].append(num_duplicate)

    def update_trace(self, optimizer, acquistion_func):        
        self.result['select_trace'].append(optimizer + '_' + acquistion_func)

    def save_select_status(self, values, counts):        
        self.result['mean_rewards'] = values
        self.result['selected_counts'] = counts

    def get_elapsed_time(self):        
        elapsed_time = 0
        if len(self.result['exec_time']) > 0:
            elapsed_time += sum(self.result['exec_time'])
        if len(self.result['opt_time']) > 0:
            elapsed_time += sum(self.result['opt_time'])

        return elapsed_time 

    def get_current_status(self):
        return self.result

    def get_values(self, property):
        if property in self.result:
            return self.result[property]
        else:
            return None

    def get_value(self, property, index):
        if property in self.result:
            if len(self.result[property]) != 0:
                if len(self.result[property]) > index:
                    return self.result[property][index]
        return None

    def get_total_duration(self, index):
        time = self.result['opt_time'][index] + self.result['exec_time'][index]
        return time


class BatchResultsRepository(ResultsRepository):
    def __init__(self, report_type='test'):
        return super(BatchResultsRepository, self).__init__('accuracy', report_type)

    def update_batch_result(self, bandits):
        
        self.result['model_idx'] = [ b['local_result'].get_values('model_idx') for b in bandits ]
        self.result['error'] = [ b['local_result'].get_values('error') for b in bandits ]
        self.result['accuracy'] = [ b['local_result'].get_values('accuracy') for b in bandits ]

        #self.result['est_exec_time'] = [ b['local_result'].get_values('est_exec_time') for b in bandits ]
        self.result['exec_time'] = [ b['local_result'].get_values('exec_time') for b in bandits ]
        self.result['opt_time'] = [ b['local_result'].get_values('opt_time') for b in bandits ]

        self.result['iters'] = [ b['cur_iters'] for b in bandits ]
        self.result['num_duplicates'] = [ b['num_duplicates'] for b in bandits ]
        self.result['select_trace'] = [ b['local_result'].get_values('select_trace') for b in bandits ]      
