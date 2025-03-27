from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import traceback

from xoa.commons.logger import *
from trainers.proto import TrainerPrototype

from lookup.hpobench import *


class TabularFCNetTrainEmulator(TrainerPrototype):

    def __init__(self, space, data_path, min_epoch,
                worst_err=100.0,
                config_type='FCNetProteinStructureBenchmark'):
        
        self.space = space
        self.data_dir = data_path
        self.min_epoch = min_epoch
        self.num_epochs = 100
        self.worst_err = worst_err

        try:
            self.benchmark = eval(config_type)
            self.config_type = config_type
        except Exception as ex:
            raise ValueError("{} benchmark: {}".format(config_type, ex))

        super(TabularFCNetTrainEmulator, self).__init__()

    def get_min_train_epoch(self):
        return self.min_epoch

    def get_interim_error(self, model_index, cur_dur=0):

        if cur_dur > 0:
            cur_epoch = self.min_epoch 
            ret = self.train(model_index, cur_epoch) 
            error = ret['valid_error']
        else:
            cur_epoch = 0
            error = self.worst_err 
        
        return error, cur_epoch

    def train(self, model_index, train_epoch=None):
        try:
            if train_epoch == None:
                train_epoch = self.num_epochs

            h = self.space.get_hpv_dict(model_index)
            b = self.benchmark(data_dir=self.data_dir)

            test, valid, rt = b.objective_function(h, budget=train_epoch)

            #debug("Final test: {}, Valid MSE: {}, Runtime: {}".format(test, valid, rt))
            if train_epoch != self.num_epochs:
                test = valid # XXX: use of interim valid error as test performance

            
            return {
                    "valid_error": float(valid), 
                    "test_error": float(test), 
                    "exec_time" : float(rt),
                    "train_epoch" : train_epoch, 
                    'early_terminated' : False
            }             

        except Exception as ex:
            warn("Training failed: {}".format(ex))
            debug(traceback.format_exc())