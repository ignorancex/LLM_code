from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time

import numpy as np
import traceback

from xoa.commons.logger import *
from trainers.proto import TrainerPrototype


class NAS101Emulator(TrainerPrototype):
    def __init__(self, bench, space, min_epoch, 
                worst_err=0.9, report_mean_test_acc=False):
        self.space = space
        self.min_epoch = min_epoch
        self.worst_err = worst_err
        self.report_mean_test_acc = report_mean_test_acc

        self.bench = bench
        super(NAS101Emulator, self).__init__()

    def get_min_train_epoch(self):
        return self.min_epoch

    def get_verifier(self):
        return self.bench

    def train(self, model_index, train_epoch=None):        
        start_time = time.time()
        total_time = 0.0

        is_failed = False
        test_acc = 0.0
        valid_acc = 0.0

        try:
            if train_epoch == None:
                train_epoch = 108
            
            h = self.space.get_hpv_dict(model_index)
            cell = self.bench.build(h)

            if self.bench.is_valid(cell):
                # Querying multiple times may yield different results. Each cell is evaluated 3
                # times at each epoch budget and querying will sample one randomly.
                result = self.bench.query(cell, epochs=train_epoch)
                accuracies = {}
                for k in result.keys():
                    if 'accuracy' in k:
                        accuracies[k] = result[k]
                
            else:
                raise ValueError("Architecture not found!")

            total_time = result['training_time']
            valid_acc = result['validation_accuracy']

            if train_epoch == 108 and self.report_mean_test_acc == True:
                # compute mean test error for the final budget
                _, metrics = self.bench.get_metrics_from_spec(cell)
                test_acc = np.mean([metrics[108][i]["final_test_accuracy"] for i in range(3)])
            else:
                test_acc = result['test_accuracy'] 

        except Exception as ex:
            #warn("Error on the configuration #{}: {}".format(model_index, ex))
            #debug(traceback.format_exc())
            is_failed = True
            if self.worst_err != None:
                test_acc = 1.0 - float(self.worst_err)                
            else:
                test_acc = 0.0
            valid_acc = test_acc
            total_time = time.time() - start_time
            train_epoch = 0
        finally:
            #debug("Training the configuration #{} with {} epochs ({:.0f}s) -> {:.4f}".format(model_index, train_epoch, total_time, test_acc))
            test_error = 1.0 - test_acc
            valid_error = 1.0 - valid_acc
            return {
                    "test_error": test_error,
                    "valid_error": valid_error, 
                    "test_accuracy": test_acc,
                    "valid_accuracy": valid_acc, 
                    "exec_time" : total_time,
                    "train_epoch" : train_epoch,
                    "train_failed": is_failed, 
                    'early_terminated' : False
            } 

    def get_interim_error(self, model_index, cur_dur=0):

        if cur_dur > 0:
            cur_epoch = self.min_epoch 
            ret = self.train(model_index, cur_epoch) 
            error = ret['valid_error']
        else:
            cur_epoch = 0
            error = self.worst_err 
        
        return error, cur_epoch



class NAS201Emulator(NAS101Emulator):
    def __init__(self, bench, space, min_epoch, 
                 worst_err=0.9, report_mean_test_acc=False):
        super(NAS201Emulator, self).__init__(bench, space, min_epoch, worst_err, report_mean_test_acc)

    def verify(self, cand):
        return True # XXX:every configuration is valid

    def train(self, model_index, train_epoch=None):
        start_time = time.time()
        is_failed = False
        try:
            if train_epoch == None:
                train_epoch = 200

            conf_dict = self.space.get_hpv_dict(model_index)
            #debug('Config: {}'.format(conf_dict))
            model_index = self.bench.get_arch_index(conf_dict)
            test_loss, val_loss, total_time, result = self.bench.train(model_index, train_epoch)

        except Exception as ex:
            #warn("Error on the configuration #{}: {}".format(model_index, ex))
            debug(traceback.format_exc())
            is_failed = True
            if self.worst_err != None:
                test_loss = float(self.worst_err)                
            else:
                test_loss = 1.0
            val_loss = test_loss
            total_time = time.time() - start_time
            train_epoch = 0
        finally:
            #debug("Training the configuration #{} with {} epochs ({:.0f}s) returns test error {:.4f}".format(model_index, train_epoch, total_time, test_loss))
            valid_acc = 1.0 - val_loss
            test_acc = 1.0 - test_loss 

            return {
                    "test_error": test_loss,
                    "valid_error": val_loss, 
                    "test_accuracy": test_acc,
                    "valid_accuracy": valid_acc, 
                    "exec_time" : total_time,
                    "train_epoch" : train_epoch,
                    "train_failed": is_failed, 
                    'early_terminated' : False
                } 


