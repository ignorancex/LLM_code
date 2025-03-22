from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import copy
import numpy as np

from xoa.commons.logger import *
from trainers.emul.nas_trainer import NAS101Emulator, NAS201Emulator


class NAS101ETREmulator(NAS101Emulator):

    def __init__(self, builder, space, survive_ratio, epoch_budgets, 
                worst_err=0.9, report_mean_test_acc=False):

        if survive_ratio < 0.0 or survive_ratio > 0.5:
            raise ValueError("Invalid survive_ratio: {}".format(survive_ratio))

        self.threshold_percentile = 100.0 - (survive_ratio * 100.0)
        self.epoch_budgets = epoch_budgets
        self.eval_start_index = 0
        self.eval_end_index = 2 # XXX: use of 36 epochs

        super(NAS101ETREmulator, self).__init__(builder, space, epoch_budgets[0], worst_err, report_mean_test_acc)

    def initialize(self):
        super(NAS101ETREmulator, self).initialize()        

    def get_acc_threshold(self, cur_acc_curve):
        mean_accs = []   
        if len(cur_acc_curve) > 0:
            cur_mean_acc = np.mean(cur_acc_curve)
            if np.isnan(cur_mean_acc) == False:
                mean_accs.append(cur_mean_acc)

        for i in range(len(self.history)):
            acc_curve_span = self.get_acc_curve(i, self.eval_start_index, self.eval_end_index)
            if len(acc_curve_span) > 0:
                mean_acc = np.mean(acc_curve_span)
                if np.isnan(mean_acc) == False:
                    mean_accs.append(mean_acc)
        
        if len(mean_accs) > 0:
            threshold = np.percentile(mean_accs, self.threshold_percentile)
        else:
            threshold = 0.0
        return threshold

    def train(self, model_index, train_epoch=None):
        
        if train_epoch == None:
            train_epoch = self.epoch_budgets[-1]
        
        val_acc_curve = []
        test_acc_curve = []
        train_times = []
        
        for b in self.epoch_budgets:
            result = super(NAS101ETREmulator, self).train(model_index, b)
            if result["train_failed"] :
                return result
            
            val_acc_curve.append(result['valid_accuracy'])
            test_acc_curve.append(result['test_accuracy'])            
            train_times.append(result['exec_time']) 
        
        #debug("{}: commencing iteration {}".format(type(self).__name__, len(self.history)))
        
        eval_epoch = self.epoch_budgets[self.eval_end_index]
        p_curve = val_acc_curve[self.eval_start_index:self.eval_end_index+1]
        threshold = self.get_acc_threshold(p_curve)
        cur_val_acc = val_acc_curve[self.eval_end_index]
        cur_test_acc = test_acc_curve[self.eval_end_index]
        debug("Checking current accuracy over the threshold: {:.4f} / {:.4f}".format(cur_val_acc, threshold))
        if cur_val_acc < threshold:
            debug("config #{} is terminated at epoch {} ({:.4f} > {:.4f} asymptote: {:.4f})".format(
                model_index, eval_epoch, threshold, cur_val_acc, max(val_acc_curve)))
            
            val_acc_curve = p_curve
            val_loss = 1.0 - cur_val_acc
            test_loss = 1.0 - cur_test_acc
            train_time = train_times[self.eval_end_index]
            train_epoch = eval_epoch
            early_terminated = True
        else:
            val_loss = 1.0 - val_acc_curve[-1]
            test_loss = 1.0 - test_acc_curve[-1]
            train_time = train_times[-1]
            early_terminated = False
        #debug("Early terminated: {}".format(early_terminated))
        self.add_train_history(val_acc_curve, train_time, eval_epoch, measure='valid_accuracy')
        debug("Training the configuration #{} with {} epochs ({:.0f}s) -> {:.4f}".format(model_index, train_epoch, train_time, 1.0 - test_loss))
        return {
                "test_error": test_loss,
                "test_accuracy": 1.0 - test_loss,
                "valid_error": val_loss,
                "valid_accuracy": 1.0 - val_loss,                
                "exec_time" : train_time, 
                "train_epoch": train_epoch,
                "train_failed": False,                 
                'early_terminated' : early_terminated
        }



class NAS201ETREmulator(NAS201Emulator):
    
    def __init__(self, builder, space, survive_ratio, epoch_budgets, 
                worst_err=0.9, report_mean_test_acc=False):

        if survive_ratio < 0.0 or survive_ratio > 0.5:
            raise ValueError("Invalid survive_ratio: {}".format(survive_ratio))

        self.threshold_percentile = 100.0 - (survive_ratio * 100.0)
        self.epoch_budgets = sorted(epoch_budgets)
        
        max_epoch = epoch_budgets[-1]
        checkpoint1 = int(max_epoch * 0.5)
        checkpoint2 = int(max_epoch * self.threshold_percentile / 100.0)
        if not checkpoint1 in self.epoch_budgets:
            self.epoch_budgets.append(checkpoint1)
        if not checkpoint2 in self.epoch_budgets:
            self.epoch_budgets.append(checkpoint2)        
        sorted(self.epoch_budgets)     
        
        self.eval_start_index = 0
        self.eval_mid_index = self.epoch_budgets.index(checkpoint1)
        self.eval_end_index = self.epoch_budgets.index(checkpoint2) 

        super(NAS201ETREmulator, self).__init__(builder, space, epoch_budgets[0], worst_err, report_mean_test_acc)

    def initialize(self):
        super(NAS201ETREmulator, self).initialize()        

    def get_acc_threshold(self, cur_acc_curve, checkpoint):
        mean_accs = []   
        if len(cur_acc_curve) > 0:
            cur_mean_acc = np.mean(cur_acc_curve)
            if np.isnan(cur_mean_acc) == False:
                mean_accs.append(cur_mean_acc)

        eval_start_index = self.eval_start_index
        eval_end_index = self.eval_end_index
        if checkpoint == 'checkpoint1':
            eval_end_index = self.eval_mid_index
        elif checkpoint == 'checkpoint2':
            eval_start_index = self.eval_mid_index
            eval_end_index = self.eval_end_index
        else:
            raise ValueError('Invalid checkpoint: {}'.format(checkpoint))

        for i in range(len(self.history)):
            acc_curve_span = self.get_acc_curve(i, eval_start_index, eval_end_index)
            if len(acc_curve_span) > 0:
                mean_acc = np.mean(acc_curve_span)
                if np.isnan(mean_acc) == False:
                    mean_accs.append(mean_acc)
        
        if len(mean_accs) > 0:
            threshold = np.percentile(mean_accs, self.threshold_percentile)
        else:
            threshold = 0.0
        return threshold

    def train(self, model_index, train_epoch=None):
        
        if train_epoch == None:
            train_epoch = self.epoch_budgets[-1]
        
        val_acc_curve = []
        test_acc_curve = []
        train_times = []
        
        for b in self.epoch_budgets:
            result = super(NAS201ETREmulator, self).train(model_index, b)
            if result["train_failed"] :
                return result
            
            val_acc_curve.append(result['valid_accuracy'])
            test_acc_curve.append(result['test_accuracy'])            
            train_times.append(result['exec_time']) 
        
        #debug("{}: commencing iteration {}".format(type(self).__name__, len(self.history)))
        
        eval_epoch = self.epoch_budgets[self.eval_mid_index]
        p_curve = val_acc_curve[self.eval_start_index:self.eval_mid_index+1]
        threshold = self.get_acc_threshold(p_curve, 'checkpoint1')
        cur_val_acc = val_acc_curve[self.eval_mid_index]
        cur_test_acc = test_acc_curve[self.eval_mid_index]
        #debug("Checkpoint-1: {:.4f} / {:.4f}".format(cur_val_acc, threshold))
        if cur_val_acc < threshold:
            debug("config #{} is early-terminated at epoch {} ({:.4f} > {:.4f} asymptote: {:.4f})".format(
                model_index, eval_epoch, threshold, cur_val_acc, max(val_acc_curve)))
            val_acc_curve = p_curve
            val_loss = 1.0 - cur_val_acc
            test_loss = 1.0 - cur_test_acc
            train_time = train_times[self.eval_mid_index]
            train_epoch = eval_epoch
            early_terminated = True
        else:
            eval_epoch = self.epoch_budgets[self.eval_end_index]
            p_curve = val_acc_curve[self.eval_mid_index:self.eval_end_index+1]
            threshold = self.get_acc_threshold(p_curve, 'checkpoint1')
            cur_val_acc = val_acc_curve[self.eval_end_index]
            cur_test_acc = test_acc_curve[self.eval_end_index]
            #debug("Checkpoint-2: {:.4f} / {:.4f}".format(cur_val_acc, threshold))
            if cur_val_acc < threshold:
                debug("config #{} is lately-terminated at epoch {} ({:.4f} > {:.4f} asymptote: {:.4f})".format(
                    model_index, eval_epoch, threshold, cur_val_acc, max(val_acc_curve)))
                val_acc_curve = p_curve
                val_loss = 1.0 - cur_val_acc
                test_loss = 1.0 - cur_test_acc
                train_time = train_times[self.eval_end_index]
                train_epoch = eval_epoch
                early_terminated = True
            else:
                val_loss = 1.0 - val_acc_curve[-1]
                test_loss = 1.0 - test_acc_curve[-1]
                train_time = train_times[-1]
                early_terminated = False
        
        self.add_train_history(val_acc_curve, train_time, eval_epoch, measure='valid_accuracy')
        debug("Training the configuration #{} with {} epochs ({:.0f}s) -> {:.4f}".format(model_index, train_epoch, train_time, 1.0 - test_loss))
        return {
                "test_error": test_loss,
                "test_accuracy": 1.0 - test_loss,
                "valid_error": val_loss,
                "valid_accuracy": 1.0 - val_loss,                
                "exec_time" : train_time, 
                "train_epoch": train_epoch,
                "train_failed": False,                 
                'early_terminated' : early_terminated
        }