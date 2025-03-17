from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lookup.dnnbench import *

import os
import time
import copy
import numpy as np
import traceback

from xoa.commons.logger import *
from trainers.emul.fcnet_trainer import TabularFCNetTrainEmulator


class TabularFCNetETREmulator(TabularFCNetTrainEmulator):

    def __init__(self, space, data_path, survive_ratio,  
                worst_err=0.9, config_type='FCNetProteinStructureBenchmark'):

        if survive_ratio < 0.0 or survive_ratio > 0.5:
            raise ValueError("Invalid survive_ratio: {}".format(survive_ratio))

        self.threshold_percentile = 100.0 - (survive_ratio * 100.0)
        self.acc_scale = 10.0
        self.etr_checked = False

        super(TabularFCNetETREmulator, self).__init__(space, data_path, 10, worst_err, config_type)
        
        self.survive_ratio = survive_ratio
        self.early_drop_percentile = (survive_ratio * 100.0)
        self.late_drop_percentile = 100 - (survive_ratio * 100.0)
        self.num_epochs = 100
        self.early_drop_epoch = int(self.num_epochs * 0.5)
        self.survive_check_epoch = int(self.num_epochs * (1.0 - self.survive_ratio))

    def initialize(self):
        super(TabularFCNetETREmulator, self).initialize()        

    def get_eval_indices(self, eval_start_ratio, eval_end_ratio):
        start_index = int(self.num_epochs * eval_start_ratio)
        if start_index > 0:
            start_index -= 1
        
        eval_start_index = start_index
        eval_end_index = int(self.num_epochs * eval_end_ratio) - 1
        return eval_start_index, eval_end_index

    def get_acc_threshold(self, cur_acc_curve, 
                        eval_start_index, eval_end_index, percentile):
        mean_accs = []
        cur_acc_curve = cur_acc_curve[eval_start_index:eval_end_index+1]   
        if len(cur_acc_curve) > 0:
            cur_mean_acc = np.mean(cur_acc_curve)
            if np.isnan(cur_mean_acc) == False:
                mean_accs.append(cur_mean_acc)

        acc_curves = self.get_acc_curves()
        for prev_curve in acc_curves:
            acc_curve_span = []
            
            if len(prev_curve) > eval_end_index:
                acc_curve_span = prev_curve[eval_start_index:eval_end_index+1]
            
            if len(acc_curve_span) > 0:
                mean_acc = np.mean(acc_curve_span)
                if np.isnan(mean_acc) == False:
                    mean_accs.append(mean_acc)
        
        if len(mean_accs) > 0:
            threshold = np.percentile(mean_accs, percentile)
        else:
            threshold = 0.0

        #debug("P:{}%, T:{:.4f}, mean accs:{}".format(percentile, threshold, ["{:.4f}".format(acc) for acc in mean_accs]))
        return threshold

    def set_acc_scale(self, loss_curve):        
        max_loss = max(loss_curve)
        if self.acc_scale < max_loss:
            if self.acc_scale > 1.0:
                warn("Scaling factor to transform loss to accuracy has to be set again due to {}".format(max_loss))
            debug("Scaling to transform loss to accuracy properly.")
            while self.acc_scale < max_loss:
                self.acc_scale = 10 * self.acc_scale
            debug("Current accuracy scale: {}".format(self.acc_scale))

    def flip_curve(self, loss_curve):
        acc_curve = []
        prev_acc = None
        self.set_acc_scale(loss_curve)
        for loss in loss_curve:
            if loss != None:
                acc = float(self.acc_scale - loss) / float(self.acc_scale)
                prev_acc = acc
            else:
                if prev_acc == None:
                    acc = 0.0
                else:
                    acc = prev_acc
            acc_curve.append(acc)
        return acc_curve 

    def get_acc_curves(self):
        acc_curves = []
        for i in range(len(self.history)):
            curve = self.history[i]["curve"]
            if not 'accuracy' in self.history[i]['measure']:
                acc_curve = self.flip_curve(curve)
            else:
                acc_curve = curve
            acc_curves.append(acc_curve)
        return acc_curves

    def stop_check(self, acc_curve, cur_epoch):
        if len(acc_curve) < self.early_drop_epoch:
            return False # no ETR rule applied due to short epochs learning
        
        if self.etr_checked == False:
            if cur_epoch >= self.early_drop_epoch and cur_epoch < self.survive_check_epoch:
                # evaluate early termination criteria
                start_index, end_index = self.get_eval_indices(0.0, 0.5)
                cur_acc = acc_curve[end_index]
                
                acc_thres = self.get_acc_threshold(acc_curve, start_index, end_index, self.early_drop_percentile)
                debug("Termination check at {} epoch: {:.6f} > 1.0".format(cur_epoch, cur_acc/acc_thres))
                if cur_acc < acc_thres:
                    debug("Evaluation is dropped in the early checkpoint.")                  
                    return True
                else:
                    self.etr_checked = "early"
        elif self.etr_checked == "early":
            if cur_epoch >= self.survive_check_epoch:
                # evaluate late survival criteria
                eval_end_ratio = 1.0 - self.survive_ratio
                start_index, end_index = self.get_eval_indices(0.5, eval_end_ratio)
                cur_acc = acc_curve[end_index]
                acc_thres = self.get_acc_threshold(acc_curve, start_index, end_index, self.late_drop_percentile)
                debug("Termination check at {} epoch: {:.6f} > 1.0".format(cur_epoch, cur_acc/acc_thres))
                if cur_acc < acc_thres:
                    debug("Evaluation is dropped in the late checkpoint.") 
                    return True
                else:
                    self.etr_checked = True
                    return False
        else:
            return False

    def train(self, model_index, train_epoch=None, no_etr=False):
        
        if train_epoch == None:
            train_epoch = self.num_epochs
        
        if no_etr == True:
            return super(TabularFCNetETREmulator, self).train(model_index, train_epoch)
        
        try:
            self.etr_checked = False
            early_terminated = False
            h = self.space.get_hpv_dict(model_index)
            b = self.benchmark(data_dir=self.data_dir)

            test_err, err_curve, rts = b.objective_function_learning_curve(h, budget=train_epoch)
            #debug("Valid MSE curve: {}, Runtimes: {}".format(err_curve, rts))
            err = test_err
            acc_curve = self.flip_curve(err_curve)
            if self.stop_check(acc_curve, self.early_drop_epoch):
                err = err_curve[self.early_drop_epoch-1]
                test_err = err # XXX: treated early stopped error as test error 
                err_curve = err_curve[:self.early_drop_epoch]
                rt = rts[self.early_drop_epoch-1]
                train_epoch = self.early_drop_epoch
                early_terminated = True
                #debug("Early dropped at {}: {}".format(self.early_drop_epoch, err))
            elif self.stop_check(acc_curve, self.survive_check_epoch):
                err = err_curve[self.survive_check_epoch-1]
                test_err = err # XXX: treated early stopped error as test error
                err_curve = err_curve[:self.survive_check_epoch]
                rt = rts[self.survive_check_epoch-1]
                train_epoch = self.survive_check_epoch
                early_terminated = True
                #debug("Not finally survived at {}: {}".format(self.survive_check_epoch, err))
            else:
                err = err_curve[-1]
                rt = rts[-1]
                #debug("Survivor performance: {}, runtime: {}".format(err, rt))

            self.add_train_history(err_curve, 
                                    rt, 
                                    train_epoch,
                                    measure='loss')
            
            if train_epoch != self.num_epochs:
                test_err = err # XXX: use of interim valid error as test performance

            return {
                    "valid_error": float(err), 
                    "test_error": float(test_err), 
                    "exec_time" : float(rt),
                    "train_epoch" : train_epoch, 
                    'early_terminated' : early_terminated
            }             

        except Exception as ex:
            warn("Training failed: {}".format(ex))
            debug(traceback.format_exc())        
