
from functools import partial

import numpy as np

from sksurv.metrics import concordance_index_ipcw, integrated_brier_score, brier_score
from scipy.interpolate import interp1d
import torch.nn.functional as F

from .utils import *



def report(surv, time_points, eval_times, et_train, et_test, interpolation = False):
    all_interp1d = []
    if interpolation:
        if len(time_points.shape) == 1:

            for i in range(len(surv)):
                # if self.time_points[i].min()>c_eval_times.min():
                #     print(self.time_points[i])
                all_interp1d.append(interp1d(time_points, surv[i], axis = -1))
        else:
            
            for i in range(len(surv)):
                # if self.time_points[i].min()>c_eval_times.min():
                #     print(self.time_points[i])
                all_interp1d.append(interp1d(time_points[i], surv[i], axis = -1))
        all_interp1d = np.array(all_interp1d)
    
    
    c_indices = []
    brier_indices = []
    for eval_time in eval_times:
        if interpolation:
            surv_at_time = np.apply_along_axis(lambda x: x[0](eval_time), 1, all_interp1d.reshape(-1,1))
        else:
            index = index_at_time(eval_time, time_points)
            surv_at_time = np.take_along_axis(surv, index.reshape(-1,1), axis = -1).reshape(-1)
            # surv_at_time = surv.gather(-1, torch.tensor(index).reshape(-1,1)).reshape(-1)
        # print(surv_at_time.shape, et_train['t'].shape, et_test['t'].shape, eval_time)
        # print(et_test)
        c = concordance_index_ipcw(et_train, et_test, 1-surv_at_time, eval_time)[0]
        brier = brier_score(et_train, et_test, surv_at_time, eval_time)
        c_indices.append(c) 
        brier_indices.append(brier[1])
    return np.array(c_indices), np.array(brier_indices)

class Eval:

    def __init__(
        self, surv: np.ndarray, time_points: np.ndarray, 
        durations: np.ndarray, events: np.ndarray, 
        train_duration: np.ndarray, train_event: np.ndarray):
        """_summary_

        Args:
            surv (_type_): 
                Survival probability in shape of (n samples, time point)
                or (n samples, time point, n risk) if multiple risk
            time_points (_type_): time point of survival prob in shape of (time point,)
                or (n samples, time point) if time point is different for each sample
        """
        self.surv = surv
        self.time_points = time_points
       
        self.durations = durations.reshape(-1)
        self.events = events.reshape(-1)
        self.train_duration = train_duration.reshape(-1)
        self.train_event = train_event.reshape(-1)
        
    def get_et(self, risk, risk_exclude):
        
        et_train = duration_event_common_format(
            self.train_duration, self.train_event, 
            risk = risk, risk_exclude = risk_exclude )
        et_test = duration_event_common_format(
            self.durations, self.events, 
            risk = risk, risk_exclude = risk_exclude )
        
        return et_train, et_test
    
    def continuous_time_single_report(
        self, c_eval_time_percentile = [0.25,0.5,0.75],
        brier_eval_time_percentile = [0.25,0.5,0.75], interpolation = False):

        
        c_eval_times = np.quantile(self.durations[self.events != 0], c_eval_time_percentile)
        brier_eval_times = np.quantile(self.durations[self.events != 0], brier_eval_time_percentile)

        et_train, et_test = self.get_et(risk = 1, risk_exclude=False)
        return report(self.surv, self.time_points, c_eval_times, et_train, et_test, interpolation = interpolation)
    

    def continuous_time_multi_report(
            self,
        all_risk, c_eval_time_percentile = [0.25,0.5,0.75],
        brier_eval_time_percentile = [0.25,0.5,0.75], interpolation = False):
     
        all_risk_report = {}
        for risk in all_risk:
            eval_times = np.quantile(self.durations[self.events == risk], c_eval_time_percentile)

            et_train, et_test = self.get_et(risk = risk, risk_exclude=False)
            surv = self.surv[:, :, risk-1]
            c, brier = report(surv, self.time_points, eval_times, et_train, et_test, interpolation = interpolation)
            all_risk_report[risk] = (c, brier)

        return all_risk_report
        
    def discrete_time_report(
        self, c_eval_time_percentile = [0.25,0.5,0.75], 
        brier_eval_time_percentile = [0.25,0.5,0.75],
        multi_risk = False):
        
        """_summary_

        Raises:
            Exception: _description_

        Returns:
            (array of c index, array of brier score)
        """

        if len(self.surv) == 2 and multi_risk == True:
            raise Exception("Cannot do multi risk")

        c_eval_times = np.quantile(self.durations[self.events != 0], c_eval_time_percentile)
       
        surv_inter = interp1d(self.time_points, self.surv, axis = -1)
        
        brier_eval_times = np.quantile(self.durations[self.events != 0], brier_eval_time_percentile)

        if multi_risk:
            c_all_risk = []
            brier_all_risk = []
            for risk in np.unique(self.train_event):
                if risk == 0:
                    continue
                et_train, et_test = self.get_et(risk = risk, risk_exclude=False)
                c_indices = []
                for eval_time in c_eval_times:
                    c = concordance_index_ipcw(et_train, et_test, 1-surv_inter(eval_time)[:, risk-1], eval_time)[0]
                    c_indices.append(c)
                c_all_risk.append(c_indices)

                brier_indices = integrated_brier_score(et_train, et_test, surv_inter(brier_eval_times)[:, risk-1, :], brier_eval_times)
                brier_all_risk.append(brier_indices)
            
            return np.array(c_all_risk), np.array(brier_all_risk)
        
        c_indices = []
        brier_indices = []
        for eval_time in c_eval_times:
            surv_at_time = surv_inter(eval_time)
            et_train, et_test = self.get_et(risk = 1, risk_exclude=False)
            
            # if have multi-risk
            if len(surv_at_time.shape) == 2:
                et_train, et_test = self.get_et(risk = 0, risk_exclude=True)
                surv_at_time = surv_at_time.mean(axis = -1)
                
            c = concordance_index_ipcw(et_train, et_test, 1-surv_at_time, eval_time)[0]

            c_indices.append(c)
        surv_at_time = surv_inter(brier_eval_times)
        if len(surv_at_time.shape) == 3:
            surv_at_time = surv_at_time.mean(axis = 1)
        
        brier_indices = integrated_brier_score(et_train, et_test, surv_at_time, brier_eval_times)
        return np.array(c_indices), np.array(brier_indices)


def discrete_hazard_to_surv(hazard: torch.Tensor, epsilon = 1e-7):
    return (1-hazard).add(epsilon).log().cumsum(-1).exp()
    


def discrete_time_cindex(et_train, et_test, cir, cir_time):

    eval_times = np.quantile(et_train['t'][et_train['t'] != 0], [0.25, 0.5, 0.75])
    c_indices_val = []
    for eval_time in eval_times:
        index = np.searchsorted(cir_time, eval_time, side ='left')

        risk = cir[:, index]
        c_index = concordance_index_ipcw(et_train, et_test, risk, eval_time)
        c_indices_val.append(c_index[0])
    return c_indices_val