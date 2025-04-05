
from functools import partial
from copy import deepcopy

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sksurv import metrics

from .preprocessing import CTCutEqualSpacing
from .dataset import add_time_continuous_features, SurvDataset
from .train_utils import train_step, test_step, val_step
from .loss import (
    nll_continuous_time_loss,
    nll_continuous_time_loss_trapezoid,
    nll_continuous_time_multi_loss_trapezoid)
from .eval import Eval, report
from .utils import pad_col, duration_event_common_format

def activation_hazard(preds: torch.Tensor, activation = F.sigmoid, epsilon = 1e-7):
    hazard = activation(preds)
    return hazard

def continuous_hazard_to_surv(hazard: torch.Tensor, epsilon = 1e-7, method = 'simple', input_times:torch.Tensor = None):
    hazard = hazard.cpu()
    n_sample = hazard.shape[0]
    n_time = hazard.shape[1]

    if method == 'simple':
        hazard = hazard[:, 1:].mul(torch.diff(input_times))
        hazard = pad_col(hazard, where='start')
        surv = hazard.cumsum(1).mul(-1).exp()

    elif method == 'trapezoid':
        filters = torch.tensor([[[0.5,0.5]]]).to(hazard.device)
        hazard = F.conv1d(hazard.reshape(n_sample, 1, n_time), filters).reshape(n_sample, n_time-1)
        hazard = hazard.mul(torch.diff(input_times))
        hazard = pad_col(hazard, where='start')
        surv = hazard.cumsum(1).mul(-1).exp()

    return surv

class ICTSurF(nn.Module):

    

    def __init__(
        self, net, 
        model_name = 'ICTSurf', 
        processor_class = CTCutEqualSpacing,
        loss_function = nll_continuous_time_loss_trapezoid) -> None:

        super().__init__()
        self.model_name = model_name
        self.net = net
        self.processor_class = processor_class
        self.hazard_activation = F.softplus
        if loss_function == nll_continuous_time_loss_trapezoid:
            self.loss_function = loss_function
            self.pred_method = 'trapezoid'
        elif loss_function == nll_continuous_time_loss:
            self.loss_function = loss_function
            self.pred_method = 'simple'
        else:
            raise NotImplementedError


    def fit(
            self, optimizer, features_train, durations_train, events_train, 
            features_val, durations_val, events_val,
            n_discrete_time = 50, patience = 100, device = 'cpu',
            batch_size=256, epochs=1000, shuffle=True, metric_after_validation = False):
        
        if metric_after_validation:
            assert np.max(durations_train)>np.max(durations_val), "durations_train should be larger than durations_val"
        self.features_val = features_val
        self.durations_val = durations_val
        self.events_val = events_val

        self.metric_after_validation = metric_after_validation
        self.processor = self.processor_class(n_discrete_time)
        train_loader = self.get_loader(
            features_train, durations_train, events_train, fit_y=True,
            batch_size=batch_size, mode = 'train')
        val_loader = self.get_loader(
            features_val, durations_val, events_val, fit_y=False,
            batch_size=batch_size, mode = 'test')

        self.train_loader = train_loader
        
        
        # -----------------------------------------------------------------------
        self.state_dict_paths = set()
        iter_since_best = 0
        best_c = 0
        best_loss = 1e10
        for i in range(epochs):
            
            train_loss = train_step(self.net, optimizer, train_loader,self.loss_function, device = device)
            preds, loss = val_step(self.net, optimizer, val_loader, self.loss_function, device = device)
            if i<5:
                continue
            # ! not done yet ------------------------------------------
            if self.metric_after_validation:

                c_index = self.after_validation(val_loader, device = device)
                if c_index > best_c:
                    best_c = c_index
                    torch.save(self.net.state_dict(), f'{self.model_name}_c.pth')
                    self.state_dict_paths.add(f'{self.model_name}_c.pth')
                    iter_since_best =-1
                    print(f"epoch {i} c_index: {c_index} best_c: {best_c}")

            # ! -------------------------------------------------------
            else:
                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.net.state_dict(), f'{self.model_name}_loss.pth')
                    self.state_dict_paths.add(f'{self.model_name}_loss.pth')
                    iter_since_best =-1
            iter_since_best += 1
            if iter_since_best >= patience:
                break
            print(f"epoch {i} val_loss: {loss} train_loss: {train_loss}")
        if self.metric_after_validation:
            return best_c
        self.net.load_state_dict(torch.load(list(self.state_dict_paths)[0]))

        return best_loss

    def after_validation(self, loader, device = 'cpu'):
        eval_times= np.quantile(loader.dataset.durations[loader.dataset.events != 0], [0.25, 0.5, 0.75])

        c_list = []
        for time in eval_times:
            val_loader = self.get_loader(
                    self.features_val, np.array([time]*len(self.events_val)), np.array([1]*len(self.events_val)),
                    fit_y = False, batch_size = 256, mode = 'test')

            preds = test_step(self.net, val_loader, device = device)
            surv = self.pred_to_surv(preds, val_loader).cpu().detach().numpy()
            
            et_train, et_test = self._get_et(self.durations_val, self.events_val)
            # print(surv.shape, len(et_test), len(et_train))
            c_index, brier = report(surv, val_loader.dataset.extended_data['continuous_times'], [time], et_train, et_test, interpolation = True)
            c_list.append(c_index[0])
        return np.mean(c_list)

    def _get_et(self, durations, events, risk = 1, risk_exclude = False):
        et_test = duration_event_common_format(durations, events, risk = risk, risk_exclude = risk_exclude )
        et_train = duration_event_common_format(self.train_loader.dataset.durations, self.train_loader.dataset.events, risk = risk, risk_exclude = risk_exclude )
        return et_train, et_test
    
    def get_loader(self, features, durations, events, fit_y = False, batch_size = 256, mode = 'test'):
        if fit_y:
            y = self.processor.fit_transform(durations, events)
        else:
            y = self.processor.transform(durations, events)
        continuous_times = y[2]

        x = add_time_continuous_features(features,continuous_times)

        dataset = SurvDataset(x, y, durations, events, continuous_times = continuous_times)
 
        if mode == 'train':
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return loader

    def evaluate(self, features, durations, events, quantile_evals = [0.25,0.5,0.75], interpolation = False, device = 'cpu'):

        assert np.max(self.train_loader.dataset.durations)>np.max(durations), "durations_train should be larger than test"

        eval_times = np.quantile(durations[events != 0], quantile_evals)

        checkpoint_name = []
        all_c_index = []
        all_brier = []
        df = pd.DataFrame()
        for eval_time, quantile_eval in zip(eval_times, quantile_evals):

            test_events = np.array([1]*len(durations))
            test_durations = np.array([eval_time]*len(durations))

            test_loader = self.get_loader(
                features, test_durations, test_events, 
                fit_y = False, batch_size = 256, mode = 'test')
            # y = self.get_y(test_durations, test_events, fit = False)
            # test_loader = self.get_dataloader(features, y, test_durations, test_events)

            for state_dict_path in self.state_dict_paths:
                checkpoint_name.append(state_dict_path)
                self.net.load_state_dict(torch.load(state_dict_path))
                preds = test_step(self.net, test_loader, device = device)
                outputs = self.pred_to_surv(preds, test_loader).cpu().detach().numpy()
               
                eval_object = Eval(
                    outputs, test_loader.dataset.extended_data['continuous_times'], 
                    durations, events, 
                    self.train_loader.dataset.durations,self.train_loader.dataset.events)
                c_index, brier = eval_object.continuous_time_single_report(
                    c_eval_time_percentile = [quantile_eval],
                    brier_eval_time_percentile = [quantile_eval], interpolation = interpolation)

                tmp_df = pd.DataFrame([[state_dict_path, quantile_eval, c_index[0], brier[0][0]]],columns = ['model', 'timepoint', 'c_index', 'brier'])
                df = pd.concat([df, tmp_df], axis = 0)

        return df

        
    def pred_to_hazard(self, preds: torch.Tensor):
        hazard = activation_hazard(preds, activation = self.hazard_activation)
        return hazard
    
    def pred_to_cif(self, preds: torch.Tensor):
        surv = self.pred_to_surv(preds)
        return 1-surv
        
    def pred_to_surv(self, preds: torch.Tensor, loader):
        hazard = self.pred_to_hazard(preds)
        return continuous_hazard_to_surv(hazard, input_times = loader.dataset.input_times, method=self.pred_method)
    
    def forward(self, x):
        return self.net(x)
    
class ICTSurFMulti(nn.Module):

    

    def __init__(
        self, net, model_name = 'ICTSurF',
        processor_class = CTCutEqualSpacing,
        loss_function = nll_continuous_time_multi_loss_trapezoid) -> None:
        super().__init__()

        self.model_name = model_name
        self.net = net

        self.processor_class = processor_class

        self.loss_function = loss_function
        self.pred_method = 'trapezoid'

    def fit(
            self, optimizer, features_train, durations_train, events_train, 
            features_val, durations_val, events_val,
            n_discrete_time = 10, patience = 100, device = 'cpu',
            batch_size=256, epochs=100, shuffle=True, metric_after_validation = False):
        
        if metric_after_validation:
            assert np.max(durations_train)>np.max(durations_val), "durations_train should be larger than durations_val"


        self.metric_after_validation = metric_after_validation
        self.processor = self.processor_class(n_discrete_time)

        train_loader = self.get_loader(
            features_train, durations_train, events_train, fit_y=True,
            batch_size=batch_size, mode = 'train')
        val_loader = self.get_loader(
            features_val, durations_val, events_val, fit_y=False,
            batch_size=batch_size, mode = 'test')

        self.train_loader = train_loader

        self.state_dict_paths = set()
        self.all_risk = range(np.max(events_train))

        iter_since_best = 0
        best_c = 0
        best_loss = 1e10
        for i in range(epochs):
            # training
            train_loss = train_step(self.net, optimizer, train_loader, self.loss_function, all_risk=self.all_risk, device = device)
            preds, loss = val_step(self.net, optimizer, val_loader, self.loss_function, all_risk=self.all_risk, device = device)

            if self.metric_after_validation:
                c_index, brier = self.after_validation(preds, val_loader, train_loader)

                if np.mean(c_index) > best_c:
                    best_c = np.mean(c_index)
                    torch.save(self.net.state_dict(), f'{self.model_name}_c.pth')
                    self.state_dict_paths.add(f'{self.model_name}_c.pth')
                    # self.state_dict_paths.append(f'{self.model_name}_c.pth')
            if loss < best_loss:
                best_loss = loss
                torch.save(self.net.state_dict(), f'{self.model_name}_loss.pth')
                self.state_dict_paths.add(f'{self.model_name}_loss.pth')
                iter_since_best =-1
            iter_since_best += 1
            if iter_since_best >= patience:
                break
            print(f"epoch {i} val_loss: {loss} train_loss: {train_loss}")
        self.net.load_state_dict(torch.load(list(self.state_dict_paths)[0]))
        return best_loss
            
    def after_validation(self, preds, val_loader, train_loader):
        outputs = self.pred_to_surv(preds, val_loader).cpu().detach().numpy()

        durations = val_loader.dataset.durations
        events = val_loader.dataset.events


        eval_object = Eval(
            outputs, val_loader.dataset.extended_data['continuous_times'], 
            val_loader.dataset.durations, val_loader.dataset.events, 
            train_loader.dataset.durations,train_loader.dataset.events)
        
        quantile_evals = [0.25,0.5,0.75]
        eval_times = np.quantile(val_loader.dataset.durations[val_loader.dataset.events != 0], quantile_evals)
        for eval_time, quantile_eval in zip(eval_times, quantile_evals):
            # print(eval_time)
            
            outputs = self.pred_to_surv(preds, test_loader).cpu().detach().numpy()
            eval_object = Eval(
                outputs, test_loader.dataset.extended_data['continuous_times'], 
                durations, events, 
                self.train_loader.dataset.durations,self.train_loader.dataset.events)
            c_index, brier = eval_object.continuous_time_single_report(
                c_eval_time_percentile = [quantile_eval],
                brier_eval_time_percentile = [quantile_eval]
            )
            print(f"test best C {c_index} {brier}")
        return c_index, brier

    def get_loader(self, features, durations, events, fit_y = False, batch_size = 256, mode = 'test'):
        if fit_y:
            y = self.processor.fit_transform(durations, events)
        else:
            y = self.processor.transform(durations, events)
        continuous_times = y[2]

        x = add_time_continuous_features(features,continuous_times)

        dataset = SurvDataset(x, y, durations, events, continuous_times = continuous_times)
 
        if mode == 'train':
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return loader

    def evaluate(
        self, features, durations, events, 
        quantile_evals = [0.25,0.5,0.75], 
        batch_size = 256,
        interpolation = False, device = 'cpu'): 

        assert np.max(self.train_loader.dataset.durations)>np.max(durations), "durations_train should be larger than test"

        all_risk = np.array(self.all_risk)+1

        df = pd.DataFrame()
        for risk in all_risk:
            eval_times = np.quantile(durations[events == risk], quantile_evals)

            for eval_time, quantile_eval in zip(eval_times, quantile_evals):

                test_events = np.array([0]*len(durations))
                test_durations = np.array([eval_time]*len(durations))
                test_loader = self.get_loader(
                    features, test_durations, test_events, 
                    fit_y = False,
                    batch_size = batch_size, mode = 'test')
                for state_dict_path in self.state_dict_paths:

                    self.net.load_state_dict(torch.load(state_dict_path))
                    preds = test_step(self.net, test_loader, device = device)
                    surv = self.pred_to_surv(preds.cpu(), test_loader).cpu().detach().numpy()

                    et_train, et_test = self.get_et(durations, events, risk = risk, risk_exclude=False)
                    c, brier = report(
                        surv[:, :, risk-1], 
                        test_loader.dataset.extended_data['continuous_times'], 
                        [eval_time], et_train, et_test, interpolation = interpolation)
                    tmp_df = pd.DataFrame(
                        [[state_dict_path, risk, quantile_eval, c[0], brier[0][0]]],
                        columns = ['model', 'risk', 'timepoint', 'c_index', 'brier'])
                    df = pd.concat([df, tmp_df], axis = 0)
        return df

    def get_et(self,durations, events, risk, risk_exclude):
        
        et_train = duration_event_common_format(
            self.train_loader.dataset.durations, self.train_loader.dataset.events,  
            risk = risk, risk_exclude = risk_exclude )
        et_test = duration_event_common_format(
            durations, events, 
            risk = risk, risk_exclude = risk_exclude )
        
        return et_train, et_test

    def pred_to_hazard(self, preds: torch.Tensor):
        hazard = activation_hazard(preds, activation = F.softplus)
        return hazard
    
    def pred_to_cif(self, preds: torch.Tensor):
        surv = self.pred_to_surv(preds)
        return 1-surv
        
    def pred_to_surv(self, preds: torch.Tensor, loader):
        hazard = self.pred_to_hazard(preds)
        surv = []
        
        for risk in self.all_risk:

            surv_tmp = continuous_hazard_to_surv(hazard[:,:,risk], input_times=loader.dataset.input_times, method=self.pred_method)
            surv.append(surv_tmp.view(*surv_tmp.shape, 1))

        return torch.cat(surv, dim = -1)

    def forward(self, x):
        return self.net(x)
if __name__ == "__main__":
    print(__file__)