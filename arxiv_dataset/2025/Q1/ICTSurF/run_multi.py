import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from sksurv import metrics as skmetrics

import torch
import torch.nn as nn
from torch import Tensor
import optuna

from ictsurf.dataset import (
    get_metabric_dataset_onehot,
    get_support2_dataset_onehot,
    get_gaussian_dataset,
    get_synthetic_dataset_compet,
    get_loader
)
from ictsurf.preprocessing import cut_continuous_time,  CTCutEqualSpacing
from ictsurf.eval import *
from ictsurf.utils import *
from ictsurf.loss import nll_continuous_time_multi_loss_trapezoid
from ictsurf.model import MLPTimeEncode
from ictsurf.train_utils import test_step
from ictsurf import ICTSurF, ICTSurFMulti

import warnings

warnings.filterwarnings("ignore")

random_state = 1234
np.random.seed(random_state)
_ = torch.manual_seed(random_state)

n_duration = 50

epochs = 1
batch_norm = True
batch_size = 256
device ='cpu'
dropout = 0.0
patience = 100
lr= 0.0002
activation = nn.ReLU


processor_class = CTCutEqualSpacing
metric_after_validation = False

search_space = {
    "time_dim": [16, 32],
    "num_nodes": [[64], [128]],
    'num_nodes_res': [[64], [128]]
    }

def objective(trial: optuna.Trial,
              features_train, durations_train, events_train,
              features_test, durations_test, events_test,
              random_state = 1234):

    output_risk = int(np.max(events_train))

    the_model = ICTSurFMulti
    loss_function = nll_continuous_time_multi_loss_trapezoid
        
    num_nodes =trial.suggest_categorical("num_nodes", search_space['num_nodes'])
    num_nodes_res = trial.suggest_categorical("num_nodes_res", search_space['num_nodes_res'])
    time_dim = trial.suggest_categorical("time_dim", search_space['time_dim'])

    while durations_train.max()<=durations_test.max():
        test_index_max = durations_test.argmax()
        durations_test = deepcopy(np.delete(durations_test, test_index_max))
        features_test = deepcopy(np.delete(features_test, test_index_max, axis = 0))
        events_test = deepcopy(np.delete(events_test, test_index_max))

    scaler =  StandardScaler()
    features_train = scaler.fit_transform(deepcopy(features_train))
    features_test = scaler.transform(deepcopy(features_test))


    time_scaler = 1
    in_features = features_train.shape[1]+1


    net = MLPTimeEncode(
    in_features, num_nodes,num_nodes_res, time_dim=time_dim,batch_norm= batch_norm,
    dropout=dropout, activation=activation, output_risk =output_risk).float()

    model = the_model(
        net, 
        processor_class = processor_class, 
        loss_function=loss_function).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_metric = model.fit(optimizer, features_train, durations_train, events_train,
                features_test, durations_test, events_test, device = device,
        n_discrete_time = n_duration, patience = patience, time_scaler =time_scaler,
        batch_size=batch_size, epochs=epochs, shuffle=True,
            metric_after_validation = metric_after_validation)

    return best_metric

def run(dataset ):
    if dataset == 'metabric':
        features, durations, events = get_metabric_dataset_onehot()

    elif dataset == 'support':
        features, durations, events = get_support2_dataset_onehot()

    elif dataset == 'gaussian':
        features, durations, events = get_gaussian_dataset()
    elif dataset =='synthetic_compet':
        features, durations, events, true_duration, true_events, risk1_durations, risk2_durations = get_synthetic_dataset_compet()
    else:
        raise ValueError("dataset should be one of ['metabric', 'support', 'gaussian', 'synthetic_compet']")
    output_risk = int(np.max(events))

    the_model = ICTSurFMulti
    loss_function = nll_continuous_time_multi_loss_trapezoid

    skf = StratifiedKFold(n_splits=5, random_state = random_state, shuffle=True)
    df = pd.DataFrame()

    (
        features, features_val_hparam, 
        durations, durations_val_hparam, 
        events, events_val_hparam, 
        true_duration, true_duration_val_hparam,  
        true_events, true_events_val_hparam,
        risk1_durations, risk1_durations_val_hparam,
        risk2_durations, risk2_durations_val_hparam) = train_test_split(
            features, durations, events, true_duration, true_events, risk1_durations, risk2_durations,
            test_size=0.15, random_state = random_state, stratify = events)

    mean_time = np.mean(durations_val_hparam)
    durations = durations/mean_time
    durations_val_hparam = durations_val_hparam/mean_time
    true_duration = true_duration/mean_time
    true_duration_val_hparam = true_duration_val_hparam/mean_time

    risk1_durations = risk1_durations/mean_time
    risk1_durations_val_hparam = risk1_durations_val_hparam/mean_time
    risk2_durations = risk2_durations/mean_time
    risk2_durations_val_hparam = risk2_durations_val_hparam/mean_time

        
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), directions = ['minimize'])
    partial_objective = lambda trial: objective(
        trial,
        features, durations, events,
        features_val_hparam, durations_val_hparam, events_val_hparam,
        random_state = random_state)
    study.optimize(partial_objective )
    num_nodes =study.best_params['num_nodes']
    num_nodes_res = study.best_params['num_nodes_res']
    time_dim = study.best_params['time_dim']


    for i, (train_index, test_index) in enumerate(skf.split(features, events)):

        print("---------------------------------------")
        features_test = features[test_index]
        features_train  = features[train_index]

        durations_test = durations[test_index]
        durations_train  = durations[train_index]

        events_test = events[test_index]
        events_train  = events[train_index]

        true_duration_test = true_duration[test_index]
        true_duration_train  = true_duration[train_index]

        true_events_test = true_events[test_index]
        true_events_train  = true_events[train_index]

        risk1_durations_test = risk1_durations[test_index]
        risk1_durations_train  = risk1_durations[train_index]

        risk2_durations_test = risk2_durations[test_index]
        risk2_durations_train  = risk2_durations[train_index]

        (
            features_train, features_val, 
            durations_train, durations_val, 
            events_train, events_val, 
            true_duration_train, _,  
            true_events_train, _ )= train_test_split(
            features_train, durations_train, events_train, true_duration_train, true_events_train,
            test_size=0.25, random_state = random_state, stratify = events_train)

        features_train = deepcopy(np.concatenate([features_train, features_val_hparam], axis = 0))
        durations_train = deepcopy(np.concatenate([durations_train, durations_val_hparam], axis = 0))
        events_train = deepcopy(np.concatenate([events_train, events_val_hparam], axis = 0))

        scaler =  StandardScaler()

        features_train = scaler.fit_transform(features_train)
        features_val = scaler.transform(features_val)
        features_test = scaler.transform(features_test)

        # -----------------------------------------------------------------------
        # +1 for time feature
        in_features = features_train.shape[1]+1
    
        net = MLPTimeEncode(
        in_features, num_nodes, num_nodes_res, time_dim=time_dim, batch_norm= batch_norm,
        dropout=dropout, activation=activation, output_risk = output_risk).float()
        
        model = the_model(
            net, 
            processor_class = processor_class, 
            loss_function=loss_function).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        best_loss = model.fit(optimizer, features_train, durations_train, events_train,
                    features_val, durations_val, events_val,
            n_discrete_time = n_duration, patience = patience, device = device,
            batch_size=batch_size, epochs=epochs, shuffle=True,
            metric_after_validation = metric_after_validation)

        all_risk = np.array(list(range(np.max(events_train)))) +1
        quantile_evals = [0.25, 0.5, 0.75]
        for risk in all_risk:
            eval_times = np.quantile(true_duration_test[true_events_test == risk], quantile_evals)

            for eval_time, quantile_eval in zip(eval_times, quantile_evals):
                
                event_risk = np.ones_like(true_events_test)
                event_risk[true_events_test != risk] = 0
                event_risk = event_risk.astype(bool)

                y_true = np.ones_like(true_events_test)
                if risk == 1:
                    y_true[risk1_durations_test > eval_time] = 0
                elif risk == 2:
                    y_true[risk2_durations_test > eval_time] = 0
                else:
                    raise Exception("Error")

                test_events = np.array([0]*len(true_events_test))
                test_durations = np.array([eval_time]*len(true_events_test))
                test_loader = get_loader(
                    features_test, test_durations, test_events, model.processor,
                    fit_y = False, 
                    batch_size = batch_size, mode = 'test')
                for state_dict_path in model.state_dict_paths:

                    model.net.load_state_dict(torch.load(state_dict_path))
                    preds = test_step(model.net, test_loader, device = device)
                    surv = model.pred_to_surv(preds.cpu(), test_loader).cpu().detach().numpy()

                    index = index_at_time(eval_time, test_loader.dataset.extended_data['continuous_times'])
                    surv = surv[:, :, risk-1]

                    surv_at_time = np.take_along_axis(surv, index.reshape(-1,1), axis = -1).reshape(-1)
                    c = skmetrics.concordance_index_censored(event_risk, true_duration_test, 1-surv_at_time)[0]
                    brier = metrics.brier_score_loss(y_true, 1-surv_at_time)

                    tmp_df = pd.DataFrame([[state_dict_path,risk, c, brier, quantile_eval, i]], columns = ['model','risk', 'c_index', 'brier', 'time', 'fold'])
                    df = pd.concat([df,tmp_df])
                    print(tmp_df)

    if processor_class == CTCutEqualSpacing:
        df.to_csv(f"{dataset}_CTCutEqualSpacing_{n_duration}.csv")
    elif processor_class == CTAddedSameSpacing:
        df.to_csv(f"{dataset}_CTAddedSameSpacing_{n_duration}.csv")

if __name__ == "__main__":

    run('synthetic_compet')

