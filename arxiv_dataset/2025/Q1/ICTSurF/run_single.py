import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
import torchtuples as tt
from torch import Tensor
import optuna

from ictsurf.dataset import (
    get_metabric_dataset_onehot,
    get_support2_dataset_onehot,
    get_gaussian_dataset
)
from ictsurf.preprocessing import cut_continuous_time,  CTCutEqualSpacing, CTAddedSameSpacing
from ictsurf.eval import *
from ictsurf.utils import *
from ictsurf.loss import nll_continuous_time_loss_trapezoid
from ictsurf.model import MLPVanilla, MLPTimeEncode
from ictsurf import ICTSurF

import warnings

warnings.filterwarnings("ignore")

random_state = 1234
np.random.seed(random_state)
_ = torch.manual_seed(random_state)

n_duration = 50

epochs = 1000
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
    "time_dim": [ 16, 32],
    "num_nodes": [[64], [128]],
    'num_nodes_res': [[64], [128]],
    }


def objective(trial: optuna.Trial,
              features_train, durations_train, events_train,
              features_test, durations_test, events_test,
              random_state = 1234):
    output_risk = int(np.max(events_train))

    the_model = ICTSurF
    loss_function = nll_continuous_time_loss_trapezoid
        

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
        n_discrete_time = n_duration, patience = patience,
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
    else:
        raise ValueError("dataset should be one of ['metabric', 'support', 'gaussian']")
    output_risk = int(np.max(events))

    print("running single risk")
    the_model = ICTSurF
    loss_function = nll_continuous_time_loss_trapezoid


    skf = StratifiedKFold(n_splits=5, random_state = random_state, shuffle=True)
    df = pd.DataFrame()

    features, features_val_hparam, durations, durations_val_hparam, events, events_val_hparam = train_test_split(
            features, durations, events, test_size=0.15, random_state = random_state, stratify = events)

    mean_time = np.mean(durations_val_hparam)
    durations = durations/mean_time
    durations_val_hparam = durations_val_hparam/mean_time
        
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

        features_train, features_val, durations_train, durations_val, events_train, events_val = train_test_split(
            features_train, durations_train, events_train, test_size=0.25, random_state = random_state, stratify = events_train)

        features_train = deepcopy(np.concatenate([features_train, features_val_hparam], axis = 0))
        durations_train = deepcopy(np.concatenate([durations_train, durations_val_hparam], axis = 0))
        events_train = deepcopy(np.concatenate([events_train, events_val_hparam], axis = 0))

        while durations_train.max()<=durations_test.max():
            test_index_max = durations_test.argmax()
            durations_test = deepcopy(np.delete(durations_test, test_index_max))
            features_test = deepcopy(np.delete(features_test, test_index_max, axis = 0))
            events_test = deepcopy(np.delete(events_test, test_index_max))
            
        while durations_train.max()<=durations_val.max():
            test_index_max = durations_val.argmax()
            durations_val = deepcopy(np.delete(durations_val, test_index_max))
            features_val = deepcopy(np.delete(features_val, test_index_max, axis = 0))
            events_val = deepcopy(np.delete(events_val, test_index_max))
        scaler =  StandardScaler()

        features_train = scaler.fit_transform(features_train)
        features_val = scaler.transform(features_val)
        features_test = scaler.transform(features_test)
        print(len(features_train), len(features_val), len(features_test))

        print(np.unique(events_train, return_counts=True))
        print(np.unique(events_test, return_counts=True))
        # -----------------------------------------------------------------------

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

        tmp_df = model.evaluate(
            features_test, durations_test, events_test, 
            quantile_evals = [0.25, 0.5, 0.75],
            interpolation = True, device = device)
        tmp_df['fold'] = i
        
        df = pd.concat([df,tmp_df])
        print(tmp_df)


    if processor_class == CTCutEqualSpacing:
        df.to_csv(f"{dataset}_CTCutEqualSpacing_{n_duration}.csv")
    elif processor_class == CTAddedSameSpacing:
        df.to_csv(f"{dataset}_CTAddedSameSpacing_{n_duration}.csv")
if __name__ == "__main__":
    run('metabric')
    run('support')
    run('gaussian')

