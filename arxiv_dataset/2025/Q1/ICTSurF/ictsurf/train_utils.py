
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import numpy as np

from .utils import *

def train_step(model, optimizer, loader, loss_function, device = 'cpu', **loss_args):
    losses = []
    model.train()

    for batch_idx, (data, y) in enumerate(loader):

        for i, d in enumerate(y):
            y[i] = d.to(device)

        data = data.to(device).float()

        optimizer.zero_grad()

        out = model(data)
        # print(len(y))
        # print(loss_args)
        loss = loss_function(out, *y, **loss_args)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().item())
    return np.mean(losses)

def val_step(model, optimizer, loader, loss_function, device = 'cpu', **loss_args):
    outputs = []
    losses = []
    weight = []
    model.eval()
    for batch_idx, (data, y) in enumerate(loader):
        weight.append(len(data))
        for i, d in enumerate(y):
            y[i] = d.to(device)
        data = data.to(device).float()

        with torch.no_grad():
            out = model(data)
            loss = loss_function(out, *y, **loss_args)
            outputs.append(out)
        losses.append(loss.cpu().detach().item())
    preds = torch.cat(outputs, dim = 0)

    return preds, np.sum(np.array(weight) * np.array(losses))/np.sum(weight)

def test_step(model, loader, device = 'cpu'):
    outputs = []
    model.eval()
    for batch_idx, (data, y) in enumerate(loader):
        for i, d in enumerate(y):
            y[i] = d.to(device)
        data = data.to(device).float()
        with torch.no_grad():
            out = model(data)
            outputs.append(out)

    preds = torch.cat(outputs, dim = 0)
    return preds


def kfold_split(features, durations, events, train_val_test_function,
        random_state = 1234, n_splits = 5, **kwargs):
    skf = StratifiedKFold(n_splits=n_splits, random_state = random_state, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(features, events)):
        features_test = features[test_index]
        features_train  = features[train_index]

        durations_test = durations[test_index]
        durations_train  = durations[train_index]

        events_test = events[test_index]
        events_train  = events[train_index]

        features_train, features_val, durations_train, durations_val, events_train, events_val = train_test_split(
            features_train, durations_train, events_train, test_size=0.2, stratify = events_train)
        
    
        while durations_train.max()<=durations_test.max():
            test_index_max = durations_test.argmax()
            durations_test = np.delete(durations_test, test_index_max)
            features_test = np.delete(features_test, test_index_max, axis = 0)
            events_test = np.delete(events_test, test_index_max)
        
        while durations_train.max()<=durations_val.max():
            test_index_max = durations_val.argmax()
            durations_val = np.delete(durations_val, test_index_max)
            features_val = np.delete(features_val, test_index_max, axis = 0)
            events_val = np.delete(events_val, test_index_max)
            
        # while durations_train.min()>=durations_test.min():
        #     test_index_max = durations_test.argmin()
        #     durations_test = np.delete(durations_test, test_index_max)
        #     features_test = np.delete(features_test, test_index_max, axis = 0)
        #     events_test = np.delete(events_test, test_index_max)
        
        model = train_val_test_function(
            (features_train, durations_train, events_train), 
            (features_val, durations_val, events_val),
            (features_test, durations_test, events_test),
             **kwargs)
        
    return model

def get_dataset(features, durations, events, random_state = 1234, n_splits = 5, **kwargs):
    
    skf = StratifiedKFold(n_splits=n_splits, random_state = random_state, shuffle=True)
    datasets = {}
    for i, (train_index, test_index) in enumerate(skf.split(features, events)):
        features_test = features[test_index]
        features_train  = features[train_index]

        durations_test = durations[test_index]
        durations_train  = durations[train_index]

        events_test = events[test_index]
        events_train  = events[train_index]

        # features_train, features_val, durations_train, durations_val, events_train, events_val = train_test_split(
        #     features_train, durations_train, events_train, test_size=0.2, random_state=random_state, stratify = events_train)
        
    
        while durations_train.max()<=durations_test.max():
            test_index_max = durations_test.argmax()
            durations_test = np.delete(durations_test, test_index_max)
            features_test = np.delete(features_test, test_index_max, axis = 0)
            events_test = np.delete(events_test, test_index_max)
        
        # while durations_train.max()<=durations_val.max():
        #     test_index_max = durations_val.argmax()
        #     durations_val = np.delete(durations_val, test_index_max)
        #     features_val = np.delete(features_val, test_index_max, axis = 0)
        #     events_val = np.delete(events_val, test_index_max)
            
        # while durations_train.min()>=durations_test.min():
        #     test_index_max = durations_test.argmin()
        #     durations_test = np.delete(durations_test, test_index_max)
        #     features_test = np.delete(features_test, test_index_max, axis = 0)
        #     events_test = np.delete(events_test, test_index_max)
        
        # model = train_val_test_function(
        #     (features_train, durations_train, events_train), 
        #     (features_val, durations_val, events_val),
        #     (features_test, durations_test, events_test),
        #      **kwargs)
        
        train = {'features_train' : features_train, 'durations_train' : durations_train, 'events_train' : events_train}
        test = {'features_test' : features_test, 'durations_test' : durations_test, 'events_test' : events_test}
        
        datasets[i] = {'train' : train, 'test' : test}
        
    return datasets