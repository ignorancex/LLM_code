import pathlib
from copy import deepcopy
import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import h5py
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder



def get_support2_dataset():
    data =  defaultdict(dict)
    basepath = pathlib.Path(__file__).parent.resolve()
    path = pathlib.Path(basepath, './data/support/support2.csv')
    df  = pd.read_csv(path)
    events = df['death'].values
    durations = df['d.time'].values
    df = df.drop(['death', 'd.time'], axis=1)

    missing_use = ['edu', 'scoma', 'race','wblc','resp', 'pafi','bili', 'sod' ,'glucose', 'urine','adls', 
    'income','avtisst','meanbp', 'hrt', 'temp', 'alb', 'crea', 'ph', 'bun', 'adlp', 'age', 'sex', 
    'dzgroup', 'dzclass', 
    'diabetes', 'dementia', 'ca']

    cat_features = ['race', 'income', 'sex', 'dzgroup', 'dzclass', 'diabetes', 'dementia', 'ca']
    num_features = [f for f in missing_use if f not in cat_features]

    df = df[missing_use]

    manual_fill = ['alb', 'pafi', 'bili', 'crea', 'bun', 'wblc', 'urine']
    df['alb'].fillna(3.5, inplace=True)
    df['pafi'].fillna(333.3, inplace=True)
    df['bili'].fillna(1.01, inplace=True)
    df['crea'].fillna(1.01, inplace=True)
    df['bun'].fillna(6.51, inplace=True)
    df['wblc'].fillna(9, inplace=True)
    df['urine'].fillna(2502, inplace=True)

    for f in num_features:
        if f not in manual_fill:
            df[f] = df[f].fillna(np.nanmean(df[f]))
    for f in cat_features:
        df[f] = df[f].fillna(df[f][~df[f].isna()].mode().values[0])
    df = df[cat_features + num_features]
    return df.values, durations, events, list(range(len(cat_features))), list(range(len(cat_features), len(cat_features) + len(num_features)))

def get_support2_dataset_onehot():
    features, durations, events, cols_categorical, cols_standardize = get_support2_dataset()
    enc = OneHotEncoder(handle_unknown='ignore')
    df = pd.DataFrame(features)
    df_cat_onehot = pd.DataFrame(enc.fit_transform(pd.DataFrame(features)[cols_categorical]).toarray())
    df = df.drop(cols_categorical, axis=1)
    df = pd.concat([df, df_cat_onehot], axis=1)
    return df.values.astype(np.float32), durations, events

def generate_synthetic_compet_b():
    n_samples = 30000
    n_features = 20

    features = np.random.randn( n_samples, n_features)

    risk1 = (features[:, 0] + features[:, 1] + features[:, 2] + features[:, 3])
    risk2 = (features[:, 0] + features[:, 1]+ features[:, 2] + features[:, 3])
    
    risk1 = np.cosh(risk1)
    risk2 = np.abs(np.sinh(risk2) + np.random.randn())

    t1 = np.random.exponential(risk1)
    t2 =  np.random.exponential(risk2)

    true_durations = np.zeros_like(t1)
    true_durations = deepcopy(t1)
    true_durations[t2<t1] = deepcopy(t2[t2<t1])
    true_events = np.ones_like(true_durations)
    true_events[t2<t1] = 2

    censor_idx = np.random.choice(list(range(n_samples)), replace=False, size = int(n_samples/2))

    events = deepcopy(true_events)
    events[censor_idx] = 0

    censor_times = [] 
    for idx in censor_idx:
        censor_time = np.random.uniform(0, true_durations[idx])
        censor_times.append(censor_time)

    durations = deepcopy(true_durations)
    durations[censor_idx] = censor_times

    df = pd.DataFrame(features)
    df['durations'] = durations
    df['events'] = events
    df['true_events'] = true_events
    df['true_durations'] = true_durations
    df['risk_1_durations'] = t1
    df['risk_2_durations'] = t2

    df.to_csv('./data/synthetic_compet/synthetic_compet_b.csv', index = False)


def get_synthetic_dataset_compet():
    data =  defaultdict(dict)
    basepath = pathlib.Path(__file__).parent.resolve()
    path = pathlib.Path(basepath, './data/synthetic_compet/synthetic_compet_b.csv')
    df = pd.read_csv(path)
    features = df.drop(columns=['durations', 'events', 'true_events', 'true_durations', 'risk_1_durations', 'risk_2_durations']).values
    durations = df['durations'].values
    events = df['events'].values
    true_durations = df['true_durations'].values
    true_events = df['true_events'].values
    risk_1_durations = df['risk_1_durations'].values
    risk_2_durations = df['risk_2_durations'].values
    return (
        features.astype(np.float32), 
        durations.astype(np.float32), 
        events.astype(np.int32), 
        true_durations.astype(np.float32), 
        true_events.astype(np.int32),
        risk_1_durations.astype(np.float32),
        risk_2_durations.astype(np.float32))

def get_metabric_dataset_onehot():
    data =  defaultdict(dict)
    basepath = pathlib.Path(__file__).parent.resolve()
    path = pathlib.Path(basepath, './data/metabric/cleaned_features_final.csv')
    label_path = pathlib.Path(basepath, './data/metabric/label.csv')
    df = pd.read_csv(path)
    df_label = pd.read_csv(label_path)
    return df.values.astype(np.float32), df_label['event_time'].values, df_label['label'].values

def get_metabric_dataset():
    
    data =  defaultdict(dict)
    basepath = pathlib.Path(__file__).parent.resolve()
    path = pathlib.Path(basepath, './data/metabric/cleaned_features_final.csv')
    label_path = pathlib.Path(basepath, './data/metabric/label.csv')

    df = pd.read_csv(path)
    label_df = pd.read_csv(label_path)
    new_df = pd.DataFrame()
    cols_categorical=[
        'grade', 'histological', 'ER_IHC_status', 
        'ER_Expr', 'PR_Expz', 'HER2_IHC_status',
        'HER2_SNP6_state', 'Her2_Expr', 'Treatment',
        'inf_men_status', 'group', 'cellularity',
        'Pam50_Subtype', 'int_clust_memb', 'site',
        'Genefu']
    cols_standardize = [
        'age_at_diagnosis', 'size', 'lymph_nodes_positive',
        'stage', 'lymph_nodes_removed', 'NPI'
    ]
    for col in cols_categorical:
        new_df[col] = np.argmax(df[[c for c in df.columns if col in c]], axis=1)
    new_df[cols_standardize] = df[cols_standardize]
    cols_categorical = list(range(len(cols_categorical)))
    cols_standardize = list(range(len(cols_categorical), len(cols_categorical) + len(cols_standardize)))

    return new_df.values.astype(np.float32), label_df['event_time'].values, label_df['label'].values, cols_categorical, cols_standardize



def get_gaussian_dataset():
    data =  defaultdict(dict)
    basepath = pathlib.Path(__file__).parent.resolve()
    path = pathlib.Path(basepath, './data/gaussian/gaussian_survival_data.h5')

    print(f'loading {path}')
    with h5py.File(path) as f:
        for ds in f: 
            for array in f[ds]:
                data[ds][array] = f[ds][array][:]


    events = np.concatenate([data['test']['e'], data['train']['e']], axis = 0)
    durations = np.concatenate([data['test']['t'], data['train']['t']], axis = 0)
    features = np.concatenate([data['test']['x'], data['train']['x']], axis = 0)
    return features, durations, events

def get_pbc2_dataset():
    data =  defaultdict(dict)
    basepath = pathlib.Path(__file__).parent.resolve()
    path = pathlib.Path(basepath, './data/pbc2/pbc2_cleaned.csv')
    df = pd.read_csv(path)
    
    df_time = df.groupby('id')[['tte']].first()
    df_time['times'] = df_time['tte']
    df_time = df_time.drop('tte', axis=1)

    df = pd.concat([df, df_time.reset_index()], axis=0).sort_values(['id', 'times'])
    df['tte'] = df['tte'].ffill()
    df['label'] = df['label'].ffill()
    df  = df.reset_index(drop=True)
    tmp = []
    for i in range(1,len(df['times'])):
        if df['times'][i-1] == df['times'][i] and df['times'][i] != 0:
            tmp.append(i)
    df = df.drop(index=tmp)
    return df.reset_index(drop=True)

class SurvDataset(Dataset):
    def __init__(self, x, y, durations, events, **extended_data):
        super().__init__()
        self.x = x
        self.y = [i for i in zip(*y)]
        self.durations = durations
        self.events = events

        self.extended_data = extended_data
    
    @property
    def input_times(self):
        return torch.tensor(self.x[:, :, -1])

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        
        return self.x[idx], self.y[idx]
    
def add_time_features(features, discrete_time):
    discrete_time = discrete_time.reshape(-1)
    n_sample = features.shape[0]
    n_discrete_time = len(discrete_time)
    features = features.repeat(n_discrete_time, 0)
    discrete_time = discrete_time.reshape(1,-1).repeat( n_sample, 0).reshape(-1, 1)

    return np.concatenate([features, discrete_time], axis = 1).reshape(n_sample,n_discrete_time, -1 )

def add_time_single_features(features, durations):

    return np.concatenate([features, durations.reshape(-1,1)], axis = 1)

def add_time_continuous_features(features, continuous_time):
    
    features = features.repeat(continuous_time.shape[1], 0)
    return np.concatenate(
        [features, continuous_time.reshape(-1,1)], 
        axis = 1).reshape(continuous_time.shape[0], continuous_time.shape[1], -1 )

def get_loader(features, durations, events, processor, fit_y = False, time_scaler = 1, batch_size = 256, mode = 'test'):
    if fit_y:
        y = processor.fit_transform(durations, events)
    else:
        y = processor.transform(durations, events)
    continuous_times = y[2]

    x = add_time_continuous_features(features,continuous_times/time_scaler)

    dataset = SurvDataset(x, y, durations, events, continuous_times = continuous_times)

    if mode == 'train':
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return loader