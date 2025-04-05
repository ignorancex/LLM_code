
from functools import partial

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .utils import index_at_time
import torch

def _cut_continuous_time(duration, spacing, n_cuts):
    n_cuts = n_cuts+1
    cuts = np.arange(duration, 0, -spacing)[::-1]

    # insert time zero
    cuts = np.concatenate([[0], cuts])

    number = n_cuts - len(cuts) + 1
    index = len(cuts)-1


    if len(cuts) == 0:
        return (np.arange(0,number)*spacing)[1:], -1
    
    if n_cuts == len(cuts):
        return cuts, index


    cuts = np.concatenate([cuts, (np.arange(0,number)*spacing + cuts[-1])[1:]])

    return cuts, index

def first_frac_and_index(duration, spacing, n_cuts):
    cuts, index = _cut_continuous_time(duration, spacing, n_cuts)
    frac_at_first = cuts[1] / spacing

    return frac_at_first, index

def cut_continuous_time(duration, spacing, n_cuts):
    cuts, _ = _cut_continuous_time(duration, spacing, n_cuts)
    return cuts


def _cut_continuous_time_covarying(durations, spacing):

    all_cuts = []
    index = []
    for i in range(len(durations)-1, 0, -1):
        cuts = np.arange(durations[i], durations[i-1], -spacing)
        index.append(len(cuts))
        all_cuts.extend(cuts)

    index = np.cumsum(index[::-1])

    index = np.concatenate(([0], index))
    all_cuts.append(0)
    assert len(index) == len(durations)
    return all_cuts[::-1], index

def cut_continuous_time_covarying(all_durations, spacing, max_len = None):
    new_durations = []
    index_at_features = []
    
    

    for durations in all_durations:
        d, index = _cut_continuous_time_covarying(durations, 50)
        new_durations.append(d)
        index_at_features.append(index)

    if max_len is not None:
        new_durations = [torch.nn.functional.pad(torch.tensor(x), pad=(0, max_len - len(x)), mode='constant', value=0) for x in new_durations]

    return new_durations, index_at_features

def fill_missing_features(features, duration_list, max_len):

    imputed_feature = features[0].reshape(1,-1)
    for i in range(1, len(duration_list)):
        n_missing = duration_list[i] - duration_list[i-1] -1

        if n_missing>0:
            imputed_feature = np.concatenate([imputed_feature, features[i-1].reshape(1,-1).repeat(n_missing, axis=0)])
        imputed_feature = np.concatenate([imputed_feature, features[i].reshape(1,-1)])
        

    try:
        assert len(imputed_feature) == duration_list[-1]+1
    except:

        raise ValueError('assertion error data not equal')
    n_missing = max_len - len(imputed_feature)
    if n_missing > 0:
        imputed_feature = np.concatenate([imputed_feature, features[-1].reshape(1,-1).repeat(n_missing, axis=0)])
    assert len(imputed_feature) == max_len
    return imputed_feature


class CTAddedSameSpacing:
    def __init__(self, n_cuts):

        self.n_cuts = n_cuts
        self.horizon = None
    
    @staticmethod
    def get_cut(horizon, n_time, insert_time, event):
        # horizon = np.quantile(durations[events ==1], np.linspace(0,1,n_time-3))
        if insert_time not in horizon:
            horizon = np.concatenate([horizon, [insert_time]])
        else:
            horizon = np.concatenate([horizon, [ horizon[1]/2 ] ])
        horizon = np.sort(horizon)
        
        duration_index = np.where(horizon == insert_time)[0][0]

        return horizon, duration_index

    def fit(self, durations, events):
        self.max_durations = np.max(durations)
        return self

    def transform(self, durations, events):
        horizon = np.linspace(0, self.max_durations, self.n_cuts-1)
        horizons = []
        duration_indices = []
        for duration, event in zip(durations, events):
            _horizon, idx = self.get_cut(horizon, self.n_cuts, duration, event)
    
            duration_indices.append(idx)
            horizons.append(_horizon)

        duration_indices = np.array(duration_indices)
        horizons = np.array(horizons)

        return duration_indices.astype(np.int64), events.astype(np.int64), horizons

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)

class CTCutEqualSpacing:
    def __init__(self, n_cuts):

        self.n_cuts = n_cuts
        self.horizon = None

    def fit(self, durations, events):
        return self

    def transform(self, durations, events):
        duration_indices = []
        horizons = []
        for duration, event in zip(durations, events):
            idx = self.n_cuts-1

            duration_indices.append(idx)

            horizons.append(np.linspace(0, duration, self.n_cuts))

        duration_indices = np.array(duration_indices)
        horizons = np.array(horizons)
        return duration_indices.astype(np.int64), events.astype(np.int64), horizons

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)

def label_encoder(features, cat_features):
    vocab_size = 0
    for _,feat in enumerate(cat_features):
        features[:, feat] = LabelEncoder().fit_transform(features[:, feat]).astype(float) + vocab_size
        vocab_size = features[:, feat].max() + 1
    return features, vocab_size

def standard_scaler(features, num_features: list, scaler, fit = False):
    if fit:
        scaler.fit(features[:, num_features])
    features[:, num_features] = scaler.transform(features[:, num_features])
    return features





