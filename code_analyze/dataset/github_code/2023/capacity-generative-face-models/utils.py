# ///////////// Copyright 2023 Vishnu Boddeti. All rights reserved. /////////////
# //
# //   Project     : Capacity of Generative Face Models
# //   File        : utils.py
# //   Description : Utility functions for capacity estimation
# //
# //   Created On: 08/20/2023
# //   Created By: Gautam Sreekumar
# ////////////////////////////////////////////////////////////////////////////

import os
import pickle
import numpy as np
import pandas as pd
import math
from scipy import interpolate
import pandas as pd

from constants import *

import config
args = config.parse_args()

def get_parameters(face_model, ref_dataset):
    tmp = THRESHOLDS[face_model][ref_dataset]
    intra_class_angle = 1 - (tmp[0]/2.)
    far = [0.1, 1.0, 10.0]

    threshold_angles = [(180 / math.pi) * math.acos(1 - x / 2) / 2 for x in tmp]
    threshold = [math.cos(math.acos(1 - x / 2) / 2) for x in tmp]

    return intra_class_angle, far, threshold_angles, threshold

def load_features(dataset, model, metadata_file=None, root_path=None):
    file = os.path.join(args.feature_path, FEAT_FILES[model][dataset])
    data = pickle.load(open(file, 'rb'))
    X = data['feats']
    
    # Remove the rows corresponding to faces not detected
    index = np.where(np.sum(np.abs(X), axis=1) == 0)
    X = np.delete(X, index, axis=0)

    df_data = {}
    if metadata_file is not None:
        df_data['age'] = data['age']
        df_data['gender'] = data['gender']
        df_data['det_score'] = data['det_score']
        df_data['fnames'] = data['fnames']
        if "ids" in data.keys():
            df_data['ids'] = data['ids']
    else:
        file = os.path.join(args.feature_path, FEAT_FILES['arcface'][dataset])
        if root_path is not None:
            file = os.path.join(root_path, file)
        if os.path.exists(file):
            data_tmp = pickle.load(open(file, 'rb'))
            df_data['age'] = data_tmp['age']
            df_data['gender'] = data_tmp['gender']
            df_data['det_score'] = data_tmp['det_score']
            df_data['fnames'] = data_tmp['fnames']
            if "ids" in data_tmp.keys():
                df_data['ids'] = data_tmp['ids']
        else:
            df_data['age'] = np.zeros(data["feats"].shape[0])
            df_data['gender'] = np.zeros(data["feats"].shape[0])
            df_data['det_score'] = np.zeros(data["feats"].shape[0])
            df_data['fnames'] = np.zeros(data["feats"].shape[0])
            if "ids" in data.keys():
                df_data['ids'] = np.zeros(data["feats"].shape[0])

    df_data['age'] = np.delete(df_data['age'], index)
    df_data['gender'] = np.delete(df_data['gender'], index)
    df_data['det_score'] = np.delete(df_data['det_score'], index)
    df_data['fnames'] = np.delete(df_data['fnames'], index)
    if "ids" in df_data.keys():
        df_data['ids'] = np.delete(df_data['ids'], index)

    df = pd.DataFrame(df_data)

    return X, df

def get_raw_values(dataset, ref_dataset, face_model, spec_key=None,
                   as_string=False):
    _1, fars, _3, threshold = get_parameters(face_model, ref_dataset)
    if spec_key is None:
        fname = os.path.join(
            args.output_path,
            'pkl_files',
            dataset,
            f'{face_model}_{ref_dataset}_capacity.pkl'
        )
    else:
        fname = os.path.join(
            args.output_path,
            'pkl_dir', 
            dataset,
            f'{face_model}_{ref_dataset}_capacity_{spec_key}.pkl'
        )
    data = pd.read_pickle(fname)

    k_ = list(data.keys())
    for k in k_:
        if not isinstance(data[k], pd.Series):
            data[k] = pd.Series((_ for _ in data[k]))

    capacity = data["capacity"]
    cos_delta = data["cos_delta"]
    fn = interpolate.interp1d(cos_delta, capacity, kind='slinear')

    try:
        val1 = fn(threshold[0])
    except ValueError as e:
        if "above the interpolation range." in str(e):
            val1 = np.nan
        else:
            raise ValueError(str(e))

    try:
        val2 = fn(threshold[1])
    except ValueError as e:
        if "above the interpolation range." in str(e):
            val2 = np.nan
        else:
            raise ValueError(str(e))

    try:
        val3 = fn(threshold[2])
    except ValueError as e:
        if "above the interpolation range." in str(e):
            val3 = np.nan
        else:
            raise ValueError(str(e))
    
    cap_at_fars = np.array((val1, val2, val3))

    if spec_key is not None:
        specific = data[spec_key]
        if as_string:
            return _stringify(cos_delta), _stringify(capacity), \
                   _stringify(fars), _stringify(cap_at_fars), \
                   _stringify(specific)
        else:
            return cos_delta, capacity, fars, cap_at_fars, specific
    else:
        if as_string:
            return _stringify(cos_delta), _stringify(capacity), \
                   _stringify(fars), _stringify(cap_at_fars)
        else:
            return cos_delta, capacity, fars, cap_at_fars

def _stringify(x):
    return [f"{_:.3E}" for _ in x]