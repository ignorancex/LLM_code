import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils import shuffle

from utils import load_data, introduce_missing_data, normalization, renormalization
from utils import compute_rmse

from knnxkde import KNNxKDE
from GAIN.gain import gain
from softimpute.softimpute import softimpute


LIST_DATASETS = [
    '2d_linear',
    '2d_sine',
    '2d_ring',
    'abalone',
    'breast',  # can have too many columns at high missing rates
    'gaussians',
    'geyser',
    'japanese_vowels',  # can have too many columns at high missing rates
    'penguin',
    'planets',
    'pollen',
    'sulfur',
    'sylvine',
    'wine_red',
    'wine_white',
]

# Set hyperparameters
OUT_DIR = 'output'
MISSING_SCENARIO = 'full_mcar'

NB_REPEAT = 20
LIST_MISS_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

LIST_TAUS = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
LIST_NB_NEIGHBORS = [1, 2, 5, 10, 20, 50, 100]
LIST_NB_TREES = [1, 2, 3, 5, 10, 15, 20]  # computationally expensive to go higher
LIST_NB_ITERS = [100, 200, 400, 700, 1000, 2000, 4000]
LIST_LAMBDAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


# Main loop

for i1 in range(len(LIST_DATASETS)):
    data_name = LIST_DATASETS[i1]
    original_data = load_data(data_name)
    original_data = shuffle(original_data)
    print(f'\n')
    print(f'=====================')
    print(f'|| ORIGINAL DATA NAME: {data_name} / SHAPE={original_data.shape}')
    print(f'=====================')
    print(f'({time.asctime()})')

    rmse_dict = {
        'knnxkde': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_TAUS))),
        'knnimputer': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_NB_NEIGHBORS))),
        'missforest': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_NB_TREES))),
        'softimpute': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_LAMBDAS))),
        'gain': np.zeros((len(LIST_MISS_RATES), NB_REPEAT, len(LIST_NB_ITERS))),
        'mice': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
        'mean': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
        'median': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
    }

    for i2 in range(len(LIST_MISS_RATES)):
        cur_miss_rate = LIST_MISS_RATES[i2]
        print(f'\n~~~ MISSING RATE = {cur_miss_rate} ~~~', flush=True)

        for i3 in range(NB_REPEAT):
            t0 = time.time()
            original_data = load_data(data_name)
            original_data = shuffle(original_data)

            miss_data = introduce_missing_data(
                original_data=original_data,
                miss_rate=cur_miss_rate,
                mode=MISSING_SCENARIO,
                data_name=data_name,
            )

            norm_miss_data, norm_params = normalization(miss_data)
            norm_original_data, _ = normalization(original_data, parameters=norm_params)

            for i4 in range(len(LIST_TAUS)):
                print(f'Iteration {i3+1}/{NB_REPEAT}', end=' => ')
                print(f'kNNxKDE... {i4+1}/{len(LIST_TAUS)}', end='\r', flush=True)
                cur_tau = 1.0/LIST_TAUS[i4]
                knnxkde = KNNxKDE(h=0.03, tau=cur_tau, metric='nan_std_eucl')
                norm_imputed_data = knnxkde.impute_mean(norm_miss_data)
                if norm_imputed_data is None:
                    rmse_dict['knnxkde'][i2, i3, i4] = np.nan
                else:
                    rmse = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)
                    rmse_dict['knnxkde'][i2, i3, i4] = rmse

            for i4 in range(len(LIST_NB_NEIGHBORS)):
                print(f'Iteration {i3+1}/{NB_REPEAT}', end=' => ')
                print(f'kNNImputer... {i4+1}/{len(LIST_NB_NEIGHBORS)}', end='\r', flush=True)
                cur_nb_neigh = LIST_NB_NEIGHBORS[i4]
                knnimputer = KNNImputer(n_neighbors=cur_nb_neigh)
                norm_imputed_data = knnimputer.fit_transform(norm_miss_data)
                rmse = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)
                rmse_dict['knnimputer'][i2, i3, i4] = rmse

            for i4 in range(len(LIST_NB_TREES)):
                print(f'Iteration {i3+1}/{NB_REPEAT}', end=' => ')
                print(f'MissForest... {i4+1}/{len(LIST_NB_TREES)}', end='\r', flush=True)
                cur_nb_trees = LIST_NB_TREES[i4]
                estimator = ExtraTreesRegressor(n_estimators=cur_nb_trees)
                missforest = IterativeImputer(estimator=estimator, max_iter=10, tol=2e-1, verbose=0)
                norm_imputed_data = missforest.fit_transform(norm_miss_data)
                rmse = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)
                rmse_dict['missforest'][i2, i3, i4] = rmse

            for i4 in range(len(LIST_LAMBDAS)):
                print(f'Iteration {i3+1}/{NB_REPEAT}', end=' => ')
                print(f'SoftImpute... {i4+1}/{len(LIST_LAMBDAS)}', end='\r', flush=True)
                cur_lambda = LIST_LAMBDAS[i4]
                norm_imputed_data = softimpute(norm_miss_data, cur_lambda)[1]
                rmse = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)
                rmse_dict['softimpute'][i2, i3, i4] = rmse

            for i4 in range(len(LIST_NB_ITERS)):
                print(f'Iteration {i3+1}/{NB_REPEAT}', end=' => ')
                print(f'GAIN... {i4+1}/{len(LIST_NB_ITERS)}', end='\r', flush=True)
                cur_nb_iters = LIST_NB_ITERS[i4]
                gain_parameters = {"batch_size": 128, "hint_rate": 0.9, "alpha": 100, "iterations": cur_nb_iters}
                norm_imputed_data = gain(norm_miss_data, gain_parameters)
                rmse = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)
                rmse_dict['gain'][i2, i3, i4] = rmse

            mice = IterativeImputer(estimator=BayesianRidge(), max_iter=10, tol=2e-1, verbose=0)
            norm_imputed_data = mice.fit_transform(norm_miss_data)
            rmse = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)
            rmse_dict['mice'][i2, i3] = rmse

            mean_imputer = SimpleImputer(strategy='mean')
            norm_imputed_data = mean_imputer.fit_transform(norm_miss_data)
            rmse = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)
            rmse_dict['mean'][i2, i3] = rmse

            median_imputer = SimpleImputer(strategy='median')
            norm_imputed_data = median_imputer.fit_transform(norm_miss_data)
            rmse = compute_rmse(norm_original_data, norm_miss_data, norm_imputed_data)
            rmse_dict['median'][i2, i3] = rmse
            
            t1 = time.time()
            print(f'                                                     ', end='\r')
            print(f'Iteration {i3+1}/{NB_REPEAT}', end=' -> ')
            print(f'time = {(t1-t0):.3f} s', flush=True)

    
    save_dir = f'{OUT_DIR}/{MISSING_SCENARIO}/rmse'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('\nSaving RMSE dictionary as pickle file...', end=' ')
    with open(f'{save_dir}/rmse_{data_name}.pkl', 'wb') as f:
        pickle.dump(rmse_dict, f)
    print('Saved!')
    print(f'({time.asctime()})')
    
print('FINISH!')
print('Bye \o/\o/')
