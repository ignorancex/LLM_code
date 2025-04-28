import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from scipy.stats import norm

from utils import load_data
from utils import introduce_missing_data, normalization, renormalization
from utils import select_best_hyperparam, compute_sample_likelihood

from knnxkde import KNNxKDE
from probabilistic_standard_methods import distribution_with_knnimputer
from probabilistic_standard_methods import distribution_with_missforest
from probabilistic_standard_methods import distribution_with_mice
from probabilistic_standard_methods import distribution_with_mean


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
    'planets',  # no labels
    'pollen',  # no labels
    'sulfur',  # no labels
    'sylvine',
    'wine_red',
    'wine_white'
]


# Set hyperparameters
OUT_DIR = 'output'
MISSING_SCENARIO = 'full_mcar'

NB_REPEAT = 20
LIST_MISS_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

SMALL_EPS = 1e-5  # used to avoid zero-likelihood points


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

    loglik_dict = {
        'knnxkde': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
        'knnimputer': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
        'missforest': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
        'mice': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
        'mean': np.zeros((len(LIST_MISS_RATES), NB_REPEAT)),
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

            print(f'Iteration {i3+1}/{NB_REPEAT}', end=' => ')
            print(f'kNNxKDE...', end='\r', flush=True)
            cur_tau = select_best_hyperparam(
                data_name=data_name,
                method='knnxkde',
                missing_rate=cur_miss_rate,
                dir_name=f'{OUT_DIR}/{MISSING_SCENARIO}',
            )
            knnxkde = KNNxKDE(h=0.03, tau=1.0/cur_tau, metric='nan_std_eucl')
            cells_distrib_knnxkde = knnxkde.impute_samples(norm_miss_data, nb_draws=10000)
            if cells_distrib_knnxkde is None:
                loglik_dict['knnxkde'][i2, i3] = np.nan
            else:
                list_likelihood = compute_sample_likelihood(
                    norm_original_data=norm_original_data,
                    method='knnxkde',
                    cells_distrib=cells_distrib_knnxkde,
                )
                log_lik = np.mean(np.log(list_likelihood+SMALL_EPS))
                loglik_dict['knnxkde'][i2, i3] = log_lik

            print(f'Iteration {i3+1}/{NB_REPEAT}', end=' => ')
            print(f'kNNImputer...', end='\r', flush=True)
            cur_nb_neigh = select_best_hyperparam(
                data_name=data_name,
                method='knnimputer',
                missing_rate=cur_miss_rate,
                dir_name=f'{OUT_DIR}/{MISSING_SCENARIO}',
            )
            cells_distrib_knnimputer = distribution_with_knnimputer(norm_miss_data, nb_neigh=cur_nb_neigh)
            list_likelihood = compute_sample_likelihood(
                norm_original_data=norm_original_data,
                method='knnimputer',
                cells_distrib=cells_distrib_knnimputer,
            )
            log_lik = np.mean(np.log(list_likelihood+SMALL_EPS))
            loglik_dict['knnimputer'][i2, i3] = log_lik

            print(f'Iteration {i3+1}/{NB_REPEAT}', end=' => ')
            print(f'MissForest...', end='\r', flush=True)
            cur_nb_trees = select_best_hyperparam(
                data_name=data_name,
                method='missforest',
                missing_rate=cur_miss_rate,
                dir_name=f'{OUT_DIR}/{MISSING_SCENARIO}',
            )
            cells_distrib_missforest = distribution_with_missforest(norm_miss_data, nb_trees=cur_nb_trees)
            list_likelihood = compute_sample_likelihood(
                norm_original_data=norm_original_data,
                method='missforest',
                cells_distrib=cells_distrib_missforest,
            )
            log_lik = np.mean(np.log(list_likelihood+SMALL_EPS))
            loglik_dict['missforest'][i2, i3] = log_lik

            cells_distrib_mice = distribution_with_mice(norm_miss_data)
            list_likelihood = compute_sample_likelihood(
                norm_original_data=norm_original_data,
                method='mice',
                cells_distrib=cells_distrib_mice,
            )
            log_lik = np.mean(np.log(list_likelihood+SMALL_EPS))
            loglik_dict['mice'][i2, i3] = log_lik

            cells_distrib_mean = distribution_with_mean(norm_miss_data)
            list_likelihood = compute_sample_likelihood(
                norm_original_data=norm_original_data,
                method='mean',
                cells_distrib=cells_distrib_mean,
            )
            log_lik = np.mean(np.log(list_likelihood+SMALL_EPS))
            loglik_dict['mean'][i2, i3] = log_lik

            t1 = time.time()
            print(f'                                                     ', end='\r')
            print(f'Iteration {i3+1}/{NB_REPEAT}', end=' -> ')
            print(f'time = {(t1-t0):.3f} s', flush=True)

    save_dir = f'{OUT_DIR}/{MISSING_SCENARIO}/loglik'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('\nSaving Log-likelihood dictionary as pickle file...', end=' ')
    with open(f'{save_dir}/loglik_{data_name}.pkl', 'wb') as f:
        pickle.dump(loglik_dict, f)
    print('Saved!')
    print(f'({time.asctime()})')
    
print('FINISH!')
print('Bye \o/\o/')