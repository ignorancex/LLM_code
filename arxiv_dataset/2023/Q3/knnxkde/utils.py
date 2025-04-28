import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

LIST_MISS_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # for the select_best_hyperparam function
LIST_TAUS = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
LIST_NB_NEIGHBORS = [1, 2, 5, 10, 20, 50, 100]
LIST_NB_TREES = [1, 2, 3, 5, 10, 15, 20]



def load_data(data_name):
    if data_name=='2d_linear':
        data = pd.read_csv('data/2d_linear.csv', header=None)
        np_data = np.array(data.iloc[:, :]).astype('float32')
    elif data_name=='2d_ring':
        data = pd.read_csv('data/2d_ring.csv', header=None)
        np_data = np.array(data.iloc[:, :]).astype('float32')
    elif data_name=='2d_sine':
        data = pd.read_csv('data/2d_sine.csv', header=None)
        np_data = np.array(data.iloc[:, :]).astype('float32')
    elif data_name=='abalone':
        data = pd.read_csv('data/abalone.csv', header=None)
        np_data = np.array(data.iloc[:, 1:-1]).astype('float32')
    elif data_name=='breast':
        data = pd.read_csv('data/breast.csv', header=None)
        np_data = np.array(data.iloc[:, 2:]).astype('float32')
    elif data_name=='gaussians':
        data = pd.read_csv('data/gaussians.csv')
        np_data = np.array(data.iloc[:, 1:-1]).astype('float32')
    elif data_name=='geyser':
        data = pd.read_csv('data/geyser.csv')
        np_data = np.array(data.iloc[:, 1:-1]).astype('float32')
    elif data_name=='japanese_vowels':
        data = pd.read_csv('data/japanese_vowels.csv')
        np_data = np.array(data.iloc[:, 3:]).astype('float32')
    elif data_name=='penguin':
        data = pd.read_csv('data/penguin.csv')
        np_data = np.array(data.iloc[:, 3:-1]).astype('float32')
    elif data_name=='planets':
        data = pd.read_csv('data/planets.csv')
        np_data = np.array(data.iloc[:, :]).astype('float32')
    elif data_name=='pollen':
        data = pd.read_csv('data/pollen.csv', header=None)
        np_data = np.array(data.iloc[:, :-1]).astype('float32')
    elif data_name=='sulfur':
        data = pd.read_csv('data/sulfur.csv', header=None)
        np_data = np.array(data.iloc[:, :]).astype('float32')
    elif data_name=='sylvine':
        data = pd.read_csv('data/sylvine.csv', header=None)
        np_data = np.array(data.iloc[:, 1:]).astype('float32')
    elif data_name=='wine_red':
        data = pd.read_csv('data/wine_red.csv', delimiter=';')
        np_data = np.array(data.iloc[:, :-1]).astype('float32')
    elif data_name=='wine_white':
        data = pd.read_csv('data/wine_white.csv', delimiter=';')
        np_data = np.array(data.iloc[:, :-1]).astype('float32')
    else:
        print('ERROR: data_name does not exist.')
        np_data = None
    return np_data



def select_columns(data_name):
    if data_name=='2d_linear':
        miss_col = 1  # y axis
        cond_col = 0  # x axis
    elif data_name=='2d_ring':
        miss_col = 1  # y axis
        cond_col = 0  # x axis
    elif data_name=='2d_sine':
        miss_col = 1  # y axis
        cond_col = 0  # x axis
    elif data_name=='abalone':
        miss_col = 0  # shell length
        cond_col = 1  # shell diameter
    elif data_name=='breast':
        miss_col = 0  # cell radius mean
        cond_col = 1  # cell radius std-dev
    elif data_name=='gaussians':
        miss_col = 0  # first component
        cond_col = 1  # second component
    elif data_name=='geyser':
        miss_col = 0  # eruption time
        cond_col = 1  # waiting time
    elif data_name=='japanese_vowels':
        miss_col = 0  # coefficient 1
        cond_col = 1  # coefficient 2
    elif data_name=='penguin':
        miss_col = 3  # body mass in g
        cond_col = 2  # flipper length in mm
    elif data_name=='planets':
        miss_col = 1  # planet mass
        cond_col = 0  # planet radius
    elif data_name=='pollen':
        miss_col = 3  # weight
        cond_col = 4  # density
    elif data_name=='sulfur':
        miss_col = 0  # feature a1
        cond_col = 1  # feautre a2
    elif data_name=='sylvine':
        miss_col = 0  # feature V1
        cond_col = 1  # feature V2
    elif data_name=='wine_red':
        miss_col = 0  # fixed acidity
        cond_col = 1  # volatile acidity
    elif data_name=='wine_white':
        miss_col = 0  # fixed acidity
        cond_col = 1  # volatile acidity
    else:
        print('ERROR: data_name does not exist or has no labels.')
        miss_col = None
        cond_col = None
    return miss_col, cond_col



def introduce_missing_data(original_data, miss_rate, mode, data_name):
    """Introduce missing data following 'mode' scenario.
    For 'full_mcar', if an observation has all features removed, do it again.
    For 'mcar', 'mar', or 'mnar', only one feature is altered.
    Args:
        - original_data: original data, shape (n, d)
        - miss_rate: missing rate between 0 and 1
        - mode: one of 'full_mcar', mcar', 'mar', 'mnar'
        - data_name: dataset name to specify relevent columns.
    Return:
        - miss_data: original data with introduced NaN, shape (n, d)
    """
    n, d = original_data.shape
    
    if mode=='full_mcar':
        miss_mask = np.zeros((n, d), dtype=bool)
        for i in range(n):
            current_mask = np.random.uniform(low=0.0, high=1.0, size=d) < miss_rate
            while np.logical_and.reduce(current_mask):
                current_mask = np.random.uniform(low=0.0, high=1.0, size=d) < miss_rate
            miss_mask[i] = current_mask
        miss_data = np.where(miss_mask, np.nan, original_data)
    
    elif mode in ['mcar', 'mar', 'mnar']:
        miss_col, cond_col = select_columns(data_name=data_name)  # missing column, conditional column (for mar)
        if mode=='mcar':
            miss_mask = np.random.uniform(low=0.0, high=1.0, size=n) < miss_rate
        elif mode=='mar':
            argsort = np.argsort(original_data[:, cond_col])  # use cond_col for MAR
            ranks = np.argsort(argsort)
            my_miss_probs = miss_rate + ((ranks / n) - 0.5) * 0.2  # between -10% and +10% of miss_rate
            miss_mask = np.random.uniform(low=0.0, high=1.0, size=n) < my_miss_probs
        elif mode=='mnar':
            argsort = np.argsort(original_data[:, miss_col])  # use miss_col for MNAR
            ranks = np.argsort(argsort)
            my_miss_probs = miss_rate + ((ranks / n) - 0.5) * 0.2  # between -10% and +10% of miss_rate
            miss_mask = np.random.uniform(low=0.0, high=1.0, size=n) < my_miss_probs
        miss_data = np.copy(original_data)
        miss_data[miss_mask, miss_col] = np.nan
    
    else:
        print('Missing data scenario not recognized!')
        return None
    
    return miss_data



def normalization(data, parameters=None):
    """Normalize data in the range [0, 1].
    Args:
        - data: original data, shape (n, d)
        - parameters: if None, default is min/max normalization
    Return:
        - norm_data: normalized data in [0, 1]
        - norm_parameters: min_val and max_val used for each column, shape (n, d)
    """
    _, dim = data.shape
    norm_data = data.copy()
    
    if parameters is None:
  
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
    
        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
        # Return norm_parameters for renormalization
        norm_parameters = {"min_val": min_val, "max_val": max_val}

    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]
    
        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
        norm_parameters = parameters
    
    return norm_data, norm_parameters



def renormalization(norm_data, norm_parameters):
    """Renormalize data from [0, 1] back to the original range.
    Args:
        - norm_data: normalized data, shape (n, d)
        - norm_parameters: min_val and max_val used for each column
    Return:
        - renorm_data: renormalized data in the original range, shape (n, d)
    """
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
    return renorm_data



def compute_rmse(original_data, miss_data, imputed_data):
    """Compute the RMSE.
    Args:
        - original_data: shape (n, d)
        - miss_data: shape (n, d)
        - imputed_data: shape (n, d)
    Return:
        - rmse: value of the RMSE
    """
    nb_miss = np.sum(np.isnan(miss_data))
    sum_of_squares = np.sum((original_data - imputed_data)**2)
    rmse = np.sqrt(sum_of_squares / nb_miss)
    return rmse



def select_best_hyperparam(data_name, method, missing_rate, dir_name):
    """Extract the best hyperparameter for a given dataset, a given method
    and a given missing rate. Looks the lowest NRMSE to determine best
    hyperparameter.Used when computing the likelihood.
    Args:
        - data_name: string of the dataset name.
        - method: string for the name of the method at use.
        One of ['knnxkde', 'knnimputer', 'missforest', 'mice', 'mean'].
        - missing_rate: value of the missing rate.
        One of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].
        - dir_name: string of the directory name where to look for
        the minimum RMSE.
    Return:
        - value of the best hyperparameter.
    """
    with open(f'{dir_name}/rmse/rmse_{data_name}.pkl', 'rb') as f:
        rmse_dict = pickle.load(f)
    miss_rate_idx = LIST_MISS_RATES.index(missing_rate)
    mean_rmse = np.mean(rmse_dict[method][miss_rate_idx], axis=0)
    best_hyperparam_idx = np.argmin(mean_rmse)
    if method=='knnxkde':
        return LIST_TAUS[best_hyperparam_idx]
    elif method=='knnimputer':
        return LIST_NB_NEIGHBORS[best_hyperparam_idx]
    elif method=='missforest':
        return LIST_NB_TREES[best_hyperparam_idx]
    else:
        print('Method not recognized.')
        return None



def compute_sample_likelihood(norm_original_data, method, cells_distrib):
    """Compute the value of the p.d.f. for each missing value in the
    imputed sample given a specific method name.
    Args:
        - norm_original_data: original data normalized in [0, 1], shape (n, d)
        - method: string for the name of the method at use.
        One of ['knnxkde', 'knnimputer', 'missforest', 'mice', 'mean'].
        - cells_distrib: dictionary of (mean, std) for each imputed cell, or
        list of the imputation sample in the case of kNNxKDE.
    Return:
        - list_likelihood: p.d.f. computed in each point, can be multiplied
        to obtain the value of the log-likehood for the imputed sample.
    """
    list_likelihood = []
    x_axis = np.linspace(start=-0.1, stop=1.1, num=121)  # from -0.1 to 1.1 with step=0.01
    for i, key in enumerate(cells_distrib.keys()):
        if method=='knnxkde':
            my_bins = np.linspace(start=-0.1, stop=1.1, num=122)  # so that y_axis has shape (121,)
            y_axis, _ = np.histogram(cells_distrib[key], bins=my_bins, density=True)
        elif method in ['knnimputer', 'missforest', 'mice', 'mean']:
            mu = cells_distrib[key][0]
            sigma = cells_distrib[key][1]
            y_axis = norm.pdf(x_axis, loc=mu, scale=sigma+1e-10)
        else:
            print('Unsupported method name to compute the likelihood.')
        
        ground_truth = norm_original_data[key[0], key[1]]
        ii = (x_axis <= ground_truth).sum()  # interesting index
        if ii==0:
            likelihood = y_axis[0]
        elif ii==121:
            likelihood = y_axis[120]
        else:
            likelihood = (y_axis[ii-1] + y_axis[ii]) / 2.0
        list_likelihood.append(likelihood)
    return np.array(list_likelihood)