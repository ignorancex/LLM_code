import numpy as np

from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge


def distribution_with_knnimputer(norm_miss_data, nb_neigh):
    """Estimate mean and std of the imputed cell with kNN-Imputer.
    This computes the mean and std-dev of the `nb_neigh` cells
    used for imputation.
    Args:
        - norm_miss_data: normalized in [0, 1], shape (n, d)
        - nb_neigh: nb of neighbors for the kNN-Imputer.
    Returns:
        - cell_distrib: dictionary with (mean, std) for each cell.
        Will be used assuming a Gaussian distribution.
    """
    (n, d) = norm_miss_data.shape
    cells_distrib = dict()

    for col in range(d):
        id_receivers = np.where(np.isnan(norm_miss_data[:, col]))[0]
        id_givers = np.where(~np.isnan(norm_miss_data[:, col]))[0]
        if id_receivers.shape[0]==0:  # move to next column if nothing is missing
            continue
        
        data_receivers = norm_miss_data[id_receivers]
        data_givers = norm_miss_data[id_givers]
        
        d_ij = nan_euclidean_distances(data_receivers, data_givers)
        for i in range(len(id_receivers)):
            id_selected_neighbors = np.argsort(d_ij[i])[:nb_neigh]
            values = data_givers[id_selected_neighbors, col]
            temp = (np.mean(values), np.std(values))
            cells_distrib[(id_receivers[i], col)] = temp

    return cells_distrib



def distribution_with_missforest(norm_miss_data, nb_trees):
    """Estimate mean and std of the imputed cell using MissForest.
    This repeats the imputation 5 times, then computes the mean
    and the std-dev of the 5 imputated datasets.
    Args:
        - norm_miss_data: normalized in [0, 1], shape (n, d)
        - nb_trees: nb of trees for MissForest.
    Returns:
        - cell_distrib: dictionary with (mean, std) for each cell.
        Will be used assuming a Gaussian distribution.
    """
    (n, d) = norm_miss_data.shape
    cells_distrib = dict()
    
    norm_imputed_data = np.zeros((5, n, d))
    for i in range(5):
        estimator = ExtraTreesRegressor(n_estimators=nb_trees)
        missforest = IterativeImputer(estimator=estimator, max_iter=10, tol=2e-1, verbose=0)
        norm_imputed_data[i] = missforest.fit_transform(norm_miss_data)

    miss_mask = np.isnan(norm_miss_data)
    for i1 in range(n):
        for i2 in range(d):
            if miss_mask[i1, i2]:
                values = norm_imputed_data[:, i1, i2]
                temp = (np.mean(values), np.std(values))
                cells_distrib[(i1, i2)] = temp

    return cells_distrib



def distribution_with_mice(norm_miss_data):
    """Estimate mean and std of the imputed cell using MICE.
    This repeats the imputation 5 times, then computes the mean
    and the std-dev of the 5 imputated datasets.
    Args:
        - norm_miss_data: normalized in [0, 1], shape (n, d)
    Returns:
        - cell_distrib: dictionary with (mean, std) for each cell.
        Will be used assuming a Gaussian distribution.
    """
    (n, d) = norm_miss_data.shape
    cells_distrib = dict()
    
    norm_imputed_data = np.zeros((5, n, d))
    for i in range(5):
        mice = IterativeImputer(estimator=BayesianRidge(), max_iter=10, tol=2e-1, verbose=0, sample_posterior=True)
        norm_imputed_data[i] = mice.fit_transform(norm_miss_data)
    
    miss_mask = np.isnan(norm_miss_data)
    for i1 in range(n):
        for i2 in range(d):
            if miss_mask[i1, i2]:
                values = norm_imputed_data[:, i1, i2]
                temp = (np.mean(values), np.std(values))
                cells_distrib[(i1, i2)] = temp
    
    return cells_distrib



def distribution_with_mean(norm_miss_data):
    """Estimate mean and std of the imputed cell with Mean imputation.
    This computes the mean and std-dev of each column while ignoring
    NaN values.
    Args:
        - norm_miss_data: normalized in [0, 1], shape (n, d)
    Returns:
        - cell_distrib: dictionary with (mean, std) for each cell.
        Will be used assuming a Gaussian distribution.
    """
    (n, d) = norm_miss_data.shape
    cells_distrib = dict()

    for col in range(d):
        temp = (np.nanmean(norm_miss_data[:, col]), np.nanstd(norm_miss_data[:, col]))
        for i in range(n):
            if np.isnan(norm_miss_data[i, col]):
                cells_distrib[(i, col)] = temp
    
    return cells_distrib