from __future__ import print_function, division

from copy import deepcopy
import numpy as np

__all__ = ['filter_samples_outwith_control_limits', 'subsample_dataset']

# Filters samples that do not adhere to control limits
def filter_samples_outwith_control_limits(samples_X, samples_U, control_limit_low, control_limit_high, debug=False):
    samples_to_remove = []
    assert samples_X.shape[0] == samples_U.shape[0]
    sample_dim = samples_X.shape[0]
    debug and print("Sample size before:", sample_dim)
    for i in range(sample_dim):
        if np.any(samples_U[i,:,:]>control_limit_high):
            debug and print(i, "exceeds UPPER limit", np.max(samples_U[i,:,:]))
            samples_to_remove.append(i)
        elif np.any(samples_U[i,:,:]<control_limit_low):
            debug and print(i, "exceeds LOWER limit", np.min(samples_U[i,:,:]))
            samples_to_remove.append(i)

    debug and print("Samples to remove", samples_to_remove)

    samples_U_filtered = np.delete(samples_U, samples_to_remove, axis=0)
    samples_X_filtered = np.delete(samples_X, samples_to_remove, axis=0)
    debug and print("Sample size after:", samples_U_filtered.shape[0])
    #samples_final_cost = np.delete(samples_final_cost, samples_to_remove, axis=0)
    return samples_X_filtered, samples_U_filtered

# Subsamples dataset (to reduce number of samples for filtration)
def subsample_dataset(samples_X, samples_U, subsampling_step=1, debug=False):
    debug and print("Sample size before:", samples_U.shape[0])
    samples_X_subsampled = samples_X[::subsampling_step,:,:]
    samples_U_subsampled = samples_U[::subsampling_step,:,:]
    debug and print("Sample size after:", samples_U_subsampled.shape[0])
    return samples_X_subsampled, samples_U_subsampled

def stack_samples_in_vector(samples_X, debug=False):
    sample_dim = samples_X.shape[0]
    state_dim = samples_X.shape[1]
    time_dim = samples_X.shape[2]
    dataset = np.zeros((sample_dim * time_dim, state_dim), dtype=np.double)
    debug and print(dataset.shape, sample_dim, state_dim, time_dim)
    for i in range(sample_dim):
        for j in range(state_dim):
            dataset[i * time_dim:(i+1) * time_dim,j] = samples_X[i,j,:].T
    return dataset

def subsample_stacked_vector(dataset, subsampling_step=1, debug=False):
    debug and print("Size before:", dataset.shape)
    X = dataset[0::subsampling_step,:]
    debug and print("Size after:", X.shape)
    return X
