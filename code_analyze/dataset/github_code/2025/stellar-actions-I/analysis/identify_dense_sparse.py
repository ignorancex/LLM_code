##unlike radial bins script, it only identifies the dense, sparse and very sparse region stars and saves their J and A, 
## which can then be used to calculate action changes using the median_change.py script.

## can get action.h5 file from the following website and change the file paths here accordingly ('stellar-actions-I/data/actions.h5' in following code)
## https://www.mso.anu.edu.au/~arunima/stellar-actions-I-data/

import numpy as np
import h5py
from scipy.spatial import cKDTree

from utils import load_star_data_hdf5
from utils import convert_to_cartesian

k = 1
i = 101

#change path here if required
with h5py.File('stellar-actions-I/data/actions.h5', 'r') as f:
    J = f['actions'][:]
    C = f['coordinates'][:]
    V = f['velocities'][:]
    A = f['age'][:]
    M = f['mass'][:]
    IDs = f['ID'][:]


def compute_5th_nn_distances(positions):
    """
    Compute the 5th nearest neighbor distances for given positions.
    """
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=6)  # k=6 because the first neighbor is the point itself
    return distances[:, 5]

# Define your distance bins 
distance_bins = np.array([0.2,0.5,1, 2.35, 3.185869,1323.5])  # Example bins: 0.5 pc is smoothing length, 2.35 and 3.18 is 89-91 percentile and 1323.5 pc is 99th percentile of 5th NN distance
    
# Initialize lists to collect star IDs for the densest and sparsest regions
dense_region_ids = []
sparse_region_ids = []
very_sparse_region_ids =[]


for t in tqdm(np.arange(A.shape[0])):
    # Identify stars born in the last 1 Myr
    valid = ~np.isnan(A[t, :])  # Valid stars
    recently_born = (A[t, valid] < 1) & (A[t, valid] > 0)
    
    # Get the indices of these stars
    star_indices = np.where(valid)[0][recently_born]
    C_cart = convert_to_cartesian(C[t, valid, :][recently_born])
    
    # Compute 5th NN distances
    fifth_nn_distances = compute_5th_nn_distances(C_cart)
    
    # Bin the stars into distance bins
    bin_indices = np.digitize(fifth_nn_distances, bins=distance_bins)
    dense_bin_indices = np.where(bin_indices <= 1)[0]  # less than 0.5
    sparse_bin_indices = np.where(bin_indices == len(distance_bins-2))[0]  # between 89 and 91 percentile
    very_sparse_bin_indices = np.where(bin_indices == len(distance_bins))[0]  # greater than last bin edge

    
    # Collect IDs
    # Map indices to IDs
    dense_region_ids.extend(IDs[star_indices[dense_bin_indices]])
    sparse_region_ids.extend(IDs[star_indices[sparse_bin_indices]])
    very_sparse_region_ids.extend(IDs[star_indices[very_sparse_bin_indices]])


sparse_list=np.array(sparse_region_ids)
sparse_star_indices = np.where(np.isin(IDs, sparse_list))[0]
very_sparse_list=np.array(very_sparse_region_ids)
very_sparse_star_indices = np.where(np.isin(IDs, very_sparse_list))[0]
dense_list=np.array(dense_region_ids)
dense_star_indices = np.where(np.isin(IDs, dense_list))[0]

J_sparse = J[:,sparse_star_indices,:]/M[:,sparse_star_indices,np.newaxis]
A_sparse = A[:,sparse_star_indices]
J_very_sparse = J[:,very_sparse_star_indices,:]/M[:,very_sparse_star_indices,np.newaxis]
A_very_sparse = A[:,very_sparse_star_indices]
J_dense = J[:,dense_star_indices,:]/M[:,dense_star_indices,np.newaxis]
A_dense = A[:,dense_star_indices]

#change path here for all the following:
np.save('path_to_save/J_sparse.npy',J_sparse)
np.save('path_to_save/A_sparse.npy',A_sparse)
np.save('path_to_save/J_very_sparse.npy',J_very_sparse)
np.save('path_to_save/A_very_sparse.npy',A_very_sparse)
np.save('path_to_save/J_dense.npy',J_dense)
np.save('path_to_save/A_dense.npy',A_dense)
