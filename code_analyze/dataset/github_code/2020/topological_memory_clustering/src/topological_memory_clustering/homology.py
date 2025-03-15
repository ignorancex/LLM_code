from __future__ import print_function, division

import numpy as np

from ripser import ripser
from persim import plot_diagrams

from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

from time import time
from copy import deepcopy

# C++ Variants that are much faster
from homology_clustering_py import trajectory_segment_distance, trajectory_mod, get_pairwise_trajectory_distance_matrix_simple

__all__ = ['stacked_vector_to_preprocessed_distance_matrix',
           'trajectory_mod',
           'trajectory_mod_python',
           'getGreedyPerm',
           'getApproxSparseDM',
           'plot_filtration_diagrams',
           'trajectory_segment_distance',
           'compute_homology_filtration',
           'compute_filtration_from_preprocessed_distance_matrix',
           'trajectory_segment_distance_python',
           'get_pairwise_trajectory_distance_matrix',
           'get_pairwise_trajectory_distance_matrix_simple',
           'get_pairwise_trajectory_distance_matrix_simple_python',
           'get_cluster_labels_from_pairwise_trajectory_distance_matrix',
           'get_num_classes_from_h1',
           # tmp
           'd_pair',
           'd_pair_cpp',
           ]

def stacked_vector_to_preprocessed_distance_matrix(samples_X, X, subsampling_step, connect_start, connect_end, debug=False):
    sample_dim = samples_X.shape[0]
    state_dim = samples_X.shape[1]
    time_dim = samples_X.shape[2]
    
    # Pair-wise distances
    tic = time()
    D = pairwise_distances(X, metric='euclidean')
    toc = time()
    debug and print("Pairwise distance took", toc-tic)
    debug and print(X.shape, samples_X.shape, time_dim, subsampling_step)

    # Postprocess D as trajectory
    tic = time()
    trajectory_mod(D, (sample_dim, int(time_dim/subsampling_step), X.shape[1]), connect_start, connect_end)
    toc = time()
    debug and print("Trajectory mod took", toc-tic)

    return D

# ~25x faster
from homology_clustering_py import trajectory_mod

def trajectory_mod_python(D, shape, connect_start=False, connect_end=True):
    val = 0. #1e-14
    if D.shape[0] != shape[0]*shape[1]:
        raise ValueError('Wrong shape: D-rows={0} when given {1}x{2}'.format(D.shape[0], shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(i+1, shape[0]):
            if connect_start:
                D[i*shape[1], j*shape[1]] = val
                D[j*shape[1], i*shape[1]] = val
            if connect_end:
                D[(i+1)*shape[1] -1, (j+1)*shape[1] -1] = val
                D[(j+1)*shape[1] -1, (i+1)*shape[1] -1] = val
        for j in range(shape[1]-1):
                D[i*shape[1]+j, i*shape[1]+j+1] = val
                D[i*shape[1]+j+1, i*shape[1]+j] = val

def getGreedyPerm(D_in):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points

    Return
    ------
    lamdas: list
        Insertion radii of all points
    """
    
    D = deepcopy(D_in) # make sure we aren't modifying D

    N = D.shape[0]
    #By default, takes the first point in the permutation to be the
    #first point in the point cloud, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]

def getApproxSparseDM(lambdas_in, eps, D_in):
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight

    Parameters
    ----------
    lambdas: list
        insertion radii for points
    eps: float
        epsilon approximation constant
    D: ndarray
        NxN distance matrix, okay to modify because last time it's used

    Return
    ------
    DSparse: scipy.sparse
        A sparse NxN matrix with the reweighted edges
    """
    lambdas = deepcopy(lambdas_in)
    D = deepcopy(D_in)
    
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps

    # Create initial sparse list candidates (Lemma 6)
    # Search neighborhoods
    nBounds = ((eps**2+3*eps+2)/eps)*lambdas

    # Set all distances outside of search neighborhood to infinity
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]

    #Prune sparse list and update warped edge lengths (Algorithm 3 pg. 14)
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    # Rule out edges between vertices whose balls stop growing before they touch
    # or where one of them would have been deleted.  M stores which of these
    # happens first
    M = np.minimum((E0 + E1)*minlam, E0*(minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    # Now figure out the metric of the edges that are actually added
    t = np.ones(len(I))

    # If cones haven't turned into cylinders, metric is unchanged
    t[D <= 2*minlam*E0] = 0

    # Otherwise, if they meet before the M condition above, the metric is warped
    D[t == 1] = 2.0*(D[t == 1] - minlam[t == 1]*E0) # Multiply by 2 convention
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()

def plot_filtration_diagrams(D, result, time_taken, fig=None):
    if fig is None:
        plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.imshow(D)
    plt.title("Original Distance Matrix: %i Edges"%result['num_edges'])
    
    plt.subplot(122)
    plot_diagrams(result['dgms'], show=False)
    plt.title("Full Filtration: Elapsed Time %g Seconds"%time_taken)

    return fig

### Use the C++ version for 100x/2 orders of magnitude speed-up
from homology_clustering_py import trajectory_segment_distance

### Vlad's new magic
def trajectory_segment_distance_python(demos):
    n = demos.shape[1] - 1
    N = demos.shape[0] * n
    D = np.zeros([N, N])
    for i in range(demos.shape[0]):
        for j in range(i, demos.shape[0]):
            for k in range(n):
                for l in range(n):
                    if i==j and k==l:
                        continue
                    a1 = demos[i,k]
                    a2 = demos[i,k+1]
                    b1 = demos[j,l]
                    b2 = demos[j,l+1]
                    d=segment_distance(a1,a2,b1,b2)
                    D[i*n+k,j*n+l] = d
                    D[j*n+l,i*n+k] = d
    return D

from homology_clustering_py import segment_distance

def segment_distance_python(point1s, point1e, point2s, point2e):
    d1  = point1e - point1s
    d2  = point2e - point2s
    d12 = point2s - point1s

    D1  = np.dot(d1, d1)  # Squared norm of segment 1 (squared length)
    D2  = np.dot(d2, d2)  # Squared norm of segment 2 (squared length)

    S1  = np.dot(d1,d12)
    S2  = np.dot(d2,d12)
    R   = np.dot(d1,d2)

    den = D1*D2-R*R

    if D1 == 0.0 or D2 == 0.0:
        if D1 != 0.0:
            u = 0.0
            t = S1 / D1
            t = np.clip(t, 0.0, 1.0)
        elif D2 != 0.0:
            t = 0.0
            u = -S2 / D2

            u = np.clip(u, 0.0, 1.0)
        else:
            t = 0.0
            u = 0.0
    elif den == 0.0:
        t = 0.0
        u = -S2 / D2
        uf = np.clip(u, 0.0, 1.0)
        if uf != u:
            t = (uf * R + S1) / D1
            t = np.clip(t, 0.0, 1.0)
            u = uf
    else:
        t = (S1 * D2 - S2 * R) / den
        t = np.clip(t, 0.0, 1.0)
        u = (t * R - S2) / D2
        uf = np.clip(u, 0.0, 1.0)
        if uf != u:
            t = (uf * R + S1) / D1
            t = np.clip(t, 0.0, 1.0)
            u = uf

    return np.linalg.norm(d1*t-d2*u-d12)

# 'compute_pair'
def compute_homology_filtration(demos, data=None, truncate_small_distances=None, truncate_below_percentage=None, connect_start=True, connect_end=True, do_cocycles=False, n_perm=None, scale=1.0, debug=False):
    # Compute the sparse filtration
    tic = time()

    # First compute all pairwise distances and do furthest point sampling
    D = trajectory_segment_distance(demos)
    debug and print('Distance computation time: ' + str(time()-tic))
    ss = time()
    trajectory_mod(D, [demos.shape[0], demos.shape[1]-1], connect_start, connect_end)
    ee = time()
    debug and print('Trajectory distance modification time:', ee-ss)

    # Truncate small distances for speed-up
    if truncate_small_distances is not None:
        D[D < truncate_small_distances] = 0.
    
    # Truncate distances less than x% of the maximum distance
    if truncate_below_percentage is not None:
        D[D < truncate_below_percentage * D.max()] = 1e-14 # Small, not zero.

    # Dense
    ss = time()
    result_dense = ripser(D, distance_matrix=True, maxdim=1, do_cocycles=do_cocycles, n_perm=n_perm)
    ee = time()
    time_dense = ee-ss
    debug and print("[Dense] Elapsed Time: %.3g seconds, %i Edges added"%(time_dense, result_dense['num_edges']))

    result_dense['time_dense'] = time_dense

    return (result_dense, D, ) #result_sparse, DSparse)

def compute_filtration_from_preprocessed_distance_matrix(D, debug=False):
    ss = time()
    result = ripser(D, distance_matrix=True, maxdim=1)
    ee = time()
    timefull = ee-ss

    debug and print("Elapsed Time: %.3g seconds, %i Edges added"%(timefull, result['num_edges']))
    return result

def get_pairwise_trajectory_distance_matrix(demos, data, debug=False):
    results = []
    s = time()
    # common_start_point = np.zeros((2, 1, demos.shape[2]))
    # common_end_point = np.zeros((2, 1, demos.shape[2]))
    # for i in range(2):
    #     common_start_point[i,:,:] = -5. * np.ones((demos.shape[2],))
    #     common_end_point[i,:,:] = 5. * np.ones((demos.shape[2],))
    for i in range(demos.shape[0]-1):
        for j in range(i+1, demos.shape[0]):
            demos_tmp = demos[[i,j],:,:].copy()
            # Add common start and end points
            # demos_tmp = np.concatenate([common_start_point, demos_tmp, common_end_point], axis=1)
            data_tmp = np.vstack([d for d in demos])
            result_pair, _ = compute_homology_filtration(demos_tmp, data_tmp)
            results.append(result_pair)
    e = time()
    debug and print("Pairwise homology filtration took", e-s)
    
    s = time()
    n = demos.shape[0]
    D = np.zeros([n,n])
    k = 0
    for i in range(n-1):
        for j in range(i+1, n):
            p1 = results[k]['dgms'][1].copy()
            p1[:,1] -= p1[:,0] # lifetime
            # d is the sum lifetime of all H1 classes (as distance from the diagonal). 
            # If it's more than 0, then there are loops between the two trajectories.
            # If there are multiple loops, d will be larger. But that's fine.
            # The two trajectories are in the same homology class only if they have H1~0
            d = np.sum(p1[:,1])
            D[i,j] = d
            D[j,i] = d
            k += 1
    e = time()
    debug and print("Computation of trajectory-wise distance matrix took", e-s)
    return D

def get_cluster_labels_from_pairwise_trajectory_distance_matrix(num_classes, D, debug=False):
    s = time()
    # clustering = AgglomerativeClustering(n_clusters=num_classes, affinity='precomputed', linkage='complete').fit(D)
    clustering = AgglomerativeClustering(n_clusters=num_classes, affinity='precomputed', linkage='single').fit(D)
    labels = clustering.labels_
    e = time()
    debug and print("Agglomerative clustering took", e-s)
    return labels

from homology_clustering_py import compute_distance_matrix_from_trajectories as d_pair_cpp

def d_pair(demos_in, data=None):
    demos = demos_in.copy()
    D_tmp = trajectory_segment_distance(demos)
    trajectory_mod(D_tmp, [demos.shape[0], demos.shape[1]-1], True, True)
    return D_tmp

def get_pairwise_trajectory_distance_matrix_simple_python(demos_in, data_in=None, debug=False):
    demos = demos_in.copy()

    common_start_point = np.zeros((2, 1, demos.shape[2]))
    common_end_point = np.zeros((2, 1, demos.shape[2]))
    for i in range(2):
        common_start_point[i,:,:] = -5. * np.ones((demos.shape[2],))
        common_end_point[i,:,:] = 5. * np.ones((demos.shape[2],))

    s = time()
    n = demos.shape[0]
    D = np.zeros([n,n])
    for i in range(n-1):
        for j in range(i+1, n):
            demos_tmp = demos[[i,j],:,:]
            D_ = d_pair(demos_tmp)
            n2 = int(D_.shape[0]/2)
            d = np.max(np.min(D_[n2:,0:n2], axis=0))
            D[i,j] = d
            D[j,i] = d
    e = time()
    debug and print("Computation of trajectory-wise distance matrix (simple) took", e-s)
    return D

def get_num_classes_from_h1(h1, ratio_between_subsequent_persistence_pairs=0.8, min_threshold_for_first_hole=0.1, min_lifetime=0.1, debug=False):
    lifetime = np.sort(h1[:,1]-h1[:,0])[::-1]
    debug and print("Potential number of holes:", lifetime.shape[0])

    num_classes = 1 # we always have at least 1 class...

    # To determine the number of classes, we look at the half-time of persistence values
    if lifetime.shape[0] > 0 and lifetime[0] > min_threshold_for_first_hole:
        num_classes += 1
    np.set_printoptions(2)
    debug and print("#classes \t ratio \t prev. \t current \t 0.5*prev")
    for i in range(1, lifetime.shape[0]):
        ratio = lifetime[i] / lifetime[i-1]
        debug and print("{:8d}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}".format(num_classes, ratio, lifetime[i-1], lifetime[i], ratio_between_subsequent_persistence_pairs*lifetime[i-1]))
        
        if lifetime[i] < ratio_between_subsequent_persistence_pairs * lifetime[i-1]:
            #print("too small, breaking: ", lifetime[i], ratio_between_subsequent_persistence_pairs * lifetime[i-1])
            break
        if lifetime[i] < min_lifetime:
            break
        num_classes += 1
    if debug:
        fig = plt.figure()
        plt.plot(lifetime)
        #plt.yscale('log')
        plt.vlines(x=num_classes, ymin=0.0, ymax=h1.max(), color='red')
        plt.xlim(0,25)
    else:
        fig = None
    return num_classes, fig
