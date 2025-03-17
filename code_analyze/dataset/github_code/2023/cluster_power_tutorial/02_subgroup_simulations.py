#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import time

import numpy
import matplotlib
from matplotlib import pyplot
import scipy.stats

from sklearnex import patch_sklearn
patch_sklearn()

import skfuzzy
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score, adjusted_rand_score, \
    silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder
from stepmix.stepmix import StepMix

# Overwrite scikit-learn's forced warnings.
# NOTE: If you're using this code after downloading it from GitHub, first 
# check that none of the warnings are pertinent to what you're doing.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Record start time.
t0 = time.time()


# # # # #
# SETTINGS

# Overwrite temporary data?
overwrite_tmp = False

# Set to True for completely independent features (covariance matrix of 0s),
# or to False for a pseudo-randomly populated covariance matrix (with 
# correlations uniformly distributed between ~-0.6 and ~0.6).
feature_independence = True

# Set to True to perform multi-dimensional scaling before doing clustering.
multi_dimensional_scaling = False
# Set to True to perform dimensionality reduction using principal component 
# analysis before doing clustering.
principal_component_analysis = True
# Set to True to perform dimensionality reduction using feature hashing (only
# applies to categorical data.)
feature_hashing = False

# Subgroup analyses to run on each simulated sample.
subgroup_methods = ["kmeans", "ward", "cmeans", "lca", "lpa", "mix"]
# LCA are super slow, so it is recommended you run them separately.
# subgroup_methods = ["lca"]

# Range of lambdas to use in simulations.
w_range = [0.75, 1.5, 3.0, 6.0, 12.0]
w_label = { \
    0.75: "(wildly optimistic)", \
    1.5: r"($\approx$Szucs & Ioannidis, 2017)", \
    3.0: "(none to very large)", \
    6.0: "(none to large)", \
    12.0: "(none to medium)", \
    }

# Simulation scenarios to run.
simulations = { \
    # Wildly optimistic scenario.
    0.75: { \
        "n_observations": [30, 50, 75], \
        "n_features": [5, 9, 14], \
        "n_repeats": 50, \
        "feature_independence": feature_independence, \
        "multi_dimensional_scaling": multi_dimensional_scaling, \
        "principal_component_analysis": principal_component_analysis, \
        "principal_component_analysis": principal_component_analysis, \
        "feature_hashing": feature_hashing, \
        }, \

    # Psychological literature scenario.
    1.5: { \
        "n_observations": [30, 50, 100, 120, 150], \
        "n_features": [20, 36, 56], \
        "n_repeats": 50, \
        "feature_independence": feature_independence, \
        "multi_dimensional_scaling": multi_dimensional_scaling, \
        "principal_component_analysis": principal_component_analysis, \
        "feature_hashing": feature_hashing, \
        }, \

    # None to very large scenario.
    3.0: { \
        "n_observations": [30, 50, 100, 120, 150, 200], \
        "n_features": [81, 144, 225], \
        "n_repeats": 50, \
        "feature_independence": feature_independence, \
        "multi_dimensional_scaling": multi_dimensional_scaling, \
        "principal_component_analysis": principal_component_analysis, \
        "feature_hashing": feature_hashing, \
        }, \

    # None to large scenario.
    6.0: { \
        "n_observations": [30, 50, 100, 200, 500, 750, 1000], \
        "n_features": [324, 576, 900], \
        "n_repeats": 50, \
        "feature_independence": feature_independence, \
        "multi_dimensional_scaling": multi_dimensional_scaling, \
        "principal_component_analysis": principal_component_analysis, \
        "feature_hashing": feature_hashing, \
        }, \

    # None to medium scenario.
    12.0: { \
        "n_observations": [100, 200, 500, 1000, 1500, 2000, 5000], \
        "n_features": [1296, 2304, 3600], \
        "n_repeats": 50, \
        "feature_independence": feature_independence, \
        "multi_dimensional_scaling": multi_dimensional_scaling, \
        "principal_component_analysis": principal_component_analysis, \
        "feature_hashing": feature_hashing, \
        }, \

    }

# Files and folders.
this_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(this_dir, "output")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# Create an output directory for the current set of simulation parameters.
name = "simulation_indepent-{}_mds-{}_pca-{}_hash-{}_{}".format( \
    feature_independence, multi_dimensional_scaling, \
    principal_component_analysis, feature_hashing, "_".join(subgroup_methods))
out_dir = os.path.join(out_dir, name)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


# # # # #
# HELPER FUNCTIONS

def find_nearest_postive_definite_matrix(A):

    """Find the nearest positive-definite matrix to input.

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2]. This solution is from Ahmed Fasih, provided on StackOverflow:
    https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = numpy.linalg.svd(B)

    H = numpy.dot(V.T, numpy.dot(numpy.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = numpy.spacing(numpy.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `numpy.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually 
    # on the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = numpy.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = numpy.min(numpy.real(numpy.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = numpy.linalg.cholesky(B)
        return True
    except numpy.linalg.LinAlgError:
        return False


def silhouette_coefficient(X, y, u=None, alpha=1.0):
    
    """Computes the silhouette coefficient, or the fuzzy silhouette if a u 
    matrix is passed. Implementation from Dalmaijer, Nord, & Astle (2020). 
    Statistical power for cluster analysis. arXiv:2003.00381, 
    https://arxiv.org/abs/2003.00381. Code implementation from accompanying
    GitHub page: https://github.com/esdalmaijer/cluster_power
    """
    
    # Compute silhoutte coefficient for all samples.
    s = silhouette_samples(X, y)
    
    # If no u matrix was passed, compute the regular silhouette score.
    if u is None:
        s_m = numpy.nanmean(s)
    # If a u matrix was passed, compute the fuzzy silhouette score.
    else:
        # Find largest and second-largest elements for all samples.
        u_sorted = numpy.sort(u, axis=1)
        u_p = u_sorted[:,-1]
        u_q = u_sorted[:,-2]
        # Compute the fuzzy silhouette score.
        s_m = numpy.nansum(((u_p-u_q)**alpha) * s) \
            / numpy.nansum((u_p-u_q)**alpha)

    return s_m, s


def run_subgroup_analysis(subgroup_methods, w, n, p, \
    feature_independence=False, multi_dimensional_scaling=False, \
    principal_component_analysis=False, feature_hashing=False, verbose=False):
    
    if multi_dimensional_scaling and principal_component_analysis:
        raise Exception("Both dimensionality reduction options " \
            + "(MDS and PCA) were set to True. Please set only one to True.")

    # Create the exponential distribution instance with lambda=w (this needs
    # to be entered as scale = 1/lambda).
    exp_dist = scipy.stats.expon(scale=1.0/w)
    
    # Sample single-feature effect sizes from the distribution.
    d = exp_dist.rvs(size=p)
    # Randomly assign signs.
    sign = numpy.random.choice([1.0, -1.0], size=d.shape, \
        p=[0.5,0.5])
    d *= sign
    # Compute intended centroid separation.
    delta = numpy.sqrt(numpy.sum((d)**2))

    # Create the means for group 1 (zeros minus half the effect
    # size, so that the overall mean is zero).
    m1 = numpy.zeros(p, dtype=numpy.float64) - d/2
    # Create the means for group 2 (group 1 plus the effect size;
    # this works because all pooled SDs will be 1).
    m2 = m1 + d
    
    # Create a covariance matrix for independent features.
    if feature_independence:
        # Create a 0 covariance matrix.
        cov = numpy.zeros((p,p))
        # Set the diagonal to 1.
        diagonal_indices = numpy.diag_indices(p, ndim=2)
        cov[diagonal_indices] = 1.0

    # Create a covariance matrix for related features.
    else:

        # Generate a symmetrical covariance matrix.
        cov = -1 + numpy.random.rand(p,p) * 2
        # Equalise the triangles.
        cov[numpy.triu_indices(p, k=1)] = \
            cov[numpy.tril_indices(p, k=-1)]
        # Set the diagonal to 1.
        diagonal_indices = numpy.diag_indices(p, ndim=2)
        cov[diagonal_indices] = 1.0
        
        # Find the nearest postive-definite matrix.
        cov = find_nearest_postive_definite_matrix(cov)
        
        # If the lowest eigenvalue is negative, the covariance matrix 
        # isn't symmetric positive semi-definite. This needs to be 
        # corrected.
        min_eig = numpy.min(numpy.real(numpy.linalg.eigvals(cov)))
        if min_eig < 0:
            # Correct the covariance matrix.
            cov -= min_eig * numpy.eye(*cov.shape)
            # Transform back to a maximum of 1.0
            cov /= numpy.max(cov)
            if verbose:
                print("BAD covariance matrix! Corrected to be positive " \
                    + "definite, with new off-diagonal covariance " \
                    + " min={} and max={}".format( \
                    numpy.round(numpy.tril(cov, -1).min(),2), \
                    numpy.round(numpy.tril(cov, -1).max(), 2)))

    # Create all observations.
    max_attempts = 100
    for i_try in range(max_attempts):
        try:
            X = numpy.random.multivariate_normal(numpy.zeros(p), cov, \
                size=2*n)
        except numpy.linalg.LinAlgError:
            X = None
        if X is not None:
            break
    if X is None:
        return None

    # Randomly assign observations to groups.
    y = numpy.hstack([numpy.zeros(n), numpy.ones(n)])
    numpy.random.shuffle(y)

    # Move the mean of group 1.
    X[y==0] += m1
    # Move the mean of group 2.
    X[y==1] += m2
    
    # If latent class analysis is among the subgroup methods, create a 
    # categorical version with a maximum of 5 options.
    if "lca" in subgroup_methods:
        X_cat = X + numpy.abs(numpy.min(X))
        X_cat = 4 * (X_cat / numpy.max(X_cat))
        X_cat = numpy.round(X_cat, 0).astype(numpy.int64)
    
    # Compute the correlation matrix.
    r_matrix = numpy.cov(X.T) / (numpy.std(X, axis=0) * \
        numpy.std(X, axis=0).reshape(X.shape[1],1))
    
    # Compute the averages for each group within each feature.
    m1_ = numpy.mean(X[y==0], axis=0)
    m2_ = numpy.mean(X[y==1], axis=0)
    d_ = m2_ - m1_
    
    # Compute achieved centroid separation.
    delta_ = numpy.sqrt(numpy.sum((d_)**2))
    
    # Optionally perform multi-dimensional scaling.
    if multi_dimensional_scaling:
        if verbose:
            print("Running MDS...")
        # Create a new MDS instance.
        mds = MDS(n_components=2)
        # Fit and transform X.
        X = mds.fit_transform(X)
        # Update achieved separation.
        delta_ = numpy.sqrt(numpy.sum((numpy.mean(X[y==0], axis=0) - \
            numpy.mean(X[y==1], axis=0))**2))

    # Optionally perform principal component analysis.
    if principal_component_analysis:
        if verbose:
            print("Running PCA...")
        # Create a new PCA instance.
        pca = PCA(n_components=2)
        # Fit and transform X.
        X = pca.fit_transform(X)
        # Update achieved separation.
        delta_ = numpy.sqrt(numpy.sum((numpy.mean(X[y==0], axis=0) - \
            numpy.mean(X[y==1], axis=0))**2))
    
    # Optionally reduce categorical data using feature hashing.
    if feature_hashing and ("lca" in subgroup_methods):
        # At two features, this results in so many collisions, and frankly
        # just does not work for our purposes.
        hasher = FeatureHasher(n_features=2, input_type="string")
        f = hasher.fit_transform(X_cat.astype(str))
        X_cat = f.toarray()

    if verbose:
        print("delta_intended={}, delta_achieved={}".format( \
            numpy.round(delta, 2), numpy.round(delta_, 2)))
    
    # SUBGROUP ANALYSES
    sil = {}
    bf10 = {}
    acc = {}
    rand = {}
    for mi, method in enumerate(subgroup_methods):
        
        if verbose:
            print(("Running method {} ({}/{}) on predictors with " \
                + "shape={}").format(method, mi+1, len(subgroup_methods), \
                X.shape))

        # k-means
        if method == "kmeans":
            model = KMeans(n_clusters=2, algorithm="lloyd")
            y_pred = model.fit_predict(X)
            u = None
            bf10[method] = numpy.NaN
        
        # Agglomerative clustering with Ward linkage
        elif method == "ward":
            # Create a connectivity matrix.
            connectivity = kneighbors_graph(X, n_neighbors=10, \
                include_self=False)
            # Make the connectivity matrix symmetrical.
            connectivity = 0.5 * (connectivity + connectivity.T)
            # Run the algorithm.
            model = AgglomerativeClustering(n_clusters=2, \
                linkage='ward', connectivity=connectivity)
            y_pred = model.fit_predict(X)
            u = None
            bf10[method] = numpy.NaN

        # c-means
        elif method == "cmeans":
            _, u, _, _, _, _, _ = skfuzzy.cluster.cmeans( \
                numpy.transpose(X), 2, 2, error=0.005, \
                maxiter=1000, init=None)
            y_pred = numpy.argmax(u, axis=0)
            u = numpy.transpose(u)
            bf10[method] = numpy.NaN

        # Latent class analysis.
        elif method == "lca":
            # Run a null model.
            null_model = StepMix(n_components=1, \
                measurement="categorical", verbose=0, \
                progress_bar=0)
            null_model.fit(X_cat)
            # Run the LCA.
            model = StepMix(n_components=2, \
                measurement="categorical", verbose=0, \
                progress_bar=0)
            model.fit(X_cat)
            y_pred = model.predict(X_cat)
            u = model.predict_proba(X_cat)
            # Compute the Bayes Factor.
            bic0 = null_model.bic(X_cat)
            bic1 = model.bic(X_cat)
            bf10[method] = numpy.exp(0.5 * (bic0 - bic1))
        
        # Latent profile analysis.
        # This is just a Gaussian mixture model in which each
        # subgroup has their own variance, and no covariance
        elif method == "lpa":
            model = GaussianMixture(n_components=2, \
                covariance_type="spherical")
            model.fit(X)
            y_pred = model.predict(X)
            u = model.predict_proba(X)
            # Compute a Bayes Factor for the 2-cluster model compared to
            # a single-group model.
            null_model = GaussianMixture(n_components=1, \
                covariance_type="spherical")
            null_model.fit(X)
            bic0 = null_model.bic(X)
            bic1 = model.bic(X)
            bf10[method] = numpy.exp(0.5 * (bic0 - bic1))
        
        # Gaussian mixture modelling.
        elif method == "mix":
            model = GaussianMixture(n_components=2, \
                covariance_type="full")
            model.fit(X)
            y_pred = model.predict(X)
            u = model.predict_proba(X)
            # Compute a Bayes Factor for the 2-cluster model compared to
            # a single-group model.
            null_model = GaussianMixture(n_components=1, \
                covariance_type="full")
            null_model.fit(X)
            bic0 = null_model.bic(X)
            bic1 = model.bic(X)
            bf10[method] = numpy.exp(0.5 * (bic0 - bic1))

        # Any other methods aren't supported.
        else:
            raise Exception(("Method '{}' not supported; choose " \
                + "from {}.").format(method, subgroup_methods))
        
        # Check that more than one group have been detected.
        if numpy.unique(y_pred).shape[0] < 2:
            sil[method] = 0.0
        else:
            # Compute and save the silhouette score in the 
            # memory-mapped array.
            if method == "lca":
                sil[method], _ = silhouette_coefficient(X_cat, y_pred, u)
            else:
                sil[method], _ = silhouette_coefficient(X, y_pred, u)
            # Compute the adjusted Rand index.the
            rand[method] = adjusted_rand_score(y, y_pred)
            # Compute the classification accuracy. Note that this does not
            # account for reverse labelling (which is arbitrary in clustering),
            # so we take the maximum of either the produced or the inversed
            # labels. (This trick works because we have only two subgroups.)
            y_pred_reversed_labels = 1 - y_pred
            acc[method] = numpy.max([accuracy_score(y, y_pred), \
                accuracy_score(y, y_pred_reversed_labels)])

        if verbose:
            print("Finished method {} ({}/{}) with silhouette={}".format( \
                method, mi+1, len(subgroup_methods), sil[method]))
    
    return sil, bf10, acc, rand, X, y, y_pred, d, delta, d_, delta_, r_matrix


# # # # #
# SIMULATIONS

# Run through all contexts.
for wi, w in enumerate(w_range):
    
    print("Running range {} ({}/{})".format(w_label[w], wi+1, len(w_range)))
    
    if w not in simulations.keys():
        print("No simulation scenarios found for w={}".format(w))
        continue

    # Get the simulation specifics.
    n_features = simulations[w]["n_features"]
    n_observations = simulations[w]["n_observations"]
    n_repeats = simulations[w]["n_repeats"]
    feature_independence = simulations[w]["feature_independence"]
    multi_dimensional_scaling = simulations[w]["multi_dimensional_scaling"]
    principal_component_analysis = \
        simulations[w]["principal_component_analysis"]
    feature_hashing = simulations[w]["feature_hashing"]
    
    # Create a variable to store silhouette coefficients and deltas in.
    shape = (len(subgroup_methods), len(n_observations), len(n_features), \
        n_repeats)
    fpath = os.path.join(out_dir, "tmp_silhouette_scores_w{}.dat".format( \
        str(w).replace(".", "-")))
    fpath_bf = os.path.join(out_dir, "tmp_BF10_w{}.dat".format( \
        str(w).replace(".", "-")))
    fpath_acc = os.path.join(out_dir, "tmp_accuracy_w{}.dat".format( \
        str(w).replace(".", "-")))
    fpath_rand = os.path.join(out_dir, "tmp_adj_rand_index_w{}.dat".format( \
        str(w).replace(".", "-")))
    fpath_delta = os.path.join(out_dir, "tmp_delta_w{}.dat".format( \
        str(w).replace(".", "-")))

    # If the file already exists, load its values.
    if os.path.isfile(fpath) and not overwrite_tmp:
        print("File already exist, so no new simulations will be run.")
        continue

    # If the file does not exist (or should be overwritten), create a new one.
    else:
        sil = numpy.memmap(fpath, mode="w+", dtype=numpy.float64, shape=shape)
        sil[:] = numpy.NaN
        bf10 = numpy.memmap(fpath_bf, mode="w+", dtype=numpy.float64, \
            shape=shape)
        bf10[:] = numpy.NaN
        acc = numpy.memmap(fpath_acc, mode="w+", dtype=numpy.float64, \
            shape=shape)
        acc[:] = numpy.NaN
        rand = numpy.memmap(fpath_rand, mode="w+", dtype=numpy.float64, \
            shape=shape)
        rand[:] = numpy.NaN
        delt = numpy.memmap(fpath_delta, mode="w+", dtype=numpy.float64, \
            shape=shape[1:])
        delt[:] = numpy.NaN
    
    # Run through all sample sizes.
    for i, n in enumerate(n_observations):
        
        print("\tRunning {} observations ({}/{})".format(n, i+1, \
            len(n_observations)))

        # Run through all numbers of features.
        for j, p in enumerate(n_features):
            
            print("\t\tRunning {} features ({}/{})".format(p, j+1, \
                len(n_features)))
    
            # Flush to ensure previous output is saved to disk.
            sil.flush()
            bf10.flush()
            delt.flush()
    
            # Run through all simulation runs.
            for k in range(n_repeats):

                print("\t\t\tRepeat {}/{}".format(k+1, n_repeats))
                
                result = run_subgroup_analysis(subgroup_methods, w, n, p, \
                    feature_independence=feature_independence, \
                    multi_dimensional_scaling=multi_dimensional_scaling, \
                    principal_component_analysis=principal_component_analysis, \
                    feature_hashing=feature_hashing, verbose=False)
                
                if result is not None:
                    sil_, bf10_, acc_, rand_, X, y, y_pred, d, delta, d_, \
                        delta_, r_matrix = result
                    delt[i,j,k] = delta_
                    for method in sil_.keys():
                        mi = subgroup_methods.index(method)
                        sil[mi,i,j,k] = sil_[method]
                        bf10[mi,i,j,k] = bf10_[method]
                        acc[mi,i,j,k] = acc_[method]
                        rand[mi,i,j,k] = rand_[method]

t1 = time.time()
print("\nCompleted in {} seconds".format(t1-t0))


# # # # #
# POWER COMPUTATION

# Run through all contexts.
for wi, w in enumerate(w_range):
    
    print("Power computation for range {} ({}/{})".format(w_label[w], \
        wi+1, len(w_range)))
    
    if w not in simulations.keys():
        print("No simulation scenarios found for w={}".format(w))
        continue

    # Get the simulation specifics.
    n_features = simulations[w]["n_features"]
    n_observations = simulations[w]["n_observations"]
    n_repeats = simulations[w]["n_repeats"]
    feature_independence = simulations[w]["feature_independence"]
    multi_dimensional_scaling = simulations[w]["multi_dimensional_scaling"]
    principal_component_analysis = \
        simulations[w]["principal_component_analysis"]
    feature_hashing = simulations[w]["feature_hashing"]
    
    # Load the silhouette coefficients.
    shape = (len(subgroup_methods), len(n_observations), len(n_features), \
        n_repeats)
    fpath = os.path.join(out_dir, "tmp_silhouette_scores_w{}.dat".format( \
        str(w).replace(".", "-")))
    fpath_bf = os.path.join(out_dir, "tmp_BF10_w{}.dat".format( \
        str(w).replace(".", "-")))
    fpath_acc = os.path.join(out_dir, "tmp_accuracy_w{}.dat".format( \
        str(w).replace(".", "-")))
    fpath_rand = os.path.join(out_dir, "tmp_adj_rand_index_w{}.dat".format( \
        str(w).replace(".", "-")))
    fpath_delta = os.path.join(out_dir, "tmp_delta_w{}.dat".format( \
        str(w).replace(".", "-")))
    sil = numpy.memmap(fpath, mode="r", dtype=numpy.float64, shape=shape)
    bf10 = numpy.memmap(fpath_bf, mode="r", dtype=numpy.float64, shape=shape)
    acc = numpy.memmap(fpath_acc, mode="r", dtype=numpy.float64, shape=shape)
    rand = numpy.memmap(fpath_rand, mode="r", dtype=numpy.float64, shape=shape)
    delt = numpy.memmap(fpath_delta, mode="r", dtype=numpy.float64, \
        shape=shape[1:])

    # Compute power for each subgroup analysis.
    power = {}
    for mi, method in enumerate(subgroup_methods):
        power[method] = numpy.nansum(sil[mi,:,:] >= 0.5, axis=2) / n_repeats
    
    # Compute the average log(BF10).
    bayes = {}
    for mi, method in enumerate(subgroup_methods):
        bayes[method] = numpy.nanmean(numpy.log(bf10[mi,:,:]), axis=2)

    # Compute average accuracy for each subgroup analysis.
    accuracy = {}
    for mi, method in enumerate(subgroup_methods):
        accuracy[method] = numpy.nanmean(acc[mi,:,:], axis=2)
    
    # Compute average adjusted Rand index for each subgroup analysis.
    adj_rand = {}
    for mi, method in enumerate(subgroup_methods):
        adj_rand[method] = numpy.nanmean(rand[mi,:,:], axis=2)

    # Compute the average centroid separation.
    m_delt = numpy.nanmean(delt, axis=2)

    # Write power to file.
    for mi, method in enumerate(subgroup_methods):
        fpath = os.path.join(out_dir, "power_w{}_{}.tsv".format( \
            str(w).replace(".", "-"), method))
        with open(fpath, "w") as f:
            header = [""] + ["p={}".format(p) for p in n_features]
            f.write("\t".join(header))
            for i, n in enumerate(n_observations):
                line = ["n={}".format(n)] + list(power[method][i,:])
                f.write("\n" + "\t".join(map(str, line)))
    
    # Write log(BF10) to file.
    for mi, method in enumerate(subgroup_methods):
        fpath = os.path.join(out_dir, "log-bayes_w{}_{}.tsv".format( \
            str(w).replace(".", "-"), method))
        with open(fpath, "w") as f:
            header = [""] + ["p={}".format(p) for p in n_features]
            f.write("\t".join(header))
            for i, n in enumerate(n_observations):
                line = ["n={}".format(n)] + list(bayes[method][i,:])
                f.write("\n" + "\t".join(map(str, line)))

    # Write accuracy to file.
    for mi, method in enumerate(subgroup_methods):
        fpath = os.path.join(out_dir, "accuracy_w{}_{}.tsv".format( \
            str(w).replace(".", "-"), method))
        with open(fpath, "w") as f:
            header = [""] + ["p={}".format(p) for p in n_features]
            f.write("\t".join(header))
            for i, n in enumerate(n_observations):
                line = ["n={}".format(n)] + list(accuracy[method][i,:])
                f.write("\n" + "\t".join(map(str, line)))

    # Write adjusted Rand to file.
    for mi, method in enumerate(subgroup_methods):
        fpath = os.path.join(out_dir, "adj-rand_w{}_{}.tsv".format( \
            str(w).replace(".", "-"), method))
        with open(fpath, "w") as f:
            header = [""] + ["p={}".format(p) for p in n_features]
            f.write("\t".join(header))
            for i, n in enumerate(n_observations):
                line = ["n={}".format(n)] + list(adj_rand[method][i,:])
                f.write("\n" + "\t".join(map(str, line)))

    # Write delta to file.
    fpath = os.path.join(out_dir, "delta_w{}.tsv".format( \
        str(w).replace(".", "-")))
    with open(fpath, "w") as f:
        header = [""] + ["p={}".format(p) for p in n_features]
        f.write("\t".join(header))
        for i, n in enumerate(n_observations):
            line = ["n={}".format(n)] + list(m_delt[i,:])
            f.write("\n" + "\t".join(map(str, line)))
