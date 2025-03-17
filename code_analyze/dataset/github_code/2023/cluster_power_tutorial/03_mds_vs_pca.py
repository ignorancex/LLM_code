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
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from stepmix.stepmix import StepMix

# Overwrite scikit-learn's forced warnings.
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

# Sample size (per group) to use in simulations.
n_observations = 100

# Range for the number of variables to generate.
p = numpy.logspace(0, 3.5, 19, base=10).astype(numpy.int32)
# Remove repeats.
p = numpy.unique(p)
# Remove values below 3, as these don't need reduction.
p = p[p>2]

# Number of repeats per combination of sample size and number of variables.
n_repeats = 10

# Set to True for completely independent features (covariance matrix of 0s),
# or to False for a pseudo-randomly populated covariance matrix (with 
# correlations uniformly distributed between ~-0.6 and ~0.6).
feature_independence = True

# Dimensionality reduction algorithms to run on each simulated sample.
dim_reduction_methods = ["none", "mds", "pca"]

# Range of lambdas to use in simulations.
w_range = [0.75, 1.5, 3.0, 6.0, 12.0]
w_label = { \
    0.75: "(wildly optimistic)", \
    1.5: r"($\approx$Szucs & Ioannidis, 2017)", \
    3.0: "(none to very large)", \
    6.0: "(none to large)", \
    12.0: "(none to medium)", \
    }

# Files and folders.
this_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(this_dir, "output")
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


def run_dimensionality_reductions(dim_reduction_methods, w, n, p, \
    feature_independence=False, verbose=False):
    
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
    
    # Compute the correlation matrix.
    r_matrix = numpy.cov(X.T) / (numpy.std(X, axis=0) * \
        numpy.std(X, axis=0).reshape(X.shape[1],1))
    
    # Compute the averages for each group within each feature.
    m1_ = numpy.mean(X[y==0], axis=0)
    m2_ = numpy.mean(X[y==1], axis=0)
    d_ = m2_ - m1_
    
    # Compute achieved centroid separation.
    delta_ = {}
    
    # Run through dimensionality reduction options.
    for di, dimensionality_reduction in enumerate(dim_reduction_methods):
        
        # Recode None as a string.
        if dimensionality_reduction is None:
            dimensionality_reduction = "none"

        # Leave all variables as they are.
        if dimensionality_reduction == "none":
            X_ = numpy.copy(X)
    
        # Optionally perform multi-dimensional scaling.
        elif dimensionality_reduction == "mds":
            if verbose:
                print("Running MDS...")
            # Create a new MDS instance.
            mds = MDS(n_components=2, n_jobs=-1)
            # Fit and transform X.
            X_ = mds.fit_transform(X)
    
        # Optionally perform principal component analysis.
        elif dimensionality_reduction == "pca":
            if verbose:
                print("Running PCA...")
            # Create a new PCA instance.
            pca = PCA(n_components=2)
            # Fit and transform X.
            X_ = pca.fit_transform(X)
        
        # Throw an error on unrecognised options.
        else:
            raise NotImplementedError("Unrecognised dimensionality reduction " \
                + "'{}'".format(dimensionality_reduction))

        # Update achieved separation.
        delta_[dimensionality_reduction] = numpy.sqrt(numpy.sum(( \
            numpy.mean(X_[y==0], axis=0) - \
            numpy.mean(X_[y==1], axis=0))**2))

        if verbose:
            print("Finished method {} ({}/{}) with delta={}".format( \
                dimensionality_reduction, di+1, len(dim_reduction_methods), \
                delta_[dimensionality_reduction]))
    
    return X, y, d, delta, d_, delta_, r_matrix


# # # # #
# SIMULATION RUNS

# Run through all effect size contexts (each governed by a lambda value).
delta_per_w = {}
for wi, w in enumerate(w_range):

    # Create the path to the temporary file.
    fpath = os.path.join(out_dir, "delta_mds_vs_pca_w{}.dat".format( \
        str(w).replace(".", "-")))
    # Construct the shape of the array.
    shape = (len(dim_reduction_methods), 2, p.shape[0], n_repeats)
    
    # Load existing data.
    if os.path.isfile(fpath) and not overwrite_tmp:
        # Create a memory-mapped array to save data in.
        deltas = numpy.memmap(fpath, mode="r", shape=shape, dtype=numpy.float64)
    
    # Create new data.
    else:
        # Create a memory-mapped array to save data in.
        deltas = numpy.memmap(fpath, mode="w+", shape=shape, dtype=numpy.float64)
        deltas[:] = numpy.NaN
    
        # Run simulations.
        for feature_independence in [False, True]:
            fi = int(feature_independence)
            for i, n_features in enumerate(p):
                for j in range(n_repeats):
                    print(("Running simulation: w={} ({}/{}), " \
                        + "independent={}, p={} ({}/{}), " \
                        + "repeat={}/{}").format(w, wi+1, len(w_range), \
                        feature_independence, n_features, i+1, p.shape[0], \
                        j+1, n_repeats))
                    result = \
                        run_dimensionality_reductions(dim_reduction_methods, \
                        w, n_observations, n_features, \
                        feature_independence=feature_independence, \
                        verbose=False)
                    if result is None:
                        continue
                    X, y, d, delta, d_, delta_, r_matrix = result
                    for dimensionality_reduction in delta_.keys():
                        di = dim_reduction_methods.index( \
                            dimensionality_reduction)
                        deltas[di,fi,i,j] = delta_[dimensionality_reduction]
    
    # Store the loaded or computed delta values.
    delta_per_w[w] = numpy.copy(deltas)
    

# # # # #
# FIGURE

# Set plot colours and line styles.
line_style = {"none": "-", "mds": "-", "pca":"-"}

# Create a colour map.
cmap = matplotlib.cm.get_cmap("viridis_r")
w_min = numpy.log(numpy.min(w_range))
w_max = numpy.log(numpy.max(w_range))
vmin = w_min - (w_max-w_min) * 0.1
vmax = w_max + (w_max-w_min) * 0.1
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# Create a new figure.
fig, axes = pyplot.subplots(nrows=2, ncols=3, figsize=(16.0,10.0), dpi=300.0, \
    sharex=True, sharey=True)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.1, top=0.96, \
    wspace=0.1, hspace=0.1)

# Plot the lines.
for feature_independence in [False, True]:
    fi = int(feature_independence)
    for di, dimensionality_reduction in enumerate(dim_reduction_methods):
        
        # Choose the axes to draw in. Top row is for correlated features,
        # bottom for uncorrelated. Columns are for no, MDS, and PCA dim
        # reduction, respectively.
        row = fi
        col = di
        ax = axes[row, col]

        # Draw the minimally required cluster separation.
        delta_min = 4.0
        ax.plot([1, p[-1]], numpy.ones(2) * delta_min, ":", alpha=0.5, \
            lw=2, color="#000000")
        if (row == axes.shape[0] - 1) and (col == axes.shape[1] - 1):
            ax.annotate( \
                "Minimally required cluster separation\n(Dalmaijer et al., 2022)", \
                (1*1.1, delta_min*1.1), fontsize=12, alpha=0.5, \
                color="#000000")
        
        # Loop through all contexts.
        for wi, w in enumerate(w_range):

            # Construct a label for the current line.
            if (row == 0) and (col == 0):
                lbl = r"$\lambda=" + str(round(w, 2)).ljust(3, "0") + r"$"
                # if w in w_label.keys():
                #     lbl += " " + w_label[w]
            else:
                lbl = None

            # Get the current colour.
            colour = cmap(norm(numpy.log(w)))

            # Compute statistics.
            m = numpy.nanmean(delta_per_w[w][di,fi,:,:], axis=1)
            sd = numpy.nanstd(delta_per_w[w][di,fi,:,:], axis=1)
            sem = sd / numpy.sqrt(n_repeats)
            ci95 = 1.96 * sem
            
            # Plot the line.
            ax.plot(p, m, "-", ls=line_style[dimensionality_reduction], \
                lw=3, color=colour, alpha=0.5, label=lbl)
            ax.fill_between(p, m-sd, m+sd, color=colour, alpha=0.2)

        # Finish the plot.
        if dimensionality_reduction in ["mds", "pca"]:
            title = dimensionality_reduction.upper()
        elif dimensionality_reduction == "none":
            title = "No dimensionality reduction"
        else:
            title = dimensionality_reduction.capitalize()
        if row == 0:
            ax.set_title(title, fontsize=18)
        if feature_independence:
            y_title = "Independent features"
        else:
            y_title = "Correlated features"
        if row == axes.shape[0] - 1:
            ax.set_xlabel("Number of included features", fontsize=16)
        ax.set_xscale("log")
        ax.set_xlim(numpy.min(p), numpy.max(p))
        xticks = [1, 10, 25, 50, 100, 250, 500, 1000, 2500]
        ax.set_xticks(xticks)
        ax.set_xticklabels(list(map(str, xticks)), rotation=45, fontsize=12)
        if col == 0:
            ax.set_ylabel("{}\n".format(y_title) \
                + r"Centroid separation, $\Delta$", fontsize=16)
        ax.set_yscale("log")
        ylim = (0, 100)
        ax.set_ylim(ylim)
        yticks = [1, 10, 25, 50, 100]
        ax.set_yticks(yticks)
        ax.set_yticklabels(list(map(str, yticks)), fontsize=12)

# Finish the plot.
axes[0,0].legend(loc="lower right", fontsize=12)
fig.savefig(os.path.join(out_dir, "fig-03_pca_vs_mds.png"))
pyplot.close(fig)
