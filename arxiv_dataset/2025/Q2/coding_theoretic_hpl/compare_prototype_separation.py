import numpy as np
from scipy.special import comb as nCk
from scipy.stats import binom
import matplotlib.pyplot as plt
import itertools

import torch
from torch import optim
from torch import nn 
import torch.nn.functional as F
from models import ReedMullerCode, KasarlaCode, MettesCode, LogSumExpCode, BCHCode, RandomCode

def get_max_inner_prod(codebook):
    if codebook is None:
        return np.nan
    else: # Assume codebook is n_classes x latent_dim
        # Calculate all inner products
        product = torch.matmul(codebook, codebook.t())
        # Diagonal is always 1, so make sure this is never selected
        product -= 2. * torch.eye(product.shape[0])
        return torch.max(product)

def solve_gilbert_varshamov_bound(dim, num_classes):
    """Solves Gilbert-Varshamov by noting a similarity with the bound and the 
    CDF of a binomial distribution."""

    k = np.ceil(np.log2(num_classes))
    if np.equal(dim, k): #special case: all codewords are used
        return 1
    elif dim-k < 0: #too short blocklength
        return np.nan
    else:
        return 2+binom.ppf(2**(1-k), dim-1, 1/2)-1 #ppf rounds up, we are interested in rounding down, hence -1

def main():
    """Compares and plots different strategies for generating prototypes."""
    # Setup
    num_classes = 10
    start_dim = np.maximum(4, num_classes/10)
    dims = 2**np.arange(start=np.floor(np.log2(start_dim)), stop=np.ceil(np.log2(4*num_classes))+1, dtype=int) #K/10 to 4*K ish

    # Main loop: get max inner products and bounds
    kasarla_maxdist = get_max_inner_prod(KasarlaCode(num_classes).codebook)
    rm_maxdist = np.zeros_like(dims, dtype=float)
    mettes_maxdist = np.zeros_like(dims, dtype=float)
    lse_maxdist = np.zeros_like(dims, dtype=float)
    bch_maxdist = np.zeros_like(dims, dtype=float)
    random_maxdist = np.zeros_like(dims, dtype=float)
    achievable_bound = np.zeros_like(dims, dtype=float)
    converse_bound = -1/(num_classes-1)*np.ones_like(dims, dtype=float)
    for i in range(len(dims)):
        dim = dims[i]
        print(f"Starting dimension {dim}...")
        print(f"\tRM")
        rm_maxdist[i] = get_max_inner_prod(ReedMullerCode(dim, num_classes).codebook)
        print(f"\tMettes")
        mettes_maxdist[i] = get_max_inner_prod(MettesCode(dim, num_classes).codebook)
        print(f"\tLSE")
        lse_maxdist[i] = get_max_inner_prod(LogSumExpCode(dim, num_classes).codebook)
        print(f"\tBCH")
        bch_maxdist[i] = get_max_inner_prod(BCHCode(dim-1, num_classes).codebook)
        print(f"\tRandom")
        random_maxdist[i] = get_max_inner_prod(RandomCode(dim, num_classes).codebook)
        achievable_bound[i] = 1-2*solve_gilbert_varshamov_bound(dim, num_classes)/dim
        print("done")

    # Create plot
    # Set plot specifications
    plt.rcParams.update(
        {"xtick.direction" : "in",
        "xtick.major.size" : 3,
        "xtick.major.width" : 0.5,
        "xtick.minor.size" : 1.5,
        "xtick.minor.width" : 0.5,
        "xtick.minor.visible" : True,
        "xtick.top" : True,
        "xtick.labelsize" : 9,
        "ytick.direction" : "in",
        "ytick.major.size" : 3,
        "ytick.major.width" : 0.5,
        "ytick.minor.size" : 1.5,
        "ytick.minor.width" : 0.5,
        "ytick.minor.visible" : True,
        "ytick.right" : True,
        "ytick.labelsize" : 9,
        "font.family": "serif",
        "font.serif" : "Times",
        "font.size" : 9,
        "legend.fontsize" : 7,
        "text.usetex" : True,
        "text.latex.preamble" : '\\usepackage{amsmath}',
        "figure.figsize" : [3.5, 2.625]
        }
    )
    # Plot with "nan guards"
    plt.figure()
    plt.plot(dims[np.logical_not(np.isnan(bch_maxdist))]-1, bch_maxdist[np.logical_not(np.isnan(bch_maxdist))], color='tab:blue', marker='D', markersize=3, linewidth=0.5, label="BCH Codes")
    plt.plot(dims, lse_maxdist, marker='s', color='tab:orange', linewidth=0.5, markersize=3, label=r"Log-sum-exp $(\mathrm{P}_{\mathrm{LSE}})$")
    plt.plot(dims, mettes_maxdist, marker='1', color='tab:cyan', linewidth=0.5, markersize=6, label=r"Mettes et al. (2019)")
    plt.plot(dims[np.logical_not(np.isnan(achievable_bound))], achievable_bound[np.logical_not(np.isnan(achievable_bound))], '--', color='k', linewidth=0.5, markersize=3)
    plt.plot(dims[np.logical_not(np.isnan(rm_maxdist))], rm_maxdist[np.logical_not(np.isnan(rm_maxdist))], marker='.', color='tab:red', linewidth=0.5, label="RM Codes")
    # plt.plot(dims, random_maxdist, marker='x', color='tab:gray', markersize=3, linewidth=0.5, label="Random Prototypes")
    plt.scatter(np.array([num_classes-1]), np.array(kasarla_maxdist), marker='*', c='tab:olive', label="Kasarla et al. (2022)")
    plt.plot(dims, converse_bound, '--', color='k', linewidth=0.5, label="Bounds")
    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\max\limits_{i \ne j} \ \langle c_i, c_j \rangle$")
    plt.grid(visible=True, alpha=0.25)
    plt.show()


if __name__ == "__main__":
    main()