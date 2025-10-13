import numpy as np
import matplotlib.pyplot as plt
from models import ReedMullerCode, BCHCode, MettesCode, LogSumExpCode, RandomCode

def main():
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

    #####
    # Distance histogram for K points in dimension n
    #####
    K=1000
    n=128
    bin_width = 0.01
    bins = np.arange(-1-bin_width/2,1+bin_width/2, bin_width)

    fig, ax = plt.subplots(5, 1, sharex=True)
    fig.subplots_adjust(hspace=0)

    # Random 
    random_codebook = RandomCode(n,K).codebook
    random_corr = (random_codebook @ random_codebook.T)[np.triu_indices(K, 1)] # upper triangular part of corr matrix, excl. diagonal. Automatically flattened
    ax[0].hist(random_corr, bins=bins, density=True, color="tab:gray", label="Random Code")
    ax[0].legend(loc="center left")
    ax[0].set_yticks([])
    ax[0].grid(visible=True, alpha=0.25)

    # Mettes
    mettes_codebook = MettesCode(n, K).codebook
    mettes_corr = (mettes_codebook @ mettes_codebook.T)[np.triu_indices(K, 1)] # upper triangular part of corr matrix, excl. diagonal. Automatically flattened
    ax[1].hist(mettes_corr, bins=bins, density=True, color='tab:cyan', label="Mettes et al. (2019)")
    ax[1].legend(loc="center left")
    ax[1].set_yticks([])
    ax[1].grid(visible=True, alpha=0.25)

    # LSE
    lse_codebook = LogSumExpCode(n, K).codebook
    lse_corr = (lse_codebook @ lse_codebook.T)[np.triu_indices(K, 1)] # upper triangular part of corr matrix, excl. diagonal. Automatically flattened
    ax[2].hist(lse_corr, bins=bins, density=True, color='tab:orange', label=r"Log-sum-exp $(\mathrm{P}_{\mathrm{LSE}})$")
    ax[2].legend(loc="center left")
    ax[2].set_yticks([])
    ax[2].grid(visible=True, alpha=0.25)

    # BCH Code
    bch_codebook = BCHCode(n-1, K).codebook
    bch_corr = (bch_codebook @ bch_codebook.T)[np.triu_indices(K, 1)] # upper triangular part of corr matrix, excl. diagonal. Automatically flattened
    ax[3].hist(bch_corr, bins=bins, density=True, color='tab:blue', label="BCH Code")
    ax[3].legend(loc="center left")
    ax[3].set_yticks([])
    ax[3].grid(visible=True, alpha=0.25)

    # RM Code
    rm_codebook = ReedMullerCode(n, K).codebook
    rm_corr = (rm_codebook @ rm_codebook.T)[np.triu_indices(K, 1)] # upper triangular part of corr matrix, excl. diagonal. Automatically flattened
    ax[4].hist(rm_corr, bins=bins, density=True, color='tab:red', label="RM Code")
    ax[4].legend(loc="center left")
    ax[4].set_yticks([])
    ax[4].grid(visible=True, alpha=0.25)

    # Format axes
    ax[2].set_ylabel(r"Relative Frequency")
    ax[4].set_xlabel(r"$\langle c_i, c_j \rangle$")
    ax[4].set_xlim([-0.55,0.55])
    ax[4].set_xticks([-0.5,-0.25,0,0.25,0.5])

    

    plt.show()



if __name__ == "__main__":
    main()