import numpy as np
import sys
import matplotlib.pyplot as plt
import diagnostics as dg


def plot_histograms(samples_list, nplot, labels, savefolder=None, suptitle='', reference_samples=None):

    nitems = len(samples_list)
    if len(labels) < nitems:
        labels = [f"sampler{i}" for i in range(nitems)]
    
    fig, ax = plt.subplots(1, nplot, figsize = (14, 3.5))    
    for i in range(nplot):
        if reference_samples is not None:
            ax[i].hist(reference_samples[..., i].flatten(), density=True, alpha=1, bins='auto', 
                    lw=2, histtype='step', color='k', label='Reference')
        for j in range(nitems):
            ax[i].hist(samples_list[j][..., i].flatten(), density=True, alpha=0.5, bins='auto', label=labels[j])
    
    plt.legend(loc=1)
    plt.suptitle(suptitle)
    if savefolder is not None: plt.savefig(f"{savefolder}/hist")
    return fig, ax


def plot_corner():
    pass


def boxplot_cost():
    pass


def boxplot_rmse(reference_samples, samples_list, counts_list, labels, 
                relative1='scatter', relative2='scatter', 
                savefolder=None, suptitle='', savename='rmse'):

    nitems = len(samples_list)
    if len(labels) < nitems:
        labels = [f"sampler{i}" for i in range(nitems)]

    toplot, toplot2, lbls = [], [], []
    for i in range(len(samples_list)):
        count, err1 = dg.cumulative_rmse_per_chain(samples_list[i], 
                                                    counts=counts_list[i], ref_samples=reference_samples, 
                                                    mode=1, relative=relative1)
        count, err2 = dg.cumulative_rmse_per_chain(samples_list[i], 
                                                    counts=counts_list[i], ref_samples=reference_samples, 
                                                    mode=2, relative=relative2)
        toplot.append([d[-1] for d in err1])
        toplot2.append([d[-1] for d in err2])
        
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].boxplot(toplot, patch_artist=True,
                boxprops=dict(facecolor='C0', color='C0', alpha=0.5), labels=labels);
    ax[1].boxplot(toplot2, patch_artist=True,
                boxprops=dict(facecolor='C0', color='C0', alpha=0.5), labels=labels);

    for axis in [ax[1]]:
        axis.set_xticks(axis.get_xticks(), axis.get_xticklabels(), rotation=45, ha='right')

    for axis in ax:
        axis.grid(which='both', lw=0.3)
        axis.set_yscale('log')
    ax[0].axhline(np.mean(toplot[0]), color='k', ls="--")
    ax[1].axhline(np.mean(toplot2[0]), color='k', ls="--")
        
    ax[0].set_ylabel(r'<$\theta$>', fontsize=12)
    ax[1].set_ylabel(r'<$\theta^2$>', fontsize=12)
    plt.suptitle(f'{suptitle} : RMSE of z-score')
    plt.tight_layout()
    if savefolder is not None: plt.savefig(f"{savefolder}/{savename}")
    
    return fig, ax



# def corner(samples, savepath="./tmp/", savename='corner', save=True, maxdims=10):
#     '''Make corner plot for the distribution from samples                                                                                                                                                                                                                                                                     
#     '''
#     D = min(samples.shape[1], maxdims)

#     fig, ax = plt.subplots(D, D, figsize=(3*D, 2*D), sharex='col')

#     for i in range(D):
#         for j in range(D):
#             if i==j:
#                 ax[i, j].hist(samples[:, i])
#                 ax[i, j].set_title('W[{}]'.format(i))
#             elif i>j:
#                 ax[i, j].plot(samples[:, j], samples[:, i], '.')
#             else:
#                 ax[i, j].axis('off')

#     plt.tight_layout()

#     if save:
#         plt.savefig(savepath + savename)
#         plt.close()
#     else: return fig, ax

