import os

basedir = os.path.dirname(__file__) + "/.."

import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .clustering import Timer

# -------------------------------------------------------------------------------------------- #
# -------------------------------------- Postproc tools -------------------------------------- #
# -------------------------------------------------------------------------------------------- #

def compute_count_map_from_label_map(label_map):
    N_elements_per_label = np.bincount(label_map.flatten())
    count_map = np.take(N_elements_per_label, label_map.flatten())
    count_map = np.reshape(count_map, label_map.shape)
    return count_map

def compute_log10mass_map_from_label_map(label_map, mp=6.350795014316703*1e8):
    count_map = compute_count_map_from_label_map(label_map)
    log10mass_map = np.log10(count_map.astype(np.float32) * mp)
    log10mass_map[label_map == 0] = np.NaN
    return log10mass_map

def compute_HMF(M_halos, volume=50**3, mp=6.350795014316703*1e8, bins=20, min=np.log(10**11.0), max=np.log(10**15.)):
    M_halos = M_halos[M_halos > mp]
    tmp_x = np.log(M_halos)
    
    if min==None: min=np.min(tmp_x)
    if max==None: max=np.max(tmp_x)
    
    counts, bin_edges = np.histogram(tmp_x, bins=bins, range=(min, max))
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return np.log10(np.exp(bin_centers)), np.log10(np.array(counts / np.diff(bin_edges) / volume))

def uniform_grid_nd(npix, L=1., endpoint=False):
    assert len(np.shape(npix)) > 0, "Please give npix in form (npixx,npixy,..)"
    L = np.ones_like(npix) * L
    ardim = [np.linspace(0, Li, npixi, endpoint=endpoint) for npixi, Li in zip(npix, L)]
    q = np.stack(np.meshgrid(*ardim, indexing="ij"), axis=-1)
    return q

def tide(lx,ly,lz, q0=np.array((25.,25.,25.)), Np=256, L=50.):
    q = uniform_grid_nd((Np,Np,Np), L=L)
    T = np.diag((lx,ly,lz))
    return np.einsum("...i,ij,...j", q-q0, T, q-q0)/2
    
def semantic_predictions_metrics_vs_thresholds(
    truth,
    semantic,
    min_semantic_threshold=0.001,
    max_semantic_threshold=0.999,
    NN_semantic_threshold = 100
):
    
    list_semantic_thresholds=np.linspace(min_semantic_threshold, max_semantic_threshold, NN_semantic_threshold)

    ground_truth = truth.flatten().astype(bool)
    P = np.sum(ground_truth)
    N = np.sum(~ground_truth)

    tmp_semantic_probabilistic_predictions = semantic.flatten()
    
    semantic_metrics = {
        'frac_collapsed' : np.zeros(NN_semantic_threshold),
        'TPR'            : np.zeros(NN_semantic_threshold),
        'TNR'            : np.zeros(NN_semantic_threshold),
        'PPV'            : np.zeros(NN_semantic_threshold),
        'ACC'            : np.zeros(NN_semantic_threshold),
        'F1'             : np.zeros(NN_semantic_threshold)
    }
    
    timer = Timer()
    for ii in range(NN_semantic_threshold):
        if (ii+1)%20==0:
            timer.logdt("Computing semantic results different thresholds %d/%d" % (ii+1, NN_semantic_threshold))

        tmp_threshold = list_semantic_thresholds[ii]
        tmp_semantic_predictions = tmp_semantic_probabilistic_predictions > tmp_threshold
        
        semantic_metrics['frac_collapsed'][ii] = np.sum(tmp_semantic_predictions) / len(tmp_semantic_predictions)
        
        TP = np.sum(tmp_semantic_predictions * ground_truth)
        TN = np.sum(~tmp_semantic_predictions * ~ground_truth)
        FP = np.sum(tmp_semantic_predictions * ~ground_truth)
        FN = np.sum(~tmp_semantic_predictions * ground_truth)

        semantic_metrics['TPR'][ii] = TP / P
        semantic_metrics['TNR'][ii] = TN / N
        semantic_metrics['PPV'][ii] = TP / (TP + FP)
        semantic_metrics['ACC'][ii] = (TP + TN) / (P + N)
        semantic_metrics['F1'][ii] = 2*TP / (2*TP + FP + FN)
        
    return list_semantic_thresholds, semantic_metrics
    
    
def compute_TPR_vs_true_mass(true_mass, semantic, semantic_threshold, min_mass=11.04, max_mass=14.3, N_bins=22):
    
    bins_edges = np.linspace(min_mass, max_mass, num=N_bins+1, endpoint=True)
    
    mass_bins =  np.zeros(N_bins)
    TPR_vs_true_mass = np.zeros(N_bins)
    probability_median = np.zeros(N_bins)
    probability_mean = np.zeros(N_bins)
    for ii in range(N_bins):
        mask = (bins_edges[ii] < true_mass) & (true_mass < bins_edges[ii+1])
        probability_median[ii] = np.median(semantic[mask])
        probability_mean[ii] = np.mean(semantic[mask])
        P = np.sum(mask)
        TP = semantic[mask] > semantic_threshold
        TPR_vs_true_mass[ii] = np.sum(TP) / P
        mass_bins[ii] = (bins_edges[ii+1] + bins_edges[ii]) / 2
    
    return mass_bins, TPR_vs_true_mass
    
# -------------------------------------------------------------------------------------------- #
# -------------------------------------- plotting tools -------------------------------------- #
# -------------------------------------------------------------------------------------------- #

def plot_loaded_fields(
    delta, potential, truth,
    list_ii_slices=[0, 128],
    size_plot=12,
    fontsize=52,
    fontsize1=42,
    cbar_width = 1.,
    cbar_height = 0.02,
    spacing = 0.01,
):
    
    vmin_mass_delta = np.nanmin(delta)
    vmax_mass_delta = np.nanmax(delta)

    vmin_mass_potential = np.nanmin(potential)
    vmax_mass_potential = np.nanmax(potential)

    vmin_mass_truth = np.nanmin(truth)
    vmax_mass_truth = np.nanmax(truth)

    custom_ticks = np.linspace(0,delta.shape[0],5)[1:-1]
    custom_labels = np.linspace(0,delta.shape[0],5)[1:-1].astype(int).astype(str)

    nrows = len(list_ii_slices)
    ncols = 3

    # Create the GridSpec with space for colorbars on the first row
    gs = mpl.gridspec.GridSpec(nrows, ncols)
    fig = mpl.pyplot.figure(figsize=(ncols*size_plot, len(list_ii_slices)*size_plot))
    
    for ii, ii_slice in enumerate(list_ii_slices):

        ax_delta = fig.add_subplot(gs[ii, 0]) # ax for delta
        tmp_imshow = delta[ii_slice]
        cb_delta = ax_delta.imshow(tmp_imshow, cmap='jet', vmin=vmin_mass_delta, vmax=vmax_mass_delta)
        for axis in ['top','bottom','left','right']:
            ax_delta.spines[axis].set_linewidth(4)
        ax_delta.set_xticks([])
        ax_delta.set_yticks([])
        ax_delta.set_ylabel(r'y position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        ax_delta.set_yticks(custom_ticks)
        ax_delta.set_yticklabels(custom_labels, minor=False, rotation=0)
        for tick in ax_delta.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        ax_delta.yaxis.set_tick_params(width=2, length=12.)

        ax_potential = fig.add_subplot(gs[ii, 1]) # ax for potential
        tmp_imshow = potential[ii_slice]
        vmin_mass = np.nanmin(potential)
        vmax_mass = np.nanmax(potential)
        cb_potential = ax_potential.imshow(tmp_imshow, cmap='jet', vmin=vmin_mass_potential, vmax=vmax_mass_potential)
        for axis in ['top','bottom','left','right']:
            ax_potential.spines[axis].set_linewidth(4)
        ax_potential.set_xticks([])
        ax_potential.set_yticks([])
        
        ax_truth = fig.add_subplot(gs[ii, 2]) # ax for truth
        tmp_imshow = truth[ii_slice]
        vmin_mass = np.nanmin(truth)
        vmax_mass = np.nanmax(truth)
        alpha = np.zeros(tmp_imshow.shape); alpha[tmp_imshow!=0]=1
        cb_truth = ax_truth.imshow(tmp_imshow, alpha=alpha, cmap='prism', vmin=vmin_mass_truth, vmax=vmax_mass_truth)
        for axis in ['top','bottom','left','right']:
            ax_truth.spines[axis].set_linewidth(4)
        ax_truth.set_xticks([])
        ax_truth.set_yticks([])

        if ii == 0:

            cax_delta_x = ax_delta.get_position().x0 + (ax_delta.get_position().width * (1 - cbar_width) / 2)
            cax_delta_width = ax_delta.get_position().width * cbar_width
            cax_delta = fig.add_axes([cax_delta_x, ax_delta.get_position().y1 + spacing, cax_delta_width, cbar_height])
            ticks = np.linspace(vmin_mass_delta, vmax_mass_delta, 5)[1:-1]
            cb = mpl.pyplot.colorbar(cb_delta, cax=cax_delta, orientation='horizontal', ticks=ticks)
            cb.ax.xaxis.tick_top()
            labels = [float(i) for i in np.round(ticks, 1).tolist()]
            cb.ax.set_xticklabels(labels, fontsize=fontsize1, color='k')
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'$\delta$', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)

            cax_potential_x = ax_potential.get_position().x0 + (ax_potential.get_position().width * (1 - cbar_width) / 2)
            cax_potential_width = ax_potential.get_position().width * cbar_width
            cax_potential = fig.add_axes([cax_potential_x, ax_potential.get_position().y1 + spacing, cax_potential_width, cbar_height])
            ticks = np.linspace(vmin_mass_potential, vmax_mass_potential, 5)[1:-1]
            cb = mpl.pyplot.colorbar(cb_potential, cax=cax_potential, orientation='horizontal', ticks=ticks)
            cb.ax.xaxis.tick_top()
            labels = [float(i) for i in np.round(ticks, 1).tolist()]
            cb.ax.set_xticklabels(labels, fontsize=fontsize1, color='k')
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'$\phi$', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)

            cax_truth_x = ax_truth.get_position().x0 + (ax_truth.get_position().width * (1 - cbar_width) / 2)
            cax_truth_width = ax_truth.get_position().width * cbar_width
            cax_truth = fig.add_axes([cax_truth_x, ax_truth.get_position().y1 + spacing, cax_truth_width, cbar_height])
            ticks = np.linspace(vmin_mass_truth, vmax_mass_truth, 5)[1:-1]
            cb = mpl.pyplot.colorbar(cb_truth, cax=cax_truth, orientation='horizontal', ticks=ticks)
            cb.ax.xaxis.tick_top()
            labels = [int(i) for i in np.round(ticks, 1).tolist()]
            cb.ax.set_xticklabels(labels, fontsize=fontsize1, color='k')
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'Halo labels', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)

    for tmp_ax in [ax_delta, ax_potential, ax_truth]:
        tmp_ax.set_xlabel(r'x position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        tmp_ax.set_xticks(custom_ticks)
        tmp_ax.set_xticklabels(custom_labels, minor=False, rotation=0)
        for tick in tmp_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        tmp_ax.xaxis.set_tick_params(width=2, length=12.)

    mpl.pyplot.subplots_adjust(wspace=0.07, hspace=0.07)
    
    return fig
    
    
    
def plot_semantic_metrics_vs_semantic_threshold(
    list_semantic_thresholds,
    semantic_metrics,
    true_collapsed_fraction,
    chaos_metrics = {'TPR':.887, 'TNR':.914,'PPV': 882, 'ACC':.903, 'F1':.884},
    semantic_threshold = 0.589
):
    
    if semantic_threshold == None:
        semantic_threshold = list_semantic_thresholds[np.argmin(semantic_metrics['frac_collapsed'] > true_collapsed_fraction)]

    fig = mpl.pyplot.figure(figsize=(8, 6)) 
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    ax = mpl.pyplot.subplot(gs[0])

    ax.plot(list_semantic_thresholds, semantic_metrics['TPR'], c='limegreen', lw=2)
    ax.plot(list_semantic_thresholds, semantic_metrics['TNR'], c='royalblue', lw=2)
    ax.plot(list_semantic_thresholds, semantic_metrics['PPV'], c='purple', lw=2)
    ax.plot(list_semantic_thresholds, semantic_metrics['ACC'], c='orange', lw=2)
    ax.plot(list_semantic_thresholds, semantic_metrics['F1'], c='gold', lw=2)

    ax.axhline(chaos_metrics['TPR'], c='limegreen', ls='--', lw=2)
    ax.axhline(chaos_metrics['TNR'], c='royalblue', ls='--', lw=2)
    ax.axhline(chaos_metrics['PPV'], c='purple', ls='--', lw=2)
    ax.axhline(chaos_metrics['ACC'], c='orange', ls='--', lw=2)
    ax.axhline(chaos_metrics['F1'], c='gold', ls='--', lw=2)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax.axvline(semantic_threshold, c='k', ls=':', lw=2) # Semantic cut

    fontsize = 26
    fontsize1 = 20
    ax.set_ylabel(r'Score', size=fontsize)

    custom_ticks = np.linspace(0.75, .95, 3)
    custom_labels = np.round(custom_ticks, 2).astype(str)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_labels, minor=False, rotation=0, fontsize=fontsize1)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        tick.set_pad(6.)
    ax.set_ylim([0.73, .97])

    ax.tick_params('both', length=5, width=2, which='major')
    [ii.set_linewidth(2) for ii in ax.spines.values()]

    custom_lines = [
        mpl.lines.Line2D([0], [0], color='limegreen', ls='-', lw=0, marker='s', markersize=12),
        mpl.lines.Line2D([0], [0], color='royalblue', ls='-', lw=0, marker='s', markersize=12),
        mpl.lines.Line2D([0], [0], color='purple', ls='-', lw=0, marker='s', markersize=12),
        mpl.lines.Line2D([0], [0], color='gold', ls='-', lw=0, marker='s', markersize=12),
        mpl.lines.Line2D([0], [0], color='orange', ls='-', lw=0, marker='s', markersize=12)
    ]
    custom_labels = [r'TPR', r'TNR', r'PPV', r'F1', r'ACC']
    legend = ax.legend(custom_lines, custom_labels, loc='upper right',
                       fancybox=True, shadow=True, ncol=5,fontsize=13)
    ax.add_artist(legend)

    custom_lines = [
        mpl.lines.Line2D([0], [0], color='grey', ls='-', lw=2, marker=None, markersize=9),
        mpl.lines.Line2D([0], [0], color='grey', ls='--', lw=2, marker=None, markersize=9),
        mpl.lines.Line2D([0], [0], color='k', ls=':', lw=2, marker=None, markersize=9),
    ]
    custom_labels = [r'Model', r'Chaos', r'Semantic threshold']
    legend = ax.legend(custom_lines, custom_labels, loc='lower right',
                       fancybox=True, shadow=True, ncol=3,fontsize=14)
    ax.add_artist(legend)

    # ------------------------ bottom subplot ------------------------ #

    ax = mpl.pyplot.subplot(gs[1])

    ax.plot(list_semantic_thresholds, semantic_metrics['frac_collapsed'], c='k', lw=2)

    ax.axhline(true_collapsed_fraction, c='k', ls='-.', lw=2) # frac collapsed
    ax.axvline(semantic_threshold, c='k', ls=':', lw=2) # Semantic cut

    ax.set_xlim([-0.01, 1.01])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        tick.set_pad(6.)

    custom_ticks = np.linspace(0.25, .75, 3)
    custom_labels = np.round(custom_ticks, 2).astype(str)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_labels, minor=False, rotation=0, fontsize=fontsize1)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        tick.set_pad(6.)
    ax.set_ylim([0.2, .8])

    ax.tick_params('both', length=5, width=2, which='major')
    [ii.set_linewidth(2) for ii in ax.spines.values()]

    ax.set_ylabel(r'$1 - \beta$', size=fontsize)
    ax.set_xlabel(r'Semantic threshold', size=fontsize)

    custom_lines = [
        mpl.lines.Line2D([0], [0], color='k', ls='-.', lw=2, marker=None, markersize=9)
    ]
    custom_labels = [r'Validation simulations']
    legend = ax.legend(custom_lines, custom_labels, loc='upper right',
                       fancybox=True, shadow=True, ncol=2,fontsize=14)
    ax.add_artist(legend)

    fig.tight_layout()
    
    return fig, semantic_threshold



def plot_semantic_map_predictions(
    truth, semantic, semantic_threshold,
    list_ii_slices=[0, 128],
    size_plot=12,
    fontsize=52,
    fontsize1=42,
    cbar_width = 1.,
    cbar_height = 0.02,
    spacing = 0.01,
):
    
    custom_ticks = np.linspace(0,truth.shape[0],5)[1:-1]
    custom_labels = np.linspace(0,truth.shape[0],5)[1:-1].astype(int).astype(str)

    nrows = len(list_ii_slices)
    ncols = 3

    # Create the GridSpec with space for colorbars on the first row
    gs = mpl.gridspec.GridSpec(nrows, ncols)
    fig = mpl.pyplot.figure(figsize=(ncols*size_plot, len(list_ii_slices)*size_plot))
    
    for ii, ii_slice in enumerate(list_ii_slices):

        ax_truth = fig.add_subplot(gs[ii, 0]) # ax for truth
        tmp_imshow = truth[ii_slice] != 0
        vmin, vmax = 0, 1
        custom_cmap = mpl.colors.ListedColormap([mpl.cm.get_cmap('seismic')(0.), mpl.cm.get_cmap('seismic')(1.)])
        cb_truth = ax_truth.imshow(tmp_imshow, cmap=custom_cmap, vmin=vmin, vmax=vmax)
        for axis in ['top','bottom','left','right']:
            ax_truth.spines[axis].set_linewidth(4)
        ax_truth.set_xticks([])
        ax_truth.set_yticks([])
        ax_truth.set_ylabel(r'y position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        ax_truth.set_yticks(custom_ticks)
        ax_truth.set_yticklabels(custom_labels, minor=False, rotation=0)
        for tick in ax_truth.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        ax_truth.yaxis.set_tick_params(width=2, length=12.)
        
        ax_semantic = fig.add_subplot(gs[ii, 1]) # ax for semantic
        tmp_imshow = semantic[ii_slice]
        cb_semantic = ax_semantic.imshow(tmp_imshow, cmap='seismic', vmin=0, vmax=1)
        for axis in ['top','bottom','left','right']:
            ax_semantic.spines[axis].set_linewidth(4)
        ax_semantic.set_xticks([])
        ax_semantic.set_yticks([])
        
        ax_semanticut = fig.add_subplot(gs[ii, 2]) # ax for confusion map
        semantic_predictions = semantic > semantic_threshold
        ground_truth = truth != 0
        FN = (~semantic_predictions * ground_truth).astype(np.int32) * 1
        TP = (semantic_predictions * ground_truth).astype(np.int32) * 2
        TN = (~semantic_predictions * ~ground_truth).astype(np.int32) * 3
        FP = (semantic_predictions * ~ground_truth).astype(np.int32) * 4
        tmp_imshow = FN + TP + TN + FP
        tmp_imshow = tmp_imshow[ii_slice]
        vmin, vmax = 1, 4
        custom_cmap = mpl.colors.ListedColormap([
            'k',
            'lime',
            'dodgerblue',
            'red'
        ])
        cb_semanticut = ax_semanticut.imshow(tmp_imshow, cmap=custom_cmap, vmin=vmin, vmax=vmax)
        for axis in ['top','bottom','left','right']:
            ax_semanticut.spines[axis].set_linewidth(4)
        ax_semanticut.set_xticks([])
        ax_semanticut.set_yticks([])        

        if ii == 0:

            cax_truth_x = ax_truth.get_position().x0 + (ax_truth.get_position().width * (1 - cbar_width) / 2)
            cax_truth_width = ax_truth.get_position().width * cbar_width
            cax_truth = fig.add_axes([cax_truth_x, ax_truth.get_position().y1 + spacing, cax_truth_width, cbar_height])
            ticks = np.linspace(0.25, 0.75, 2)
            cb = mpl.pyplot.colorbar(cb_truth, cax=cax_truth, orientation='horizontal', ticks=ticks)
            cb.ax.xaxis.tick_top()
            labels = [r'$\notin halo$', r'$\in halo$']
            cb.ax.set_xticklabels(labels, fontsize=fontsize1, color='k')
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'Ground truth halos', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)

            cax_semantic_x = ax_semantic.get_position().x0 + (ax_semantic.get_position().width * (1 - cbar_width) / 2)
            cax_semantic_width = ax_semantic.get_position().width * cbar_width
            cax_semantic = fig.add_axes([cax_semantic_x, ax_semantic.get_position().y1 + spacing, cax_semantic_width, cbar_height])
            ticks = np.linspace(0, 1, 5)
            cb = mpl.pyplot.colorbar(cb_semantic, cax=cax_semantic, orientation='horizontal', ticks=ticks)
            cb.ax.xaxis.tick_top()
            labels = [float(i) for i in np.round(ticks, 1).tolist()]
            cb.ax.set_xticklabels(labels, fontsize=fontsize1, color='k')
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'$P_\mathrm{pred}\left(\in \mathrm{halo} \right)$', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)
            
            cax_semanticut_x = ax_semanticut.get_position().x0 + (ax_semanticut.get_position().width * (1 - cbar_width) / 2)
            cax_semanticut_width = ax_semanticut.get_position().width * cbar_width
            cax_semanticut = fig.add_axes([cax_semanticut_x, ax_semanticut.get_position().y1 + spacing, cax_semanticut_width, cbar_height])
            ticks = np.linspace(1.5, 3.5, 4)
            cb = mpl.pyplot.colorbar(cb_semanticut, cax=cax_semanticut, orientation='horizontal', ticks=ticks)
            cb.ax.xaxis.tick_top()
            labels = [r'False -', r'True +', r'True -', r'False +']
            cb.ax.set_xticklabels(labels, fontsize=fontsize1, color='k')
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'Error map', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)

    for tmp_ax in [ax_truth, ax_semantic, ax_semanticut]:
        tmp_ax.set_xlabel(r'x position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        tmp_ax.set_xticks(custom_ticks)
        tmp_ax.set_xticklabels(custom_labels, minor=False, rotation=0)
        for tick in tmp_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        tmp_ax.xaxis.set_tick_params(width=2, length=12.)

    mpl.pyplot.subplots_adjust(wspace=0.07, hspace=0.07)
    
    return fig
    
    
def plot_TPR_vs_true_mass(mass_bins, TPR_vs_true_mass):

    mass_bins_chaos = np.array([11.11409091, 11.26227273, 11.41045455, 11.55863636, 11.70681818,
       11.855     , 12.00318182, 12.15136364, 12.29954545, 12.44772727,
       12.59590909, 12.74409091, 12.89227273, 13.04045455, 13.18863636,
       13.33681818, 13.485     , 13.63318182, 13.78136364, 13.92954545,
       14.07772727, 14.22590909])

    TPR_vs_true_mass_chaos = np.array([0.73746777, 0.78422449, 0.81624534, 0.8301802 , 0.83600913,
       0.84138568, 0.84957405, 0.86089855, 0.86481533, 0.86995947,
       0.8712267 , 0.88179121, 0.89030832, 0.89032424, 0.90812362,
       0.90079202, 0.90310466, 0.9092058 , 0.9228153 , 0.90007783,
       0.92411626, 0.92229241])

    fig, ax = mpl.pyplot.subplots(1, 1, figsize=(8, 6))
    fontsize = 26
    fontsize1 = 20
    ax.set_ylabel(r'True Positive Rate', size=30)
    ax.set_xlabel(r'$\log_{10}M_{\mathrm{True}}\; [\mathrm{h}^{-1} M_\odot]$', size=30)

    ax.plot(mass_bins, TPR_vs_true_mass, c='lime', lw=4, ls='-')
    ax.plot(mass_bins_chaos, TPR_vs_true_mass_chaos, c='k', lw=4, ls='-')

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        tick.set_pad(6.)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        tick.set_pad(6.)
    ax.yaxis.set_major_locator(mpl.pyplot.MaxNLocator(5))
    ax.xaxis.set_major_locator(mpl.pyplot.MaxNLocator(6))
    ax.set_ylim([0.6, 0.94])
    ax.tick_params('both', length=5, width=2, which='major')
    [ii.set_linewidth(2) for ii in ax.spines.values()]

    # legend lines
    custom_lines = [
        mpl.lines.Line2D([0], [0], color='k', ls='-', lw=4, marker=None, markersize=16),
        mpl.lines.Line2D([0], [0], color='lime', ls='-', lw=4, marker=None, markersize=16),
    ]
    custom_labels = [
        r'Chaotic Simulations',
        r'Predictions',
    ]
    legend = ax.legend(custom_lines, custom_labels, loc='lower right',fancybox=True, shadow=True, ncol=1,fontsize=fontsize1)
    ax.add_artist(legend)

    fig.tight_layout()
    
    return fig
    
    
def plot_instance_mass_map_predictions(
    true_mass, pred_mass,
    list_ii_slices=[0, 128],
    size_plot=12,
    fontsize=52,
    fontsize1=42,
    cbar_width = 1.,
    cbar_height = 0.02,
    spacing = 0.01,
    vmin_mass=11.,
    vmax_mass=None
):
    if vmin_mass == None:
        vmin_mass = np.min([np.nanmin(true_mass), np.nanmin(pred_mass)])
    if vmax_mass == None:
        vmax_mass = np.max([np.nanmax(true_mass), np.nanmax(pred_mass)])
    
    custom_ticks = np.linspace(0,true_mass.shape[0],5)[1:-1]
    custom_labels = np.linspace(0,true_mass.shape[0],5)[1:-1].astype(int).astype(str)

    nrows = len(list_ii_slices)
    ncols = 2

    # Create the GridSpec with space for colorbars on the first row
    gs = mpl.gridspec.GridSpec(nrows, ncols)
    fig = mpl.pyplot.figure(figsize=(ncols*size_plot, len(list_ii_slices)*size_plot))
    
    for ii, ii_slice in enumerate(list_ii_slices):

        ax_true = fig.add_subplot(gs[ii, 0]) # ax for true_mass
        tmp_imshow = true_mass[ii_slice]
        alpha = np.zeros(tmp_imshow.shape); alpha[tmp_imshow!=0]=1
        cb_true = ax_true.imshow(tmp_imshow, alpha=alpha, cmap='jet', vmin=vmin_mass, vmax=vmax_mass)
        for axis in ['top','bottom','left','right']:
            ax_true.spines[axis].set_linewidth(4)
        ax_true.set_xticks([])
        ax_true.set_yticks([])
        ax_true.set_ylabel(r'y position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        ax_true.set_yticks(custom_ticks)
        ax_true.set_yticklabels(custom_labels, minor=False, rotation=0)
        for tick in ax_true.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        ax_true.yaxis.set_tick_params(width=2, length=12.)
        
        ax_pred = fig.add_subplot(gs[ii, 1]) # ax for pred_mass
        tmp_imshow = pred_mass[ii_slice]
        alpha = np.zeros(tmp_imshow.shape); alpha[tmp_imshow!=0]=1
        cb_pred = ax_pred.imshow(tmp_imshow, alpha=alpha, cmap='jet', vmin=vmin_mass, vmax=vmax_mass)
        for axis in ['top','bottom','left','right']:
            ax_pred.spines[axis].set_linewidth(4)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        
        if ii == 0:

            cax_true_x = ax_true.get_position().x0 + (ax_true.get_position().width * (1 - cbar_width) / 2)
            cax_true_width = ax_true.get_position().width * cbar_width
            cax_true = fig.add_axes([cax_true_x, ax_true.get_position().y1 + spacing, cax_true_width, cbar_height])
            ticks = np.linspace(vmin_mass, vmax_mass, 6)[1:-1]
            cb = mpl.pyplot.colorbar(cb_true, cax=cax_true, orientation='horizontal', ticks=ticks)
            cb.ax.xaxis.tick_top()
            labels = [float(i) for i in np.round(ticks, 1).tolist()]
            cb.ax.set_xticklabels(labels, fontsize=fontsize1, color='k')
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'$\log_{10}M_\mathrm{True} \; [\mathrm{h}^{-1} M_\odot]$', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)

            cax_pred_x = ax_pred.get_position().x0 + (ax_pred.get_position().width * (1 - cbar_width) / 2)
            cax_pred_width = ax_pred.get_position().width * cbar_width
            cax_pred = fig.add_axes([cax_pred_x, ax_pred.get_position().y1 + spacing, cax_pred_width, cbar_height])
            ticks = np.linspace(vmin_mass, vmax_mass, 6)[1:-1]
            cb = mpl.pyplot.colorbar(cb_pred, cax=cax_pred, orientation='horizontal', ticks=ticks)
            cb.ax.xaxis.tick_top()
            labels = [float(i) for i in np.round(ticks, 1).tolist()]
            cb.ax.set_xticklabels(labels, fontsize=fontsize1, color='k')
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'$\log_{10}M_\mathrm{Pred} \; [\mathrm{h}^{-1} M_\odot]$', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)

    for tmp_ax in [ax_true, ax_pred]:
        tmp_ax.set_xlabel(r'x position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        tmp_ax.set_xticks(custom_ticks)
        tmp_ax.set_xticklabels(custom_labels, minor=False, rotation=0)
        for tick in tmp_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        tmp_ax.xaxis.set_tick_params(width=2, length=12.)

    mpl.pyplot.subplots_adjust(wspace=0.07, hspace=0.07)
    
    return fig


def violin_plot(
    true_array,
    pred_array,
    vmin_mass=11.052,
    vmax_mass=14.68,
    N_bins = 19,
    fontsize1=17
):    
    if vmin_mass == None:
        vmin_mass = np.min([np.nanmin(true_mass), np.nanmin(pred_mass)])
    if vmax_mass == None:
        vmax_mass = np.max([np.nanmax(true_mass), np.nanmax(pred_mass)])
    
    bins_edges = np.linspace(vmin_mass, vmax_mass, num=N_bins+1, endpoint=True)

    N_sample = int(1.*len(pred_array))
    index = np.random.choice(len(true_array), N_sample, replace=False)
    XX = true_array[index]
    YY = pred_array[index]

    mask_x = np.isfinite(XX)
    mask_y = np.isfinite(YY)
    mask = mask_x & mask_y
    XX = XX[mask]
    YY = YY[mask]

    tmp_mask = (vmin_mass < XX) & (XX < vmax_mass)
    XX = XX[tmp_mask]
    YY = YY[tmp_mask]

    XX_median = np.zeros(N_bins)
    YY_median = np.zeros(N_bins)

    fig, ax = mpl.pyplot.subplots(1, 1, figsize=(6, 6))

    ax.set_xlabel(r'$\log_{10}M_\mathrm{Truth}\; [\mathrm{h}^{-1} M_\odot]$', fontsize=16)
    ax.set_ylabel(r'$\log_{10}M_\mathrm{Pred}\; [\mathrm{h}^{-1} M_\odot]$', fontsize=16)
    ax.set_xlim([vmin_mass, vmax_mass])
    ax.set_ylim([10, 15])

    tmp_xx = np.linspace(vmin_mass, vmax_mass, 100)
    ax.plot(tmp_xx, tmp_xx, c='k')

    xx_violin = []
    yy_violin = []
    for ii in range(N_bins):

        mask = (bins_edges[ii] < XX) & (XX < bins_edges[ii+1])
        tmp_xx = XX[mask]
        tmp_yy = YY[mask]

        XX_median[ii] = np.median(tmp_xx)
        YY_median[ii] = np.median(tmp_yy)

        tmp_indexes_sort = np.argsort(tmp_yy)
        tmp_yy = tmp_yy[tmp_indexes_sort]
        tmp_xx = tmp_xx[tmp_indexes_sort]

        xx_violin.append(bins_edges[ii] + (bins_edges[ii+1] - bins_edges[ii])/ 2)
        yy_violin.append(tmp_yy)

    parts = ax.violinplot(
        yy_violin,
        positions=xx_violin,
        widths=bins_edges[1]-bins_edges[0],
        showextrema=False
    )

    violin_xx = []
    violin_yy = []
    for pc in parts['bodies']:
        pc.set_facecolor('k')
        pc.set_edgecolor('k')
        pc.set_alpha(0.5)

        tmp_violin = pc.get_paths()[0].vertices
        violin_xx.append(tmp_violin[0:int(len(tmp_violin)/2)][:,0])
        violin_yy.append(tmp_violin[0:int(len(tmp_violin)/2)][:,1])
        violin_xx.append(tmp_violin[-int(len(tmp_violin)/2):][:,0])
        violin_yy.append(tmp_violin[-int(len(tmp_violin)/2):][:,1])

    ax.scatter(XX_median, YY_median, s=50, c='k')
    
    # violin_chaos = np.load(os.path.join('../data/violin_chaos.npy'))
    # for ii in range(len(violin_chaos)):
        # ax.plot(
            # violin_chaos[ii,:,0],
            # violin_chaos[ii,:,1],
            # c='limegreen',
            # ls='-', lw=2
        # )
    
    # legend lines hist plots
    custom_lines = [
        mpl.lines.Line2D([0], [0], color='k', ls='None', marker='s', markersize=16),
        # mpl.lines.Line2D([0], [0], color='limegreen', ls='None', marker='s', markersize=16)
    ]

    custom_labels = [
        r'This work',
        # r'Chaotic limit'
    ]
    legend = ax.legend(custom_lines, custom_labels, loc='upper left',fancybox=True, shadow=True, ncol=1,fontsize=fontsize1)
    ax.add_artist(legend)

    
    return fig
    
    
def plot_HMF(xx_true, yy_true, xx_pred, yy_pred):
    
    fig, ax = mpl.pyplot.subplots(1, 1, figsize=(8, 6))
    fontsize = 26
    fontsize1 = 16
    ax.set_ylabel(r'$\log_{10}\left( d\mathrm{n} / d\ln \mathrm{M} \right)$', size=30)
    ax.set_xlabel(r'$\log_{10}M \; [\mathrm{h}^{-1} \mathrm{M}_\odot]$', size=30)

    ax.plot(xx_true, yy_true, lw=4, ls='-', alpha=0.9, c='k')
    ax.plot(xx_pred, yy_pred, lw=4, ls='--', alpha=0.9, c='k')
    
    ax.axvline(12, ls=':', lw=1, c='k')

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        tick.set_pad(6.)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        tick.set_pad(6.)
    # ax.set_ylim([-5.2, -2.])
    # ax.set_xlim([11.65, 14.75])
    ax.tick_params('both', length=5, width=2, which='major')
    [ii.set_linewidth(2) for ii in ax.spines.values()]

    # legend lines
    custom_lines = [
        mpl.lines.Line2D([0], [0], color='k', ls='-', marker=None, lw=4, markersize=16),
        mpl.lines.Line2D([0], [0], color='k', ls='--', marker=None, lw=4, markersize=16),
    ]
    custom_labels = [
        'Truth (simulations)',
        'Pred (simulations)',
    ]
    legend = ax.legend(custom_lines, custom_labels, loc='upper right',fancybox=True, shadow=True, ncol=1,fontsize=fontsize1)
    ax.add_artist(legend)

    custom_ticks = np.linspace(-5., -2., 3)
    custom_labels = np.linspace(-5., -2., 3).astype(str)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_labels, minor=False, rotation=0)

    custom_ticks = np.linspace(11., 15., 5)
    custom_labels = custom_ticks.astype(str)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels, minor=False, rotation=0)

    fig.tight_layout()
    
    return fig
    
    
def plot_experiments(
    experiments_mass,
    ii_slice = 180,
    size_plot=12,
    fontsize=52,
    fontsize1=42,
    cbar_width = 1.,
    cbar_height = 0.02,
    spacing = 0.01,
    vmin_mass = 11.,
    vmax_mass = 14.5,
    FoV = 256
):

    tmp_slice = slice(int(experiments_mass[0].shape[0]/2 - FoV/2), int(experiments_mass[0].shape[0]/2 + FoV/2))
    custom_ticks = np.linspace(0, FoV, 5)[1:-1]
    custom_labels = np.linspace(int(experiments_mass[0].shape[0]/2 - FoV/2), int(experiments_mass[0].shape[0]/2 + FoV/2),5)[1:-1].astype(int).astype(str)
    
    nrows = len(list(experiments_mass.keys()))
    ncols = 1
    
    # Create the GridSpec with space for colorbars on the first row
    gs = mpl.gridspec.GridSpec(nrows, ncols)
    fig = mpl.pyplot.figure(figsize=(ncols*size_plot+3, nrows*size_plot))
    
    for ii, ii_key in enumerate(experiments_mass.keys()):
    
        ax = fig.add_subplot(gs[ii, 0])
        tmp_imshow = experiments_mass[ii_key][ii_slice][tmp_slice, tmp_slice]
        
        alpha = np.zeros(tmp_imshow.shape); alpha[tmp_imshow!=0]=1
        cb = ax.imshow(tmp_imshow.T, origin="lower", alpha=alpha.T, cmap='jet', vmin=vmin_mass, vmax=vmax_mass)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r'z position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_labels, minor=False, rotation=0, fontsize=fontsize)
        ax.yaxis.set_tick_params(width=2, length=12.)
        
        if ii == 0:
    
            cax_x = ax.get_position().x0 + (ax.get_position().width * (1 - cbar_width) / 2)
            cax_width = ax.get_position().width * cbar_width
            cax = fig.add_axes([cax_x, ax.get_position().y1 + spacing, cax_width, cbar_height])
            tmp_custom_ticks = np.linspace(11., 14., 4)
            tmp_custom_labels = np.around(tmp_custom_ticks, 2).astype(str)
            cb = mpl.pyplot.colorbar(cb, cax=cax, orientation='horizontal', ticks=tmp_custom_ticks)
            cb.ax.xaxis.tick_top()
            cb.ax.set_xticks(tmp_custom_ticks)
            cb.ax.set_xticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0)
            cb.ax.tick_params(axis="x", pad=1, color='k')
            cb.outline.set_edgecolor(color='k')
            cb.outline.set_linewidth(2)
            cb.ax.set_title(r'$\log_{10}M \; [\mathrm{h}^{-1} M_\odot]$', fontsize=fontsize, pad=18.)
            cb.ax.tick_params(length=12)
    
    for tmp_ax in [ax]:
        tmp_ax.set_xlabel(r'y position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        tmp_ax.set_xticks(custom_ticks)
        tmp_ax.set_xticklabels(custom_labels, minor=False, rotation=0, fontsize=fontsize)
        tmp_ax.xaxis.set_tick_params(width=2, length=12.)
    
    mpl.pyplot.subplots_adjust(wspace=0.07, hspace=0.07)

    return fig
    
    
def plot_example_combination_lattices(
    experiments,
    ii_slice = 0,
    size_plot=12,
    fontsize=52,
    fontsize1=42,
    cbar_width = 1.,
    cbar_height = 0.02,
    spacing = 0.01,
    FoV = 256
):

    tmp_slice = slice(int(experiments[0].shape[0]/2 - FoV/2), int(experiments[0].shape[0]/2 + FoV/2))
    custom_ticks = np.linspace(0, FoV, 5)[1:-1]
    custom_labels = np.linspace(int(experiments[0].shape[0]/2 - FoV/2), int(experiments[0].shape[0]/2 + FoV/2),5)[1:-1].astype(int).astype(str)
    
    nrows = len(list(experiments.keys()))
    ncols = 1
    
    # Create the GridSpec with space for colorbars on the first row
    gs = mpl.gridspec.GridSpec(nrows, ncols)
    fig = mpl.pyplot.figure(figsize=(ncols*size_plot+3, nrows*size_plot))

    list_titles = ['Lattice1', 'Lattice2', 'Combined']
    grid_lines = [np.linspace(0, 256, 5)[1:-1]-1, np.linspace(0, 256, 5)[1:]-32, None]
    for ii, ii_key in enumerate(experiments.keys()):
        ax = fig.add_subplot(gs[ii, 0])
        tmp_imshow = experiments[ii_key][ii_slice][tmp_slice, tmp_slice]
        
        alpha = np.zeros(tmp_imshow.shape); alpha[tmp_imshow!=0]=1
        cb = ax.imshow(tmp_imshow%1337, alpha=alpha, cmap='prism')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r'y position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_labels, minor=False, rotation=0, fontsize=fontsize)
        ax.yaxis.set_tick_params(width=2, length=12.)

        ax.text(10, 20, list_titles[ii], style='italic', fontsize=42, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round',pad=.2))

        ax.vlines(x=grid_lines[ii], ymin=0, ymax=256, colors='k', ls='--', lw=3)
        ax.hlines(y=grid_lines[ii], xmin=0, xmax=256, colors='k', ls='--', lw=3)

        ax.set_xlim([0, 256])
        ax.set_ylim([256, 0])
        
    for tmp_ax in [ax]:
        tmp_ax.set_xlabel(r'x position $\left[ \mathrm{h}^{-1}\mathrm{Mpc} \right]$', size=fontsize, labelpad=12.)
        tmp_ax.set_xticks(custom_ticks)
        tmp_ax.set_xticklabels(custom_labels, minor=False, rotation=0, fontsize=fontsize)
        tmp_ax.xaxis.set_tick_params(width=2, length=12.)
    
    mpl.pyplot.subplots_adjust(wspace=0.07, hspace=0.07)
    
    return fig