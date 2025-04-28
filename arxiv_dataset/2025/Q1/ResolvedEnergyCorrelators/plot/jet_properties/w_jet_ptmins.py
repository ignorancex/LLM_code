import numpy as np

from histogram import HistogramData1D
from plotter import property_data_dir, property_figure_dir
from plotter import combine_plotters


# =====================================
# Flags
# =====================================
# Whether to make OD plots
opendata = True

# Which pythia data to plot
pythia = []
pythia.append('qcd')
pythia.append('w')
pythia.append('top')
npyth  = '100k_200bins'

# Plot colors for different jet radii
colors = {
    '400' : 'mediumseagreen',
    '500' : 'mediumorchid',
    '600': 'lightcoral',
}


# =====================================
# Plot parameters
# =====================================
# 1D plots
plot_kwargs = {
    key: {
        'axes.labelsize': 20,
        'ylim': (0, 1e0),
        'xlim': (3.5e2, 2e3),
        'xlabel': r'$p_{T\,\text{jet}}$',
        'ylabel':
        r'$\frac{\text{d}\Sigma}{\text{d}p_T}$',
        'y_scale': 'lin',
        'x_scale': 'lin',
    }
    for key in ['opendata', 'qcd', 'w', 'top']
}


def stamp_1d(ax, **metadata):
    # Getting jet radius (more complicated than probably needs to be)
    jet_info =  r'$\!|\eta^\text{jet}| <$ 1.9'

    # Jet information
    ax.text(0.025, 0.87, jet_info, fontsize=12,
            transform=ax.transAxes)

    process = metadata['outstate_str']
    if process is None or not process in ['qcd', 'w', 'top']:
        raise ValueError(f"Must specify a pythia process in "
                         "[qcd, w, top] when opendata=False.")
    process_str  = r'$\texttt{Pythia 8.310},\,\,pp\to\,\,$'
    process_str += 'hadrons, ' if process == 'qcd' else \
                   (r'$W^+\,W^-\!,$ ' if process == 'w'
                    else r'$t\overline{t},$ ')
    process_str += r'$\sqrt{s}=\,$14 TeV'
    ax.text(0.025, 0.93, process_str,
            fontsize=14, transform=ax.transAxes)


def plot_jet_pt(file_name=None, hist_data=None, **kwargs):
    # Loading and validating histogram
    hist1d = HistogramData1D(file_name=file_name,
                             hist_data=hist_data,
                             variable_order=['pT'])

    # Plotting
    if kwargs:
        # Preparing file name for figure
        save_name = kwargs.pop('save', None)

        # Making plot
        hist1d.make_plot(**kwargs)

        # Saving
        if save_name:
            hist1d.plot.savefig(save_name+'.pdf',
                                property_figure_dir)

    return hist1d



# =====================================
# Main
# =====================================
if __name__ == "__main__":
    # =====================================
    # Pythia Plots
    # =====================================
    process = 'w'
    ptmins = [400, 500, 600]

    plotters = []
    metadata = []
    norms = []
    means = []

    moment = -2

    for ptmin, col in colors.items():
        # Getting histogram and plotting
        label = rf'$p_{{T\,\text{{min}}}} = {ptmin}$ GeV'
        hist = plot_jet_pt(
            file_name=property_data_dir/
                f'lhc_{process}_ptmin{ptmin}_nomax_AKT8_pT.py',
            color=col, label=label, save=None,
            ptmin=ptmin, **plot_kwargs[process])

        # Computing mean
        pts = hist.centers['pT']
        dpt = np.diff(hist.edges['pT'])
        ys  = hist.hist
        norms.append(np.nansum(ys*dpt))
        means.append(np.nansum(pts**moment * ys * dpt))

        # Saving plotters
        plotters.append(hist.plot)
        metadata.append(hist.metadata)

    # Combining plots
    cplotter = combine_plotters(plotters)
    stamp_1d(cplotter.axes[0], **metadata[0])
    cplotter.axes[0].legend(loc=(0.03, 0.42))
    cplotter.fig.tight_layout()
    cplotter.savefig(f'{process}_jet_ptmins.pdf',
                     property_figure_dir)

    print(norms)
    print(np.array(means)**(1/moment))
