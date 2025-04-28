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
    'AKT4' : 'mediumseagreen',
    'AKT6' : 'cornflowerblue',
    'AKT8' : 'mediumorchid',
    'AKT10': 'lightcoral',
    'AKT12': 'sandybrown'
}


# =====================================
# Plot parameters
# =====================================
# 1D plots
plot_kwargs = {
    key: {
        'axes.labelsize': 20,
        # 'ylim': (1e-6, 7e-1),
        'ylim': (0, 8e-2),
        # 'xlim': (1e-1, 1e3),
        'xlim': (0, 2e2),
        'xlabel': r'$m_{\text{jet}}$',
        # 'ylabel':
        # r'$\frac{\text{d}\Sigma}{\text{d}\log_{10}m}$',
        'ylabel':
        r'$\frac{\text{d}\Sigma}{\text{d}m}$',
        # 'y_scale': 'log',
        'y_scale': 'lin',
        # 'x_scale': 'log',
        'x_scale': 'lin',
    }
    for key in ['opendata', 'qcd', 'w', 'top']
}


def stamp_1d(ax, **metadata):
    # Getting jet radius (more complicated than probably needs to be)
    jet_info =  r'$\!|\eta^\text{jet}| <$ 1.9'
    jet_info += r', $p_T^\text{jet} \in\,$[500, 550] GeV'

    # Jet information
    ax.text(0.025, 0.87, jet_info, fontsize=12,
            transform=ax.transAxes)

    # Dataset information
    if metadata['level'] == 'data':
        ax.text(0.025, 0.93,
                r'$\textbf{CMS Open Data}:$ '\
                '2011A Jet Primary Dataset',
                fontsize=14, transform=ax.transAxes)
    else:
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


def plot_jet_mass(file_name=None, hist_data=None, **kwargs):
    # Loading and validating histogram
    hist1d = HistogramData1D(file_name=file_name,
                           hist_data=hist_data)
    print(hist1d)

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
    # Opendata Plots
    # =====================================
    if opendata:
        pass

    # =====================================
    # Pythia Plots
    # =====================================
    for process in pythia:
        plotters = []
        metadata = []
        for label, col in colors.items():
            plotter = plot_jet_mass(
                            file_name=property_data_dir/
                                f'{process}_{label}_{npyth}_mass.py',
                            color=col,
                            label=label,
                            save=None,
                            **plot_kwargs[process],
                        )
            plotters.append(plotter.plot)
            metadata.append(plotter.metadata)

        # Combining plots
        cplotter = combine_plotters(plotters)
        stamp_1d(cplotter.axes[0], **metadata[0])
        cplotter.axes[0].legend(loc=(0.03, 0.42))
        cplotter.fig.tight_layout()
        cplotter.savefig(f'{process}_jet_mass.pdf',
                         property_figure_dir)
