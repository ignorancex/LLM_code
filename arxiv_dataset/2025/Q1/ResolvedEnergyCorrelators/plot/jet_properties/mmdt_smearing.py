import numpy as np

from histogram import HistogramData1D
from plotter import property_data_dir, property_figure_dir
from plotter import combine_plotters


# =====================================
# Flags
# =====================================
# Plot colors for different jet radii
colors = {
    'smeared' : 'lightgray',
    # 'charged' : 'dimgray',
    'all': 'k',
}
linestyles = {
    'smeared': 'solid',
    'charged': (0, (5,2)),
    'all': (0, (2, 1.3)),
}
labels = {
    'smeared': r'All (Smeared $p_T$)',
    'charged': 'Charged',
    'all': 'All',
}

info = {
    'smeared': 'smeared_',
    'charged': 'charged_',
    'all': '',
}

# =====================================
# Plot parameters
# =====================================
# 1D plots
plot_kwargs = {
    key: {
        'axes.labelsize': 20,
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
        hist1d.make_plot(xlabel=r'$p_{T\,\text{jet}}$ (GeV)',
                xlim=(3.5e2, 2e3),
                ylabel=r'$\frac{\text{d}\Sigma}{\text{d}p_T}$',
                ylim=(0,1e-1),
                **kwargs)

        # Saving
        if save_name:
            hist1d.plot.savefig(save_name+'.pdf',
                                property_figure_dir)

    return hist1d

def plot_jet_mass(file_name=None, hist_data=None, **kwargs):
    # Loading and validating histogram
    hist1d = HistogramData1D(file_name=file_name,
                             hist_data=hist_data,
                             variable_order=['mass'])

    # Plotting
    if kwargs:
        # Preparing file name for figure
        save_name = kwargs.pop('save', None)

        # Making plot
        hist1d.make_plot(xlabel=r'$m_{\text{jet}}$ (GeV)',
                xlim=(0, 420),
                ylabel=r'$\frac{\text{d}\Sigma}{\text{d}m}$',
                ylim=(0,1e-1),
                **kwargs)

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

    pt_plotters = []
    mass_plotters = []
    pt_metadata = []
    mass_metadata = []

    for key, col in colors.items():
        ls = linestyles[key]
        label = labels[key]

        # pT: Getting histograms and plotting
        hist = plot_jet_pt(
            file_name=property_data_dir/
                f'groomed_mmdt_lhc_{process}_{info[key]}ptmin500_AKT8_Escheme_pT.py',
            color=col, ls=ls, label=label, save=None,
            ptmin=500, **plot_kwargs[process])
        # Saving plotters
        pt_plotters.append(hist.plot)
        pt_metadata.append(hist.metadata)

        # mass: Getting histograms and plotting
        hist = plot_jet_mass(
            file_name=property_data_dir/
                f'lhc_{process}_ptmin500_nomax_AKT8_{info[key]}mass.py',
            color=col, ls=ls, label=label, save=None,
            ptmin=500, **plot_kwargs[process])
        # Saving plotters
        mass_plotters.append(hist.plot)
        mass_metadata.append(hist.metadata)

    # Combining pT plots
    cplotter = combine_plotters(pt_plotters)
    stamp_1d(cplotter.axes[0], **pt_metadata[0])
    cplotter.axes[0].legend(loc=(0.03, 0.42))
    cplotter.fig.tight_layout()
    cplotter.savefig(f'{process}_smearing_jet_pts.pdf',
                     property_figure_dir)

    # Combining mass plots
    cplotter = combine_plotters(mass_plotters)
    stamp_1d(cplotter.axes[0], **mass_metadata[0])
    cplotter.axes[0].legend(loc=(0.03, 0.42))
    cplotter.fig.tight_layout()
    cplotter.savefig(f'{process}_smearing_jet_masses.pdf',
                     property_figure_dir)
