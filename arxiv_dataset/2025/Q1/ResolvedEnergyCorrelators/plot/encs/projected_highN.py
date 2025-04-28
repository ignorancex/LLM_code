import numpy as np

from re import search

from plotter import enc_data_dir, enc_figure_dir, combine_plotters
from encs.plots import plot_1d_enc


# =====================================
# Flags
# =====================================
# Whether to make OD plots
opendata = True

# Which pythia data to plot
pythia = []
# pythia.append('qcd')
# pythia.append('w')
# pythia.append('top')
npyth  = '1M_500bins'
npyth  = '500TeV_100k_500bins'


# =====================================
# Plot parameters
# =====================================
# 1D plots
plot_1d_params = {
    key: {
        'axes.labelsize': 20,
        # 'ylim': (1e-5, 8e0),
        'ylim': (0, 1.3e1),
        'xlim': (5e-3, 2e0),
        'xlabel': r'$R_1$',
        'ylabel':
        r'$\frac{\text{d}\Sigma}{\text{d}\log_{10}R_1}$',
        # 'y_scale': 'log',
        'y_scale': 'lin',
        'x_scale': 'log',
        # 'x_scale': 'lin',
    }
    for key in ['opendata', 'qcd', 'w', 'top']
}

plot_1d_params['opendata']['ylim'] = (0, 7e0)
plot_1d_params['opendata']['xlim'] = (5e-3, 1e0)
# plot_1d_params['opendata']['xlim'] = (1e-5, 2e0)
# plot_1d_params['opendata']['xlim'] = (0, 1.25)


colors = {
    '1-00' : 'mediumseagreen',
    '9-00' : 'cornflowerblue',
    # '49-00' : 'mediumorchid',
    '99-00': 'lightcoral',
    '999-00': 'sandybrown'
}


def stamp_1d(ax, **metadata):
    # Getting jet radius (more complicated than probably needs to be)
    jet_rad = 10*float(metadata['jet_rad'])
    rnd_digit = len(search("\.(0*)", str(jet_rad)).group(1))
    jet_rad = round(jet_rad, rnd_digit)
    jet_info = metadata['jet_alg'].upper()
    if jet_info in ['ANTI-KT', 'ANTIKT', 'AKT']:
        jet_info = 'AK'

    jet_info += str(jet_rad)[:-2] if str(jet_rad)[-2:] == '.0' \
                   else str(jet_rad)
    jet_info += r' Jets, $\!|\eta^\text{jet}| <$ 1.9'
    jet_info += r', $p_T^\text{jet} \in\,$'
    # jet_info += '[500, 550] GeV'
    jet_info += f'[10, 12] TeV'

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
        process_str += r'$\sqrt{s}=\,$'+str(int(metadata['energy']))+' TeV'
        ax.text(0.025, 0.93, process_str,
                fontsize=14, transform=ax.transAxes)


# =====================================
# Main
# =====================================
if __name__ == "__main__":
    # =====================================
    # Opendata Plots
    # =====================================
    if opendata:
        plotters = []
        metadata = []

        for nu, color in colors.items():
            N = eval(nu.replace('-', '.')) + 1
            hist = plot_1d_enc(
                    file_name=enc_data_dir/
                            f'2particle_od_100k_200bins_nu{nu}.py',
                    variable_order=['theta1'],
                    color=color,
                    label=rf'$N={N}$',
                    save=None,
                    **plot_1d_params['opendata'],
            )
            plotters.append(hist.plot)
            metadata.append(hist.metadata)

        # Combining plots
        cplotter = combine_plotters(plotters)
        stamp_1d(cplotter.axes[0], **metadata[0])
        legend = cplotter.axes[0].legend(
                            loc=(0.03, 0.43),
                            handletextpad=0.5)


        cplotter.fig.tight_layout()
        cplotter.savefig(
            f'od_highN_1d.pdf',
            enc_figure_dir/'supplemental/2particle/')


    # =====================================
    # Pythia Plots
    # =====================================
    for process in pythia:
        plotters = []
        metadata = []
        for nu, color in colors.items():
            N = eval(nu.replace('-', '.')) + 1
            hist = plot_1d_enc(
                    file_name=enc_data_dir/
                            f'2particle_{process}_{npyth}_nu{nu}.py',
                    variable_order=['theta1'],
                    color=color,
                    label=rf'$N={N}$',
                    save=None,
                    **plot_1d_params[process],
            )
            plotters.append(hist.plot)
            metadata.append(hist.metadata)

        # Combining plots
        cplotter = combine_plotters(plotters)
        stamp_1d(cplotter.axes[0], **metadata[0])
        cplotter.axes[0].legend(loc=(0.03, 0.43))
        cplotter.fig.tight_layout()
        cplotter.savefig(
            f'{process}_highN_1d.pdf',
            enc_figure_dir/'supplemental/2particle/')
