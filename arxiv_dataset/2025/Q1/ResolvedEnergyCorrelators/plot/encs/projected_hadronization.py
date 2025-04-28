import numpy as np
from matplotlib.lines import Line2D

from re import search

from utils.plot_utils import adjust_lightness
from encs.plots import plot_1d_enc
from plotter import enc_data_dir, enc_figure_dir, combine_plotters

# =====================================
# Flags
# =====================================
# Which pythia data to plot
pythia = []
pythia.append('qcd')
npyth  = '1M_500bins'


# =====================================
# Plot parameters
# =====================================
# 1D plots
plot_1d_params = {
    'axes.labelsize': 20,
    'ylim': (6e-5, 1e2),
    'xlim': (5e-4, 3e0),
    'xlabel': r'$R_1$',
    'ylabel': r'PENC($R_1$)',
    'y_scale': 'log',
    'x_scale': 'log',
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


# =====================================
# Main
# =====================================
if __name__ == "__main__":
    # =====================================
    # Pythia Plots
    # =====================================
    nu_N_col = {
        3/2: ('0-50', '3/2', 'mediumseagreen'),
        2  : ('1-00', '2', 'cornflowerblue'),
        # 3  : ('2-00', 'mediumorchid'),
        # 4  : ('3-00', 'indianred'),
        5  : ('4-00', '5', 'mediumorchid'),
        100: ('99-00', '100', 'goldenrod'),
    }

    hadron_hists = {}
    parton_hists = {}
    p_h_ratios = {}
    for process in pythia:
        for N, (nu, printN, col) in nu_N_col.items():
            hadron_hists[N] = plot_1d_enc(
                    file_name=enc_data_dir/
                        f'2particle_{process}_{npyth}_nu{nu}.py',
                    plot_kwargs=plot_1d_params,
                    variable_order=['theta1'],
                    color=col,
                    label=rf'$N={printN}$',
                    save=None,
                    **plot_1d_params,
            )
            parton_hists[N] = plot_1d_enc(
                    file_name=enc_data_dir/
                        f'2particle_{process}_parton_{npyth}_nu{nu}.py',
                    plot_kwargs=plot_1d_params,
                    variable_order=['theta1'],
                    color=adjust_lightness(col, 0.8),
                    ls='dashed',
                    save=None,
                    **plot_1d_params,
            )

            p_line = parton_hists[float(N)].plot.axes[0].lines[0]
            h_line = hadron_hists[float(N)].plot.axes[0].lines[0]

            assert all(np.isclose(p_line.get_xdata(),
                                  h_line.get_xdata()))

            p_h_ratios[N] = (p_line.get_xdata(),
                             p_line.get_ydata()/h_line.get_ydata())

        # Combining plots
        hadron_plotters = [hadron.plot
                           for hadron in hadron_hists.values()]
        parton_plotters = [parton.plot
                           for parton in parton_hists.values()]

        cplotter = combine_plotters(
            [*hadron_plotters, *parton_plotters],
            style_from=None,
            ratio_plot=True, ylabel_ratio='P/H', ylim_ratio=(.01, 100),
            **plot_1d_params)

        # Plotting ratios
        for N in nu_N_col.keys():
            cplotter.axes[1].plot(p_h_ratios[N][0], p_h_ratios[N][1],
                                  color=nu_N_col[N][2])

        # Stamping, etc.
        stamp_1d(cplotter.axes[0], **hadron_hists[2].metadata)
        legend2 = cplotter.axes[0].legend(
                [Line2D([], [], lw=2, ls='solid', color='gray'),
                 Line2D([], [], lw=2, ls='dashed', color='gray')],
                ['Hadrons', 'Partons'],
                loc=(0.74, 0.78),
                handletextpad=.3,)
        cplotter.axes[0].legend(loc=(0.58, 0.02))
        cplotter.axes[0].add_artist(legend2)
        cplotter.fig.tight_layout()
        cplotter.savefig(
            f'{process}_pvh_combined_1d.pdf',
            enc_figure_dir/'supplemental/2particle/')
