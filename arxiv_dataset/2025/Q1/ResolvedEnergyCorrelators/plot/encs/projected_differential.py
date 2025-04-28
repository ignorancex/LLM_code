import numpy as np

from re import search

from encs.plots import plot_1d_enc

from plotter import enc_data_dir, enc_figure_dir, combine_plotters

# =====================================
# Flags
# =====================================
# Whether to make OD plots
opendata = True

# Which pythia data to plot
pythia = []
# pythia = ['qcd', 'w', 'top']
npyth  = '1M_500bins'


# =====================================
# Plot parameters
# =====================================
# 1D plots
plot_1d_params = {
    key: {
        'axes.labelsize': 20,
        'ylim': (1e-4, 1e1),
        'xlim': (5e-4, 3e0),
        'xlabel': r'$R_1$',
        'ylabel': r'PENC($R_1$)',
        'y_scale': 'log',
        'x_scale': 'log',
    }
    for key in ['opendata', 'qcd', 'w', 'top']
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
    # Opendata Plots
    # =====================================
    if opendata:
        od_nu05 = plot_1d_enc(
                file_name=enc_data_dir/
                        '2particle_od_100k_200bins_nu0-50.py',
                variable_order=['theta1'],
                color='mediumseagreen',
                label=r'$N=3/2$',
                save=None,
                **plot_1d_params['opendata'],
        )
        od_nu1 = plot_1d_enc(
                file_name=enc_data_dir/
                        '2particle_od_100k_200bins_nu1-00.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='cornflowerblue',
                label=r'$N=2$',
                save=None,
                **plot_1d_params['opendata'],
        )
        od_nu2 = plot_1d_enc(
                file_name=enc_data_dir/
                        '2particle_od_100k_200bins_nu2-00.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='mediumorchid',
                label=r'$N=3$',
                save=None,
                **plot_1d_params['opendata'],
        )
        od_nu3 = plot_1d_enc(
                file_name=enc_data_dir/
                        '2particle_od_100k_200bins_nu3-00.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='indianred',
                label=r'$N=4$',
                save=None,
                **plot_1d_params['opendata'],
        )
        od_nu4 = plot_1d_enc(
                file_name=enc_data_dir/
                        '2particle_od_100k_200bins_nu4-00.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='goldenrod',
                label=r'$N=5$',
                save=None,
                **plot_1d_params['opendata'],
        )

        # Combining plots
        od_plotters = [od_nu05.plot, od_nu1.plot,
                       od_nu2.plot, od_nu3.plot,
                       od_nu4.plot]

        cplotter = combine_plotters(od_plotters)
        stamp_1d(cplotter.axes[0], **od_nu1.metadata)
        cplotter.axes[0].legend(loc='lower center')
        cplotter.fig.tight_layout()
        cplotter.savefig(
            f'od_combined_1d.pdf',
            enc_figure_dir/'supplemental/2particle/')


    # =====================================
    # Pythia Plots
    # ==================================={npyth}_comparison==
    for process in pythia:
        if process is None:
            continue

        pythia_nu05 = plot_1d_enc(
                file_name=enc_data_dir/
                        f'2particle_{process}_{npyth}_nu0-50.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='mediumseagreen',
                label=r'$N=3/2$',
                save=None,
                **plot_1d_params[process],
        )
        pythia_nu1 = plot_1d_enc(
                file_name=enc_data_dir/
                        f'2particle_{process}_{npyth}_nu1-00.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='cornflowerblue',
                label=r'$N=2$',
                save=None,
                **plot_1d_params[process],
        )
        pythia_nu2 = plot_1d_enc(
                file_name=enc_data_dir/
                        f'2particle_{process}_{npyth}_nu2-00.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='mediumorchid',
                label=r'$N=3$',
                save=None,
                **plot_1d_params[process],
        )
        pythia_nu3 = plot_1d_enc(
                file_name=enc_data_dir/
                        f'2particle_{process}_{npyth}_nu3-00.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='indianred',
                label=r'$N=4$',
                save=None,
                **plot_1d_params[process],
        )
        pythia_nu4 = plot_1d_enc(
                file_name=enc_data_dir/
                        f'2particle_{process}_{npyth}_nu4-00.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color='goldenrod',
                save=None,
                label=r'$N=5$',
                **plot_1d_params[process],
        )


        # Combining plots
        pythia_plotters = [pythia_nu05.plot, pythia_nu1.plot,
                           pythia_nu2.plot, pythia_nu3.plot,
                           pythia_nu4.plot]

        cplotter = combine_plotters(pythia_plotters)
        stamp_1d(cplotter.axes[0], **pythia_nu1.metadata)
        cplotter.axes[0].legend(loc='lower center')
        cplotter.fig.tight_layout()
        cplotter.savefig(
            f'{process}_combined_1d.pdf',
            enc_figure_dir/'supplemental/2particle/')
