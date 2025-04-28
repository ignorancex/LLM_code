import numpy as np
from matplotlib.lines import Line2D
from itertools import product

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
npyth  = '100k_500bins'

# Ghost properties
ghost_labels = ['1G', '100M', '10M', '1M']
ghost_densities = np.array([1, .1, .01, .001])
ghost_densities *= 1e3

# Plot properties
cumulative = True

show_base = True
show_ghost = True
show_corr = False

legend_y = 0.75 - 0.07*show_base - 0.07*show_ghost - 0.07*show_corr


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



def rho_pair(r, R_jet):
    """Distribution of distances between pairs of points
    uniformly distributed in a disk of radius R_jet."""
    th = np.arccos(r/(2*R_jet))
    return (
            th * np.sin(th)**2. * np.cos(th)
            -
            np.sin(th) * np.cos(th)**2.
            +
            th * np.cos(th)**3.
        ) * 8 / (np.pi * R_jet)


def analytic_subtraction(xs, ys, pt_jet, R_jet, rho_bg):
    """Attempt at analytically producing the original EEC in the
    presence of uniform background with pt density rho_bg.
    """
    jet_corr = pt_jet/(pt_jet+rho_bg*np.pi*R_jet**2)
    bkg_corr = 1 - jet_corr
    # bkg_corr = rho_bg/(pt_jet/(np.pi*R_jet**2)+rho_bg)

    # Old: jet is uniform
    # bg_contribution = rho_pair(xs, R_jet) *
    #                     bkg_corr**2.
    #                     +
    #                     2.*rho_bg*jet_corr
    #                 )

    # New: bkg is uniform, jet is pointlike
    bg_contribution = rho_pair(xs, R_jet) * bkg_corr**2.
    cross_contribution = 2.*bkg_corr*jet_corr * 2*xs/R_jet

    # Correcting and rescaling
    corrected = (ys - bg_contribution) / jet_corr**2.

    return corrected


# =====================================
# Main
# =====================================
if __name__ == "__main__":
    nu_N_col = {
        # 3/2: ('0-50', '3/2', 'mediumseagreen'),
        2  : ('1-00', 'cornflowerblue'),
        3  : ('2-00', 'mediumorchid'),
        4  : ('3-00', 'indianred'),
        # 5  : ('4-00', '5', 'mediumorchid'),
        # 100: ('99-00', '100', 'goldenrod'),
    }

    """
    # =====================================
    # Ghost-Dominated Plots
    # =====================================
    # List of histograms
    ghost_hists = {}
    uniform_hists = {}
    gp_ratios = {}

    # Filename descriptor
    file_desc = f'2particle_pure_ghosts_100k_500bins'

    for N, (nu, col) in nu_N_col.items():
        ghost_hists[N] = plot_1d_enc(
                file_name=enc_data_dir/
                    f'{file_desc}_nu{nu}.py',
                plot_kwargs=plot_1d_params,
                variable_order=['theta1'],
                color=adjust_lightness(col, 0.8),
                ls='solid', label=rf'${N=}$',
                save=None,
                cumulative=cumulative,
                **plot_1d_params,
        )
        line = ghost_hists[N].plot.axes[0].lines[0]
        xs, ys = line.get_xdata(), line.get_ydata()

        # DEBUG: Should be N-dependent
        uniform_hists[N] = rho_pair(xs, 0.8)

        gp_ratios[N] = (xs, uniform_hists[N]/ys)


    # Combining plots
    ghost_plotters = [ghost.plot for ghost in ghost_hists.values()]

    cplotter = combine_plotters(ghost_plotters,
        style_from=None,
        ratio_plot=show_base, ylabel_ratio='Ratio',
        ylim_ratio=(.01, 100),
        **plot_1d_params)


    # Analytic distributions
    for N, (nu, col) in nu_N_col.items():
        xs = gp_ratios[N][0]

        cplotter.axes[0].plot(xs, uniform_hists[N],
              ls='dashed',
              color=adjust_lightness(nu_N_col[N][1], 0.5))

        # Ghosts
        cplotter.axes[1].plot(xs, gp_ratios[N][1],
                      ls='solid',
                      color=adjust_lightness(nu_N_col[N][1], 0.8))

    # Base
    cplotter.axes[1].plot(gp_ratios[N][0],
                          np.ones_like(gp_ratios[N][0]),
                          color='dimgrey', ls='dashed')

    # Legend for line types
    handles = []
    labels = []
    handles.append(Line2D([], [], lw=2, ls='solid',
                          color='darkgray'))
    labels.append('Uniform Ghosts (Numeric)')

    handles.append(Line2D([], [], lw=2, ls='dashed',
                          color='dimgray'))
    labels.append('Uniform Pairs (Analytic)')

    legend2 = cplotter.axes[0].legend(handles, labels,
                                      loc=(0.015, legend_y),
                                      handletextpad=.3,)

    # Legend for N
    handles = []
    labels = []
    for N, (nu, col) in nu_N_col.items():
        handles.append(Line2D([], [], lw=2, ls='solid',
                              color=adjust_lightness(col, 1.1)))
        labels.append(rf'$N={N}$')
    legend1 = cplotter.axes[0].legend(handles, labels,
                                      loc=(0.55, 0.02))

    cplotter.axes[0].add_artist(legend2)

    cplotter.fig.tight_layout()
    cplotter.savefig(f'pure_ghosts_combined_1d.pdf',
                     enc_figure_dir/'ghosts/2particle/')
    """

    # =====================================
    # Pythia Plots
    # =====================================
    # for process, level in product(pythia, ['parton', 'nompi']):
    ghost_props = zip(ghost_labels, ghost_densities)
    for process, level, (ghost_label, ghost_density) in\
            product(pythia, ['nompi'], ghost_props):
        # List of histograms
        base_hists = {}
        ghost_hists = {}
        gb_ratios = {}
        corrected_hists = {}
        cb_ratios = {}

        # Filename descriptors
        base_desc = f'{process}_{level}_{npyth}'
        ghost_desc = f'{process}_{ghost_label}_ghosts_{level}_{npyth}'

        for N, (nu, col) in nu_N_col.items():
            # Filename suffix
            suffix = f'_nu{nu}.py'

            # Base histogram (no ghosts)
            base_hists[N] = plot_1d_enc(
                    file_name=enc_data_dir/
                        f'2particle_{base_desc}{suffix}',
                    plot_kwargs=plot_1d_params,
                    variable_order=['theta1'],
                    color=adjust_lightness(col, 0.5),
                    ls='dotted',
                    save=None,
                    cumulative=cumulative,
                    **plot_1d_params,
            )

            # Histograms with ghosts
            ghost_hists[N] = plot_1d_enc(
                    file_name=enc_data_dir/
                        f'2particle_{ghost_desc}{suffix}',
                    plot_kwargs=plot_1d_params,
                    variable_order=['theta1'],
                    color=adjust_lightness(col, 0.8),
                    ls='dashed',
                    save=None,
                    cumulative=cumulative,
                    **plot_1d_params,
            )

            b_line = base_hists[float(N)].plot.axes[0].lines[0]
            g_line = ghost_hists[float(N)].plot.axes[0].lines[0]

            # Corrected distribtuions
            g_xs, g_ys = g_line.get_xdata(), g_line.get_ydata()
            corrected_hists[N] = analytic_subtraction(g_xs, g_ys,
                                                      500, 0.8,
                                                      ghost_density)

            assert all(np.isclose(g_line.get_xdata(),
                                  b_line.get_xdata()))

            gb_ratios[N] = (b_line.get_xdata(),
                            g_line.get_ydata()/b_line.get_ydata())
            cb_ratios[N] = (b_line.get_xdata(),
                            corrected_hists[N]/b_line.get_ydata())


        # Combining plots
        base_plotters = [base.plot for base in base_hists.values()]
        ghost_plotters = [ghost.plot
                          for ghost in ghost_hists.values()]

        all_plotters = []
        if show_ghost:
            all_plotters = ghost_plotters
        if show_base:
            all_plotters = [*all_plotters, *base_plotters]

        cplotter = combine_plotters(all_plotters,
            style_from=None,
            ratio_plot=show_base, ylabel_ratio='Ratio',
            ylim_ratio=(.01, 100),
            **plot_1d_params)


        # Corrected distributions
        if show_corr:
            for N, (nu, col) in nu_N_col.items():
                xs = cb_ratios[N][0]
                cplotter.axes[0].plot(xs, corrected_hists[N],
                      zorder=0.5, ls='solid',
                      color=adjust_lightness(nu_N_col[N][1], 1.1))
        # Plotting ratios
        for N in nu_N_col.keys():
            if show_corr:
                # Corrected
                cplotter.axes[1].plot(cb_ratios[N][0],
                          cb_ratios[N][1], ls= 'solid',
                          color=adjust_lightness(nu_N_col[N][1], 1.1))

            if show_ghost:
                # Ghosts
                cplotter.axes[1].plot(gb_ratios[N][0],
                          gb_ratios[N][1], ls='dashed',
                          color=adjust_lightness(nu_N_col[N][1], 0.8))

        # Base
        if show_base:
            cplotter.axes[1].plot(gb_ratios[N][0],
                                  np.ones_like(gb_ratios[N][0]),
                                  color='dimgrey', ls='dotted')

        # Stamp
        stamp_1d(cplotter.axes[0], **base_hists[2].metadata)

        # Legend for line types
        handles = []
        labels = []
        if show_corr:
            handles.append(Line2D([], [], lw=2, ls='solid',
                                  color='darkgray'))
            labels.append('Corrected')
        if show_ghost:
            handles.append(Line2D([], [], lw=2, ls='dashed',
                                  color='dimgray'))
            labels.append('Uniform Ghosts')
        if show_base:
            handles.append(Line2D([], [], lw=2, ls='dotted',
                                  color='k'))
            labels.append('Base')

        legend2 = cplotter.axes[0].legend(handles, labels,
                                          loc=(0.015, legend_y),
                                          handletextpad=.3,)

        # Legend for N
        handles = []
        labels = []
        for N, (nu, col) in nu_N_col.items():
            handles.append(Line2D([], [], lw=2, ls='solid',
                                  color=adjust_lightness(col, 1.1)))
            labels.append(rf'$N={N}$')
        legend1 = cplotter.axes[0].legend(handles, labels,
                                          loc=(0.55, 0.02))

        cplotter.axes[0].add_artist(legend2)

        cplotter.fig.tight_layout()
        cplotter.savefig(
            f'{process}_{level}_{ghost_label}_ghosts_combined_1d.pdf',
            enc_figure_dir/'ghosts/2particle/')
