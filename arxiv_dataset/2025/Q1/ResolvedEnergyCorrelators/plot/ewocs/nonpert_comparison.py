import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker

from histogram import HistogramData
from plotter import ewoc_data_dir

from qcd.qcd_basics import M_W
from qcd.observable_utils import expected_obs_peak

from ewocs.ewoc_plotter import EWOCPlotter, check_normalization
from ewocs.plot_info import LHC_W_PARAMS, plot_style


nonpert_params = LHC_W_PARAMS.copy()
nonpert_params['jet_algorithm'] = 'akt'
nonpert_params['jet_radius'] = 0.8
nonpert_params['subjet_algorithm'] = 'kt'

connect_inset = False
show_scaling = False
plot_fit = False

plot_feature_locs = {
    'title': (0.5, 0.945),
    'process': (0.02, 0.93),
    'stamp': (0.02, 0.99),
    'peak_text': (0.02, 0.94),
    'main_legend': (0.67, 0.60),
    'line_legend': (.15, 0.05)
}
plot_fontsizes = {
    'title': 18,
    'process': 14,
    'stamp': 12,
    'peak': 20,
    'main_legend': 14,
    'line_legend': 12,
    'x_axis': 24,
    'y_axis': 28,
}


def quadratic_peak_fit(x_data, y_data):
    """
    Estimate the peak location by fitting a quadratic polynomial to the provided data.
    """
    # Fit a 2nd order polynomial (a quadratic) to the data
    # coeffs will be [a, b, c] such that y = a*x^2 + b*x + c
    coeffs = np.polyfit(x_data, y_data, 2)
    a, b, c = coeffs

    # The vertex of a parabola y = a*x^2 + b*x + c is at x = -b/(2a)
    peak_x = -b / (2.0 * a)

    return peak_x, coeffs


if __name__ == "__main__":
    # subjet radii paired with observables
    rs = [0.3, 0.0]
    observables = ['mass', 'deltaR']
    # rs = [0.0]
    # observables = ['deltaR']

    # Weights for all
    weights = [1.0, 2.0, 3.0]

    # Dict of histogram data
    data = {}
    metadata = {}

    # Looping
    for weight, (rsub, pair_obs) in product(weights,
                                zip(rs, observables)):
        if pair_obs == 'deltaR' and weight != 1:
            continue

        data[(pair_obs, rsub, weight)] = {}
        metadata[(pair_obs, rsub, weight)] = {}

        # =================================
        # Parton vs. Hadron vs. MPI
        # =================================
        # Initializing plotter
        plotter = EWOCPlotter(pair_obs,
                              plot_feature_locs,
                              plot_fontsizes, None)
        ax = plotter.axes[0]

        if rsub in [0.3, 0.0]:
            if pair_obs == 'mass':
                if weight == 1:
                    inset_bbox = (.12, .1, .4, .4)
                if weight == 2:
                    inset_bbox = (.68, .1, .3, .3)
                if weight == 3:
                    inset_bbox = (.14, .435, .3, .3)
            if pair_obs == 'deltaR':
                inset_bbox = (.14, .12, .35, .35)

            inset_ax = inset_axes(ax, width="100%", height="100%",
                                  bbox_to_anchor=inset_bbox,
                                  bbox_transform=ax.transAxes)
            # Inset axis scaling
            inset_ax.set_xscale('log')

            if pair_obs == 'mass':
                inset_ax.axvline(M_W, alpha=0.2, lw=3,
                                 color='firebrick')
            if pair_obs == 'deltaR':
                peak = expected_obs_peak('deltaR', energy=14000,
                                         outstate='w',
                                         PID_1=2212, PID_2=2212)
                inset_ax.axvline(peak, alpha=0.2, lw=3,
                                 color='firebrick')

            # Setting up inset tickers
            if pair_obs == 'mass':
                xax_ticker = ticker.StrMethodFormatter("{x:.0f}")
                if weight == 1:
                    yax_ticker = ticker.StrMethodFormatter("{x:.1f}")
                elif weight == 2:
                    yax_ticker = ticker.StrMethodFormatter("{x:.2f}")
                else:
                    yax_ticker = ticker.StrMethodFormatter("{x:.2f}")
            if pair_obs == 'deltaR':
                xax_ticker = ticker.StrMethodFormatter("{x:.3f}")
                yax_ticker = ticker.StrMethodFormatter("{x:.2f}")

            # x axis ticker
            inset_ax.xaxis.set_minor_formatter(xax_ticker)
            inset_ax.xaxis.set_major_formatter(xax_ticker)
            # y axis ticker
            inset_ax.yaxis.set_minor_formatter(yax_ticker)
            inset_ax.yaxis.set_major_formatter(yax_ticker)

            # Inset plot bounds
            if pair_obs == 'mass':
                xlim_res = (77.5, 83.3)
                ylim_res = (1.15, 1.65) if weight == 1 else (
                                (3.8e-2, 5.5e-2) if weight == 3 else
                                (1.9e-1, 2.88e-1))
            if pair_obs == 'deltaR':
                xlim_res = (.284, .316)
                ylim_res = (.885, .94)
                # ylim_res = (.87, .895)

            inset_ax.set_xlim(xlim_res)
            inset_ax.set_ylim(ylim_res)

        # Looping on non-perturbative features
        for level, mpi in zip(['parton', 'hadron', 'hadron'],
                              [False, False, True]):
            # Plotting data
            file_name = f'{pair_obs}_lhc_w_ptmin500_nomax_'\
                +('parton_' if level == 'parton' else '')\
                +('nompi_' if(level == 'hadron' and not mpi)
                  else '')\
                +(f'weight{weight:.1f}_'.replace('.', '-') if weight != 1 else '')\
                +f'jet0-80_subjet{rsub:.2f}'.replace('.', '-')\
                +'.py'

            # Getting data
            hist1d = HistogramData(file_name=ewoc_data_dir/file_name,
                                   variable_order=['pair_obs'])

            vals, ewoc = hist1d.centers['pair_obs'], hist1d.hist
            data[(pair_obs, rsub, weight)][(level, mpi)] = (vals, ewoc)
            metadata[(pair_obs, rsub, weight)][(level, mpi)] = hist1d.metadata

            # Plotting data
            ax.plot(vals, ewoc, lw=2.5,
                    label=('Hadron+MPI' if mpi else
                           ('Hadron' if level=='hadron'
                            else 'Parton')),
                    **plot_style(**hist1d.metadata))

            if rsub in [0.3, 0.0]:
                file_name_res = \
                    f'{pair_obs}_lhc_w_highres120bins_'\
                    +('parton_' if level == 'parton' else '')\
                    +('nompi_' if(level == 'hadron' and not mpi)
                      else '')\
                    +(f'weight{weight:.1f}_'.replace('.', '-') \
                      if weight != 1 else '')\
                    +f'jet0-80_subjet{rsub:.2f}'.replace('.', '-')\
                    +'.py'

                # Getting data
                hist1d_res = HistogramData(
                                file_name=ewoc_data_dir/file_name_res,
                                variable_order=['pair_obs'])

                vals_res, ewoc_res = hist1d_res.centers['pair_obs'], \
                                     hist1d_res.hist



                # Plotting data
                style = plot_style(**hist1d_res.metadata)
                inset_ax.plot(vals_res, ewoc_res, lw=2.5,
                              **style)

                # Plotting fit
                if plot_fit:
                    if pair_obs == 'mass':
                        # lowval, highval
                        l, h = 79.0, 82.0
                    else:
                        l, h = 0.285, 0.315

                    x_res = vals_res[(vals_res > l) & (vals_res < h)]
                    y_res = ewoc_res[(vals_res > l) & (vals_res < h)]
                    try:
                        _, (a, b, c) = quadratic_peak_fit(x_res,
                                                          y_res)
                        fit_res = a*vals_res**2. + b*vals_res+ c
                        inset_ax.plot(vals_res, fit_res, lw=2.0,
                                      alpha=0.3, **style)
                    except:
                        pass

                    # Inset legend
                    inset_ax.legend(
                        [Line2D([], [], lw=0, color='grey',
                                marker='s', markersize=3.0),
                         Line2D([], [], lw=0, color='grey',
                                marker='s', markersize=3.0,
                                alpha=0.3,),],
                        ['Sim.', 'Fit'],
                        loc=(0.25, 0.00), fontsize=13,
                        labelspacing=0.13, handletextpad=-0.55)


                # Indicating inset axes
                rect = Rectangle((xlim_res[0], ylim_res[0]),
                                 xlim_res[1]-xlim_res[0],
                                 ylim_res[1]-ylim_res[0],
                                 linewidth=1, edgecolor='grey',
                                 facecolor='none', zorder=0)
                ax.add_patch(rect)

                if connect_inset:
                    if pair_obs == 'mass':
                        if weight == 1:
                            patch_locs = [(0,1), (1,0)]
                        if weight == 2:
                            patch_locs = [(0,0), (1,1)]
                        if weight == 3:
                            patch_locs = [(1,0), (1,1)]
                    if pair_obs == 'deltaR':
                        patch_locs = [(0,1), (1,0)]
                    for loc in patch_locs:
                        ax.add_patch(ConnectionPatch(
                                        xyA=(xlim_res[loc[0]],
                                             ylim_res[loc[1]]),
                                        xyB=(loc[0], loc[1]),
                                        axesA=ax, axesB=inset_ax,
                                        coordsA="data",
                                        coordsB="axes fraction",
                                        lw=0.7, color='grey'))

        # Processing figure
        ylim = (8e-3, 5e0)
        xlim = (9, 450)
        if pair_obs == 'mass':
            if rsub == 0.03:
                ylim = (1e-6, 1e2)
            if rsub == 0.3:
                ylim = (3e-8, 1e2)
        if pair_obs == 'deltaR':
            xlim = (9e-3, 1.2e0)

        vline_fraction = 0.85
        if pair_obs == 'mass' and rsub == 0.3 and weight == 1:
            vline_fraction = 8
        if pair_obs == 'deltaR':
            vline_fraction = 1.1
        plotter.process_figure(dict(**nonpert_params,
                                    subjet_radius=rsub,
                                    weight_1=weight, weight_2=weight),
                               legend=False,
                               vline_fraction=vline_fraction,
                               vline_alpha=0.1)
        plotter.set_axes(pair_obs, xlim=xlim, ylim=ylim)

        if pair_obs == 'mass':
            legend_loc = (0.6, 0.03)
            if rsub == 0.03:
                ax.text(58, 5e-5, r'$m_W$',
                        color='firebrick', size=16)
            if rsub == 0.3:
                if weight == 1:
                    ax.text(84, 6, r'$m_W$',
                            color='firebrick', size=16)
                if weight == 2:
                    legend_loc = (0.1, 0.03)
                    ax.text(75, 2e0, r'$m_W$',
                            color='firebrick', size=16)
                if weight == 3:
                    ax.text(75, 2e0, r'$m_W$',
                            color='firebrick', size=16)
        if pair_obs == 'deltaR':
            legend_loc = (0.6, 0.75)
            ax.text(1.7e-1, 4e-2,
                r'$c \frac{m_W}{\langle p_{T\,\text{jet}}\rangle}$',
                color='firebrick', size=18)
        plotter.axes[0].legend(loc=legend_loc)

        if pair_obs == 'deltaR':
            inset_ax.tick_params(axis='x', which='minor',
                                 labelsize=8,
                                 labelrotation=30)

        # Saving figure
        plotter.savefig(f'nonpert_comparison/{pair_obs}_pvhvmpi_'+
                    f'weight{weight:.1f}_'.replace('.', '-')+
                    f'rsub{rsub}'.replace('.', '-')+
                    '.pdf')
        plt.close()


    if show_scaling:
        # Looping again, after storing data
        for weight, (rsub, pair_obs) in product(weights,
                                    zip(rs, observables)):
            if pair_obs == 'deltaR' and weight != 1:
                continue

            this_data = data[(pair_obs, rsub, weight)]
            this_metadata = metadata[(pair_obs, rsub, weight)]

            # =================================
            # Relative Scaling
            # =================================
            plotter = EWOCPlotter(pair_obs,
                                  plot_feature_locs,
                                  plot_fontsizes, None)

            # Getting data
            p_xs, parton = this_data[('parton', False)]
            h_xs, hadron = this_data[('hadron', False)]
            m_xs, mpi    = this_data[('hadron', True)]

            # Plotting data
            assert all(np.isclose(p_xs, h_xs)) and \
                all(np.isclose(p_xs, m_xs)), \
                f"data x values are not close:"\
                +f"\n\t{p_xs=}\n\t{h_xs=}\n\t{m_xs=}"

            ax = plotter.axes[0]
            ax.plot(p_xs, abs(hadron-parton), lw=2.5,
                    label='Hadron-Parton',
                    **plot_style(**this_metadata[('hadron', False)]))
            ax.plot(h_xs, abs(mpi-hadron), lw=2.5,
                    label='MPI-Hadron',
                    **plot_style(**this_metadata[('hadron', True)]))

            # Processing figure
            ylim = None
            if pair_obs == 'mass':
                if weight == 1:
                    xlim = (1e-1, 8e2)
                    ylim = (1e-9, 1e2)
                    legend_loc = (0.3, 0.02)
                else:
                    xlim = (1e-1, 8e2)
                    ylim = (1e-12, 1e0)
                    legend_loc = (0.02, 0.45)
                    if rsub == 0.03:
                        legend_loc = (0.33, 0.02)
            if pair_obs == 'deltaR':
                xlim = (8e-6, 4e0)
                ylim = (1e-9, 1e2)
                legend_loc = (0.3, 0.02)
            plotter.set_axes(pair_obs, xlim=xlim, ylim=ylim)
            plotter.process_figure(dict(**nonpert_params,
                                        subjet_radius=rsub),
                                   legend=False,
                                   vline_fraction=1, vline_alpha=0.1)
            ax.set_ylabel(
                r"$\Delta\,\,\frac{\text{d}\Sigma}{\text{d}\log m}$")
            ax.legend(loc=legend_loc)

            # Saving figure
            plotter.savefig(f'scaling/{pair_obs}_absscaling_'+\
                        f'weight{weight:.1f}_'.replace('.', '-')+\
                        f'r{rsub}'.replace('.', '-')+'.pdf')
            plt.close()
            # =================================
            # Relative Scaling
            # =================================
            plotter = EWOCPlotter(pair_obs,
                                  plot_feature_locs,
                                  plot_fontsizes, None)

            # Getting data
            p_xs, parton = this_data[('parton', False)]
            h_xs, hadron = this_data[('hadron', False)]
            m_xs, mpi    = this_data[('hadron', True)]

            # Plotting data
            assert all(np.isclose(p_xs, h_xs)) and \
                all(np.isclose(p_xs, m_xs)), \
                f"data x values are not close:"\
                +f"\n\t{p_xs=}\n\t{h_xs=}\n\t{m_xs=}"

            ax = plotter.axes[0]
            ax.plot(p_xs, abs(hadron-parton)/hadron, lw=2.5,
                    label='Hadron-Parton',
                    **plot_style(**this_metadata[('hadron', False)]))
            ax.plot(h_xs, abs(mpi-hadron)/mpi, lw=2.5,
                    label='MPI-Hadron',
                    **plot_style(**this_metadata[('hadron', True)]))

            # Processing figure
            ylim = None
            if pair_obs == 'mass':
                if weight == 1:
                    xlim = (1e-1, 8e2)
                    ylim = (1e-5, 1e3)
                    legend_loc = (0.3, 0.02)
                else:
                    xlim = (1e-1, 8e2)
                    ylim = (1e-5, 1e3)
            if pair_obs == 'deltaR':
                xlim = (8e-6, 4e0)
                ylim = (1e-5, 1e3)
            plotter.set_axes(pair_obs, xlim=xlim, ylim=ylim)
            plotter.process_figure(dict(**nonpert_params,
                                        subjet_radius=rsub,
                                        weight_1=weight,
                                        weight_2=weight),
                                   legend=False,
                                   vline_fraction=1, vline_alpha=0.1)
            ax.set_ylabel(
                r"$\Delta\,\,\frac{\text{d}\Sigma}{\text{d}\log m}/\Sigma$")
            ax.legend(loc='lower left')

            # Saving figure
            plotter.savefig(f'scaling/{pair_obs}_relscaling_'+\
                        f'weight{weight:.1f}_'.replace('.', '-')+\
                        f'r{rsub}'.replace('.', '-')+'.pdf')
            plt.close()
