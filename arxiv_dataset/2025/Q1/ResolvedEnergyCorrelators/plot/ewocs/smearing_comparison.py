import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker

from histogram import HistogramData
from plotter import ewoc_data_dir

from utils.plot_utils import adjust_lightness

from qcd.qcd_basics import M_W
from qcd.observable_utils import expected_obs_peak

from ewocs.ewoc_plotter import EWOCPlotter, check_normalization
from ewocs.plot_info import LHC_W_PARAMS, plot_style


smearing_params = LHC_W_PARAMS.copy()
smearing_params['jet_algorithm'] = 'akt'
smearing_params['jet_radius'] = 0.8
smearing_params['subjet_algorithm'] = 'kt'

show_inset = True
# show_inset = False
connect_inset = False
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

    # Weights for all
    weights = [1.0]

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

        # Inset
        if show_inset:
            if rsub in [0.0, 0.3]:
                if pair_obs == 'mass':
                    # inset_bbox = (.308, .10, .28, .28)
                    # inset_bbox = (.108, .10, .28, .28)
                    # inset_bbox = (.308, .10, .26, .26)
                    inset_bbox = (.71, .68, .28, .28)
                if pair_obs == 'deltaR':
                    inset_bbox = (.69, .715, .30, .30)

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
                    xax_ticker = ticker.StrMethodFormatter(
                                                    "{x:.0f}")
                    if weight == 1:
                        yax_ticker = ticker.\
                            StrMethodFormatter("{x:.1f}")
                    else:
                        yax_ticker = ticker.\
                            StrMethodFormatter("{x:.2f}")
                if pair_obs == 'deltaR':
                    xax_ticker = ticker.StrMethodFormatter(
                                                    "{x:.3f}")
                    yax_ticker = ticker.StrMethodFormatter(
                                                    "{x:.2f}")

                # x axis ticker
                inset_ax.xaxis.set_minor_formatter(xax_ticker)
                inset_ax.xaxis.set_major_formatter(xax_ticker)
                # y axis ticker
                inset_ax.yaxis.set_minor_formatter(yax_ticker)
                inset_ax.yaxis.set_major_formatter(yax_ticker)

                # Inset plot bounds
                if pair_obs == 'mass':
                    xlim_res = (77.5, 83.3)
                    ylim_res = (1.15, 1.55) if weight == 1 else (
                                    (3.8e-2, 5.5e-2) if weight == 3 else
                                    (1.9e-1, 2.88e-1))
                if pair_obs == 'deltaR':
                    # xlim_res = (.292, .312)
                    # ylim_res = (.885, .915)
                    xlim_res = (.285, .315)
                    ylim_res = (.83, .924)
                inset_ax.set_xlim(xlim_res)
                inset_ax.set_ylim(ylim_res)

        # Looping on choices
        for charged, smeared in zip([False, True, False],
                                    [False, False, True]):
            level = 'hadron'
            mpi = True

            info  = 'charged_' if charged else ''
            if smeared:
                info += 'onesmeared_'

            # Plotting data
            file_name = f'{pair_obs}_lhc_w_ptmin500_nomax_'\
                +info\
                +f'jet0-80_subjet{rsub:.2f}'.replace('.', '-')\
                +'.py'

            # Getting data
            hist1d = HistogramData(file_name=ewoc_data_dir/file_name,
                                   variable_order=['pair_obs'])

            vals, ewoc = hist1d.centers['pair_obs'], hist1d.hist
            # TOGGLE: shift charged-only mass values (isospin limit)
            # if pair_obs == 'mass' and charged:
            #     vals *= 3/2
            data[(pair_obs, rsub, weight)][(level, mpi)] = \
                    (vals, ewoc)
            metadata[(pair_obs, rsub, weight)][(level, mpi)] = \
                    hist1d.metadata


            style = plot_style(**hist1d.metadata)

            linestyle = 'solid' if smeared else (
                                        (0, (5,2)) if charged else
                                        (0, (2, 1.3)))
            brightness = .6 if smeared else (
                                1.0 if charged else 1.4)

            style.update({'linestyle': linestyle,
                          'color': adjust_lightness(style['color'],
                                                     brightness),
                          'zorder': brightness})

            # Plotting data
            label = ('Charged' if charged else 'All') \
                    + ('' if not smeared else r' (Smeared $p_T$)')
            ax.plot(vals, ewoc, lw=2.5,
                    label=label,
                    **style)

            # Inset
            if show_inset:
                if rsub in [0.3, 0.0]:
                    file_name_res = \
                        f'{pair_obs}_lhc_w_highres120bins_'\
                        +info\
                        +f'jet0-80_subjet{rsub:.2f}'.replace('.', '-')\
                        +'.py'

                    # Getting data
                    hist1d_res = HistogramData(
                                    file_name=ewoc_data_dir/file_name_res,
                                    variable_order=['pair_obs'])

                    vals_res, ewoc_res = \
                        hist1d_res.centers['pair_obs'],\
                        hist1d_res.hist
                    vals_res = np.array(vals_res)
                    ewoc_res = np.array(ewoc_res)

                    # Plotting data
                    inset_ax.plot(vals_res, ewoc_res, lw=2.5,
                                  **style)

                    # Plotting fit
                    if plot_fit:
                        if pair_obs == 'mass':
                            # lowval, highval
                            l, h = 79.0, 82.0
                        else:
                            l, h = 0.285, 0.315
                        x_res = vals_res[(vals_res > l) &
                                         (vals_res < h)]
                        y_res = ewoc_res[(vals_res > l) &
                                         (vals_res < h)]
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
                            loc=(0.26, 0.00 if pair_obs == 'mass'
                                        else 0.35),
                            fontsize=12,
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
        vline_fraction = 0.85
        if pair_obs == 'mass' and rsub == 0.3 and weight == 1:
            vline_fraction = 1.35
        if pair_obs == 'deltaR':
            vline_fraction = 0.9
        plotter.process_figure(dict(**smearing_params,
                                    subjet_radius=rsub,
                                    weight_1=weight, weight_2=weight),
                               legend=False,
                               vline_fraction=vline_fraction,
                               vline_alpha=0.1)
        if pair_obs == 'mass':
            ylim = (8e-2, 2.1e0)
            xlim = (35, 150)
            plotter.set_axes(pair_obs, xlim=xlim, ylim=ylim)
            # Remove scientific notation from axis ticks
            xax_ticker = ticker.StrMethodFormatter("{x:.0f}")
            ax.xaxis.set_minor_formatter(xax_ticker)
            ax.xaxis.set_major_formatter(xax_ticker)
        if pair_obs == 'deltaR':
            xlim = (1.1e-1, 8e-1)
            ylim = (1.7e-1, 1.1e0)
            plotter.set_axes(pair_obs, xlim=xlim, ylim=ylim)
            # Remove scientific notation from axis ticks
            yax_ticker = ticker.StrMethodFormatter("{x:.1f}")
            ax.yaxis.set_minor_formatter(yax_ticker)
            ax.yaxis.set_major_formatter(yax_ticker)
            xax_ticker = ticker.StrMethodFormatter("{x:.1f}")
            ax.xaxis.set_minor_formatter(xax_ticker)
            ax.xaxis.set_major_formatter(xax_ticker)

        if pair_obs == 'mass':
            legend_loc = (0.015, 0.48)
            if rsub == 0.03:
                ax.text(58, 5e-5, r'$m_W$',
                        color='firebrick', size=16)
            if rsub == 0.3:
                if weight == 1:
                    ax.text(78, 0.5, r'$m_W$',
                            color='firebrick', size=16)
                if weight == 2:
                    ax.text(75, 2e0, r'$m_W$',
                            color='firebrick', size=16)
                if weight == 3:
                    ax.text(75, 2e0, r'$m_W$',
                            color='firebrick', size=16)
        if pair_obs == 'deltaR':
            legend_loc = (0.065, 0.015)
            ax.text(3.1e-1, 2e-1,
                r'$c \frac{m_W}{\langle p_{T\,\text{jet}}\rangle}$',
                color='firebrick', size=18)
        ax.legend(loc=legend_loc)

        if pair_obs == 'deltaR' and show_inset:
            inset_ax.tick_params(axis='y', which='both',
                                 labelsize=8,
                                 labelrotation=0)
            inset_ax.tick_params(axis='x', which='both',
                                 labelsize=8,
                                 labelrotation=30)


        # Saving figure
        plotter.savefig(f'smearing_comparison/{pair_obs}_smearing_'+
                        f'rsub{rsub}'.replace('.', '-')+'.pdf')
        plt.close()
