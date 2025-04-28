import numpy as np
from matplotlib import ticker

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from histogram import HistogramData
from plotter import ewoc_data_dir, property_data_dir

from utils.plot_utils import adjust_lightness
from qcd.qcd_basics import M_W
from qcd.observable_utils import pt_fudge, pt_val_from_min,\
    default_moment

from ewocs.ewoc_plotter import EWOCPlotter, check_normalization
from ewocs.plot_info import LHC_W_PARAMS, plot_style


plot_feature_locs = {
    'title': (0.5, 0.945),
    'process': (0.02, 0.93),
    'peak_text': (0.5, 0.94),
    'main_legend': (0.58, 0.68),
    'line_legend': (.72, 0.44),
    'stamp': (0.025, 0.9),
}
plot_fontsizes = {
    'title': 18,
    'process': 16,
    'peak': 11.5,
    'main_legend': 15,
    'line_legend': 12,
    'stamp': 13,
    'x_axis': 24,
    'y_axis': 28,
}


def deltaR_text(moment=default_moment):
    if moment == 2:
        return r'$c \sqrt{\!\frac{m_W^2}{\langle p_{T\,\text{jet}}^2\rangle}\,}$'
    if moment == 1:
        return r'$c \frac{m_W}{\langle p_{T\,\text{jet}}\rangle}$'
    if moment == 1:
        return r'$c \big\langle\frac{m_W}{\langle p_{T\,\text{jet}}}\big\rangle$'
    if moment == -2:
        return r'$c \sqrt{\!\big\langle\frac{m_W^2}{p_{T\,\text{jet}}^2}\big\rangle\,}$'


show_inset = True
# show_inset = False
plot_fit = False


def deltaR_val(pt_min, moment=default_moment):
    energyhat = pt_val_from_min(pt_min)
    return 2*pt_fudge()*M_W / energyhat


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
    # Parameter setup
    vary_pt_params = LHC_W_PARAMS.copy()
    vary_pt_params['jet_algorithm'] = 'akt'
    vary_pt_params['jet_radius'] = 0.8
    vary_pt_params['subjet_algorithm'] = 'kt'
    vary_pt_params['level'] = 'hadron'
    vary_pt_params['mpi'] = True

    fixed_pt_params = vary_pt_params.copy()
    fixed_pt_params['pt_min'] = 500


    # =================================
    # W estimation:
    # =================================
    for pair_obs, r_sub in zip(['mass', 'deltaR'], [0.3, 0.0]):
        if pair_obs == 'deltaR':
            plot_feature_locs.update({
                'stamp': (0.025, 1.05),
                'peak_text': (0.834, 0.575),
            })
            vline_frac = 0.783
            fwhm_symbol = r'~'
        else:
            plot_feature_locs.update({
                'stamp': (0.025, 1.03),
                'peak_text': (0.78, 0.935),
            })
            vline_frac = 0.818
            fwhm_symbol = '='

        plotter = EWOCPlotter(pair_obs,
                              plot_feature_locs,
                              plot_fontsizes, None)
        ax = plotter.axes[0]

        # Plotting data
        file_name = f'{pair_obs}_lhc_w_ptmin500_nomax_'\
                    f'jet0-80_subjet{r_sub:.2f}'.replace('.', '-')+\
                    '.py'

        hist1d = HistogramData(file_name=ewoc_data_dir/file_name,
                               variable_order=['pair_obs'])
        check_normalization(hist1d)

        vals, ewoc = hist1d.centers['pair_obs'], hist1d.hist

        ax.plot(vals, ewoc, color='black', lw=2.5)

        # Processing figure
        # TOGGLE: LINEAR
        # plotter.set_axes(pair_obs, yscale='linear')
        # xlim = None if pair_obs == 'deltaR' else (0.2, 400)
        # ylim = (1e-3, 200) if pair_obs == 'deltaR' else (1e-8, 300)
        # TOGGLE: LOGARITHMIC
        xlim = (9.1e-2, 0.83) if pair_obs == 'deltaR' else (40, 150)
        ylim = (1.85e-1, 1.4) if pair_obs == 'deltaR' else (3e-2, 3.3)
        plotter.set_axes(pair_obs, xlim=xlim, ylim=ylim)
        plotter.process_figure(fixed_pt_params,
                               show_w_guess=(pair_obs=='mass'),
                               fwhm_symbol=fwhm_symbol,
                               vline_fraction=vline_frac)
        if pair_obs == 'mass':
            ax.text(77, 1.65, r'$m_W$', color='firebrick', size=16)
        if pair_obs == 'deltaR':
            ax.text(0.27, 1.0, deltaR_text(),
                    color='firebrick', size=16)

        # Remove scientific notation from axis ticks
        if pair_obs == 'deltaR':
            xax_ticker = ticker.StrMethodFormatter("{x:.1f}")
            ax.xaxis.set_minor_formatter(xax_ticker)
            ax.xaxis.set_major_formatter(xax_ticker)
            yax_ticker = ticker.StrMethodFormatter("{x:.1f}")
            ax.yaxis.set_minor_formatter(yax_ticker)
            ax.yaxis.set_major_formatter(yax_ticker)
        else:
            xax_ticker = ticker.StrMethodFormatter("{x:.0f}")
            ax.xaxis.set_minor_formatter(xax_ticker)
            ax.xaxis.set_major_formatter(xax_ticker)

        # Saving figure
        plotter.savefig(f'lhc_tuning/{pair_obs}_w_estimation.pdf')


    # =================================
    # EWOC vs. (groomed) Mass:
    # =================================
    xlim = (2, 450)
    ylim = (4e-5, 20)

    plotter = EWOCPlotter('mass',
                    plot_feature_locs,
                    plot_fontsizes, None)
    ax = plotter.axes[0]

    insets = []

    # Inset
    if show_inset:
        for inset_bbox, ylim_res in zip(
                [(.78, .15, .2, .2), (.805, .815, .2, .2)],
                [(1.23, 1.65), (9.0, 23.5)],
            ):
            xlim_res = (77.8, 82.8)

            inset_ax = inset_axes(ax, width="100%", height="100%",
                                  bbox_to_anchor=inset_bbox,
                                  bbox_transform=ax.transAxes)
            # Inset axis axes, scaling, lines
            inset_ax.set_xscale('log')

            inset_ax.set_xlim(*xlim_res)
            inset_ax.set_ylim(*ylim_res)

            inset_ax.axvline(M_W, alpha=0.2, lw=3,
                             color='firebrick')

            # Setting up inset tickers
            xax_ticker = ticker.StrMethodFormatter("{x:.0f}")
            yax_ticker = ticker.StrMethodFormatter("{x:.1f}")

            # x axis ticker
            inset_ax.xaxis.set_minor_formatter(xax_ticker)
            inset_ax.xaxis.set_major_formatter(xax_ticker)
            inset_ax.tick_params(axis='x', which='both',
                                 labelsize=8,
                                 labelrotation=45)
            inset_ax.tick_params(axis='y', which='both',
                                 labelsize=12)
            # y axis ticker
            inset_ax.yaxis.set_minor_formatter(yax_ticker)
            inset_ax.yaxis.set_major_formatter(yax_ticker)

            # Indicating inset axes
            rect = Rectangle((xlim_res[0], ylim_res[0]),
                             xlim_res[1]-xlim_res[0],
                             ylim_res[1]-ylim_res[0],
                             linewidth=1, edgecolor='grey',
                             facecolor='none', zorder=0)
            ax.add_patch(rect)

            insets.append(inset_ax)

    # Plotting groomed masses
    for groomer, label, color in zip(['mmdt', 'sd2'],
                ['mMDT Mass '
                 r'{\normalsize{($z_\text{cut}=0.1$)}}',
                 'Soft Drop Mass '
                 r'{\normalsize{($\beta_\text{SD}=2,\,z_\text{cut}=0.1$)}}'],
                ['orange', 'cadetblue']):
        if groomer != 'mmdt': continue

        ax.plot(-1, -1, label=label, color=color, lw=2.5)

        # Both MPI and no MPI
        for i, (info, linestyle) in enumerate(zip(
                    ['', '_nompi', '_parton'],
                    ['solid', (0, (5,2)),
                     (0, (2, 1.3))])):
            file_name = f'groomed_{groomer}_zc1_lhc_w{info}_ptmin500_'\
                        +'AKT8_Escheme_mass.py'
            lightness = 0.75 if i == 0 else (0.95 if i == 1 else \
                                            (1.45 if i == 2 else None))
            linecolor = adjust_lightness(color, lightness)
            style = {'lw': 2, 'ls': linestyle, 'color': linecolor}

            hist1d = HistogramData(
                            file_name=property_data_dir/file_name,
                            variable_order=['mass'])
            vals, dist = hist1d.centers['mass'], hist1d.hist

            ax.plot(vals, dist, **style)

            if show_inset:
                file_name_res = f'groomed_{groomer}_zc1_lhc_w_'\
                    +f'highres120bins{info}_mass.py'
                hist1d = HistogramData(
                                file_name=property_data_dir/
                                          file_name_res,
                                variable_order=['mass'])
                vals, dist = hist1d.centers['mass'], hist1d.hist
                for inset_ax in insets:
                    inset_ax.plot(vals, dist, **style)

                    if plot_fit:
                        # lowval, highval
                        l, h = 79.5, 81.5

                        x = vals[(vals > l) & (vals < h)]
                        y = dist[(vals > l) & (vals < h)]

                        _, (a, b, c) = quadratic_peak_fit(x, y)
                        style.update({'lw': 2.0})

                        fit = a*vals**2. + b*vals+ c
                        inset_ax.plot(vals, fit, alpha=0.3, **style)

                    if plot_fit:
                        inset_ax.legend(
                            [Line2D([], [], lw=0, color='grey',
                                    marker='s', markersize=1.0),
                             Line2D([], [], lw=0, color='grey',
                                    marker='s', markersize=1.0,
                                    alpha=0.3,),],
                            ['Sim.', 'Fit'],
                            loc=(0.25, 0.00), fontsize=8,
                            labelspacing=0.1, handletextpad=-0.55)



    # Plotting EWOC
    rsub = 0.3

    label = r'Mass EWOC {\normalsize{($r_\text{sub}=0.3$)}}'
    ax.plot(-1, -1, label=label, lw=3.0, color='firebrick')

    for info in ['', '_nompi', '_parton']:
        file_name = f'mass_lhc_w_ptmin500_nomax{info}_'\
                    f'jet0-80_subjet0-30.py'

        hist1d = HistogramData(file_name=ewoc_data_dir/file_name,
                               variable_order=['pair_obs'])
        style = plot_style(adjust_lightness=True,
                           **hist1d.metadata)
        style.update({'lw': 3})
        vals, dist = hist1d.centers['pair_obs'], hist1d.hist

        ax.plot(vals, dist, **style)

        if show_inset:
            file_name_res = f'mass_lhc_w_highres120bins{info}_'\
                +f'jet0-80_subjet{rsub:.2f}'.replace('.', '-')\
                +'.py'
            hist1d = HistogramData(
                            file_name=ewoc_data_dir/
                                      file_name_res,
                            variable_order=['pair_obs'])
            vals, dist = hist1d.centers['pair_obs'], hist1d.hist
            for inset_ax in insets:
                inset_ax.plot(vals, dist, **style)

                if plot_fit:
                    # lowval, highval
                    l, h = 79.5, 81.5

                    x = vals[(vals > l) & (vals < h)]
                    y = dist[(vals > l) & (vals < h)]

                    _, (a, b, c) = quadratic_peak_fit(x, y)
                    style.update({'lw': 2.0})

                    fit = a*vals**2. + b*vals+ c
                    inset_ax.plot(vals, fit, alpha=0.3, **style)

            if plot_fit:
                inset_ax.legend(
                    [Line2D([], [], lw=0, color='grey',
                            marker='s', markersize=1.0),
                     Line2D([], [], lw=0, color='grey',
                            marker='s', markersize=1.0, alpha=0.3,),],
                    ['Sim.', 'Fit'],
                    loc=(0.25, 0.00), fontsize=8,
                    labelspacing=0.1, handletextpad=-0.55)

    # Processing figure
    # xlim = (0.2, 400)
    plotter.set_axes('mass', xlim=xlim, ylim=ylim)

    # Custom level legend
    mline = Line2D([0], [0], label='Hadron+MPI', color='k')
    hline = Line2D([0], [0], label='Hadron',
                   color=adjust_lightness('dimgrey', 0.9),
                   linestyle=(0, (5, 3)))
    pline = Line2D([0], [0], label='Parton',
                   color=adjust_lightness('darkgrey', 0.9),
                   linestyle=(0, (2, 1.3)))
    # level_legend = ax.legend(loc=(0.62, 0.75),
    level_legend = ax.legend(loc=(0.015, 0.35),
                             handles=[mline, hline, pline])

    # Axis labels
    ax.set_xlabel(r'$m$ (GeV)')
    ax.xaxis.get_label().set_fontsize(plot_fontsizes['x_axis'])
    # ax.set_ylabel(r'$\frac{d\,\Sigma}{d\,\log m}$',)
    # ax.yaxis.get_label().set_fontsize(plot_fontsizes['y_axis'])
    ax.set_ylabel('Distribution')
    ax.yaxis.get_label().set_fontsize(20)

    # plotter.stamp(*(0.25, 0.20),
    plotter.stamp(*(0.025, 0.93),
                  line_0=r'\texttt{Pythia 8.307}',
                  line_1=r'$p\,p$ to $W^+ W^-\!$, '+
                         r'$\sqrt{s}$=14.0 TeV',
                  line_2='AKT8 Jets, 500 GeV '+
                         r'$< p_{T,\,\text{jet}}$',
                  fontsize=plot_fontsizes['stamp'],
                  ha='left')
    ax.vlines(M_W, -1, 2e1, lw=3.5, colors='firebrick',
              zorder=0, alpha=0.3).set_capstyle('round')
    ax.text(85, 2e-2, r'$m_W$', color='firebrick', size=16)

    # ax.legend(loc=(.02, .75))
    ax.legend(loc=(.15, .01))
    ax.add_artist(level_legend)

    # Saving figure
    plotter.savefig('lhc_tuning/mass_grooming_comparison.pdf')


    # =================================
    # Different subjet radii
    # =================================
    xlim = (5, 250)
    ylim = (4e-5, 20)

    plot_fontsizes.update({'peak': 18})
    plot_feature_locs.update({
    #     'stamp': (0.025, 0.95),
    #     'main_legend': (0.65, 0.55)
        'stamp': (0.025, 0.97),
        'main_legend': (0.345, 0.015)
    })
    plot_fontsizes.update({'stamp': 14})

    # Initializing plotter
    plotter = EWOCPlotter('mass',
                          plot_feature_locs,
                          plot_fontsizes, None)
    ax = plotter.axes[0]

    # Plotting data
    for r_sub in [0.0, 0.03, 0.3, 0.6]:
        file_name = f'mass_lhc_w_ptmin500_nomax_'\
                f'jet0-80_subjet{r_sub:.2f}'.replace('.', '-')\
                +'.py'
        hist1d = HistogramData(file_name=ewoc_data_dir/file_name,
                               variable_order=['pair_obs'])
        check_normalization(hist1d)

        vals, ewoc = hist1d.centers['pair_obs'], hist1d.hist

        ax.plot(vals, ewoc, label=rf'$r_\text{{sub}}={r_sub}$',
                lw=3 if r_sub==0.3 else 2,
                **plot_style(adjust_lightness=False,
                             **hist1d.metadata))

    # Processing figure
    # plotter.set_axes('mass')
    plotter.set_axes('mass', xlim=xlim, ylim=ylim)
    plotter.process_figure(fixed_pt_params)
    ax.text(84, 4e0, r'$m_W$', color='firebrick', size=16)

    # Saving figure
    plotter.savefig('lhc_tuning/mass_subjet_choice.pdf')


    # =================================
    # Different pT_mins
    # =================================
    xlim = (5, 250)
    ylim = (4e-5, 20)

    plot_fontsizes.update({'peak': 18})
    plot_feature_locs.update({
    #     'stamp': (0.025, 0.95),
    #     'main_legend': (0.65, 0.55)
        'stamp': (0.025, 0.97),
        'main_legend': (0.345, 0.015)
    })
    plot_fontsizes.update({'stamp': 14})

    for pair_obs, r_sub in zip(['deltaR', 'mass'],
                               [0.0, 0.3]):
        pt_feature_locs = plot_feature_locs.copy()
        if r_sub == 0:
            pt_feature_locs['main_legend'] = (0.015, 0.4)
        else:
            pt_feature_locs['main_legend'] = (0.015, 0.38)
        # Initializing plotter
        plotter = EWOCPlotter(pair_obs,
                              pt_feature_locs,
                              plot_fontsizes, None)
        ax = plotter.axes[0]

        # Plotting data
        for ptmin in [400, 500, 600]:
            file_name = f'{pair_obs}_lhc_w_ptmin{ptmin}_nomax_'\
                    f'jet0-80_subjet{r_sub:.2f}'.replace('.', '-')\
                    +'.py'
            hist1d = HistogramData(file_name=ewoc_data_dir/file_name,
                                   variable_order=['pair_obs'])
            check_normalization(hist1d)

            vals, ewoc = hist1d.centers['pair_obs'], hist1d.hist

            ax.plot(vals, ewoc,
                    label=fr'$p_{{T,\,\text{{jet}}}}\!>\,$'\
                          f'{ptmin} GeV',
                    lw=2.5)

        # Processing figure
        if pair_obs == 'mass':
            ax.text(82, 7e-2, r'$m_W$', color='firebrick', size=16)
            plotter.process_figure({**vary_pt_params,
                                    'subjet_radius': r_sub},
                                   vline_fraction=2.24)
            plotter.set_axes(pair_obs='mass',
                             xlim=(28,130), ylim=(2e-2,4e0))
        if pair_obs == 'deltaR':
            ax.text(0.255, 1.02, deltaR_text(),
                color=adjust_lightness('darkgrey'), size=16)

            plotter.process_figure({**vary_pt_params,
                                    'subjet_radius': r_sub},
                                   vline_alpha = 0)
            plotter.set_axes(pair_obs='deltaR',
                             xlim=(4e-2, 8e-1), ylim=(1.56e-1, 1.13e0))

            # Adding labels for pts
            peaks = [deltaR_val(pt) for pt in [400, 500, 600]]
            colors = ['cornflowerblue', 'firebrick', 'mediumorchid']
            for peak, color in zip(peaks, colors):
                ax.plot([peak, peak], [0, 0.95],
                        alpha=0.4, solid_capstyle='round',
                        color=color)

        # Remove scientific notation from axis ticks
        if pair_obs == 'deltaR':
            yax_ticker = ticker.StrMethodFormatter("{x:.1f}")
            ax.yaxis.set_minor_formatter(yax_ticker)
            ax.yaxis.set_major_formatter(yax_ticker)
        else:
            xax_ticker = ticker.StrMethodFormatter("{x:.0f}")
            ax.xaxis.set_minor_formatter(xax_ticker)
            ax.xaxis.set_major_formatter(xax_ticker)

        # Saving figure
        plotter.savefig(f'lhc_tuning/{pair_obs}_'+
                        f'rsub{r_sub}'.replace(".", "-")+
                        '_ptmins.pdf')
