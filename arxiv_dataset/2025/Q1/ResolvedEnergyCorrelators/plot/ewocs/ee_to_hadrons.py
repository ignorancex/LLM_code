import numpy as np
from matplotlib.lines import Line2D

from itertools import product

from histogram import HistogramData
from plotter import ewoc_data_dir

from utils.plot_utils import adjust_lightness

from qcd.qcd_basics import alpha_s, CF, CA, N_F

from ewocs.ewoc_plotter import EWOCPlotter, check_normalization
from ewocs.plot_info import EE2HADRON_PARAMS, plot_style
from ewocs.montecarlo import MonteCarloMassEWOC


plot_feature_locs = {
    'title': (0.5, 0.945),
    'process': (0.02, 0.93),
    'stamp': (0.02, 0.98),
    'peak_text': (0.5, 0.94),
    'main_legend': (0.67, 0.60),
    'line_legend': (.72, 0.45)
}
plot_fontsizes = {
    'title': 18,
    'process': 14,
    'stamp': 14,
    'peak': 18,
    'main_legend': 14,
    'line_legend': 14,
    'x_axis': 24,
    'y_axis': 28,
}

ylim, xlim =(1e-6, 1.2e1), (7e-2, 1.8e2)
level_legend_loc = (0.25, 0.015)
rsub_legend_loc = (0.60, 0.015)

# ==============================================
# Analytic Expressions
# ==============================================
def m2_ewoc_lo(m2, R_jet, r_sub, energy,
               augment_ll):
    mu = energy*R_jet/2.

    a_s = alpha_s(mu) / (4 * np.pi)

    m2jet = (energy * R_jet / 4)**2.
    m2sub = (energy * r_sub / 4)**2.

    larger_region = np.nan_to_num(np.greater(m2jet, m2)*\
        (1.-1/6.*m2/m2jet)*np.sqrt(1.-m2/m2jet))

    smaller_region = np.nan_to_num(np.greater(m2sub, m2)*\
        (1.-1/6.*m2/m2sub)*np.sqrt(1.-m2/m2sub))

    larger_region = np.nan_to_num(larger_region)

    lo_result = 3*a_s*CF*(larger_region - smaller_region)/m2

    # Additional scaling inserted by hand
    # obtained from LL computation
    if augment_ll:
        a_s = alpha_s(np.sqrt(m2)) / (4 * np.pi)

        # coefficients
        f_1 = np.sqrt((84.*CA - 125.*CF)**2.\
                      + 160.*(21.*CA-19.*CF)*N_F\
                      + 400.*N_F**2.)
        f_2 = 55.*CF - 20.*N_F

        # exponents
        kappa_1 = 2.*a_s*f_1/15.
        kappa_2 = -a_s*(84.*CA + 125.*CF + 20.*N_F + f_1)/15.

        ll_factor = \
                (m2/mu**2.)**(kappa_2/2.) * \
                (
                     (f_1 + f_2)
                     +
                     84*CA*((m2/mu**2.)**(kappa_1/2.) - 1.)\
                     +
                     (f_1-f_2)*(m2/mu**2.)**(kappa_1/2.)\
                ) / (2*f_1)
        return lo_result * ll_factor

    return lo_result


def mass_ewoc_lo(mass, R_jet, r_sub, energy, augment_ll):
    return 2*mass * m2_ewoc_lo(mass**2., R_jet, r_sub, energy,
                               augment_ll)


if __name__ == "__main__":
    pair_obs = 'mass'
    rs = [0.05, 0.1, 0.2]
    E_cm = 1000

    # ==============================================
    # Pythia
    # ==============================================
    levels = ['parton', 'hadron']
    scheme = 'WTA'

    # Initializing plotter
    plotter = EWOCPlotter('mass', plot_feature_locs,
                          plot_fontsizes, None)
    ax = plotter.axes[0]

    # Plotting data
    for level, rsub in product(levels, rs):
        # Plotting data
        file_name = f'{pair_obs}_ee_qcd_{scheme}_{level}_'\
            +f'jet0-50_subjet{rsub:.2f}'.replace('.', '-')\
            +'.py'

        # Getting data
        hist1d = HistogramData(file_name=ewoc_data_dir/file_name,
                               variable_order=['pair_obs'])

        vals, ewoc = hist1d.centers['pair_obs'], hist1d.hist

        # Plotting data
        style = plot_style(collision_type='ee', **hist1d.metadata)
        label = None

        if level == 'hadron':
            label = r'$r_\text{sub}$='+f'{rsub:.1f}'
            if rsub < .1:
                label = r'$r_\text{sub}$='+f'{rsub:.2f}'

        ax.plot(vals, ewoc, lw=2.5, label=label, **style)
        if level == 'hadron':
            ax.axvline(rsub*500/2, ymin=.62, alpha=0.4,
                       solid_capstyle='round', color=style['color'])
            ax.plot([rsub*500/2, np.sqrt(rsub*500/2*.1*500/2)],
                    [2.3e-2, 4e-3],
                    alpha=0.4, solid_capstyle='round',
                    color=style['color'])


    # Processing figure
    ax.text(0.1*500/2, 1.5e-3, r'$r_\text{sub}\sqrt{s}/2$',
            fontsize=14, color='grey', ha='center')
    plotter.process_figure(dict(**EE2HADRON_PARAMS,
                                jet_algorithm='ca',
                                jet_radius=0.5,
                                pt_min=100),
                           vline_alpha=0, legend=False)
    plotter.set_axes('mass', ylim=ylim, xlim=xlim)
    # plotter.set_axes('mass', ylim=(0, .4), xlim=xlim, yscale='linear')

    ax = plotter.axes[0]

    # Custom level legend
    pline = Line2D([0], [0], label='Parton',
                   color=adjust_lightness('darkgrey', 0.9),
                   linestyle=(0, (5, 3)))
    hline = Line2D([0], [0], label='Hadron', color='k')
    level_legend = ax.legend(loc=level_legend_loc,
                             handles=[pline, hline])

    # Subjet radius legend
    rsub_legend = ax.legend(loc=rsub_legend_loc, title_fontsize=14)
    rsub_legend.set_title('C/A subjets:')

    ax.add_artist(level_legend)

    # Saving figure
    plotter.savefig('calculation/ee2hadrons_pythia.pdf')


    # ==============================================
    # Analytic
    # ==============================================
    num_samples = 3e10

    # Adjusting style
    plot_feature_locs.update({'stamp': (0.02, 0.95)})

    # Perturbative accuracy of numerical results:
    # for accuracy in ['lo', 'pre-ll', 'post-ll', 'll', 'mll']:
    for accuracy in ['']:
        # Initializing plotter
        plotter = EWOCPlotter('mass', plot_feature_locs,
                              plot_fontsizes, None)
        ax = plotter.axes[0]
        vals = np.logspace(-2, 3, 500)

        print(f"Performing {accuracy.upper()} Calculation\n")

        for rsub in rs:
            """
            # Getting numerical data
            num_file_name = f'{pair_obs}_ewoc_{accuracy.lower()}_'\
                +f'jet0-50_subjet{rsub:.2f}'.replace('.', '-')\
                +f'_{num_samples:0.0}.py'.replace('+0', '')\
                                            .replace('+', '')

            try:
                # Loading EWOC
                hist1d = HistogramData(
                            file_name=ewoc_data_dir/num_file_name)
                vals_num = hist1d.centers['mass']
                ewoc_num = hist1d.hist
            except:
                # Setting up bins
                m_min = 0.06
                m_mid = 10.5
                m_max = 5e2

                # Number of points in each region
                N_log = 20  # Number of logarithmic points
                N_lin = 75  # Number of linear points

                # Create logarithmically spaced points from m_min to m_mid
                m_log = np.logspace(np.log10(m_min), np.log10(m_mid),
                                    N_log, endpoint=False)

                # Create linearly spaced points from m_mid to m_max
                m_lin = np.linspace(m_mid, m_max, N_lin)

                # Combine the grids
                m_edges = np.concatenate([m_log, m_lin])
                m_centers = np.concatenate([np.sqrt(m_log[1:]*m_log[:-1]),
                                            [np.sqrt(m_log[-1]*m_lin[0])],
                                            (m_lin[1:]+m_lin[:-1])/2])

                # via Monte Carlo integration
                integrator = MonteCarloMassEWOC(initial_flavor=0,
                                         E_cm=E_cm,
                                         R_jet=0.5, r_sub=rsub,
                                         epsilon=1e-10,
                                         obs_bins=m_edges,
                                         obs_centers=m_centers,
                                         accuracy=accuracy)

                integrator.integrate(num_samples=num_samples)
                integrator.save()

                # Loading EWOC
                hist1d = HistogramData(file_name=ewoc_data_dir/num_file_name)
                vals_num, ewoc_num = hist1d.centers['mass'], hist1d.hist
            """

            # Plotting
            style = plot_style(collision_type='ee', sub_rad=rsub)

            # LO Analytic result
            ewoc_lo = mass_ewoc_lo(vals, 0.5, rsub, 1000, False)
            ax.plot(vals, vals*np.log(10)*ewoc_lo, lw=2.5,
                    label=rf'$r_\text{{sub}}={rsub}$',
                    **style)
                    # **plot_style(collision_type='ee',
                    #              sub_rad=rsub, level='LO'))

            # # Numerical
            # ax.plot(vals_num, vals_num*np.log(10)*ewoc_num, lw=2.5,
            #         label=rf'$r_\text{{sub}}={rsub}$',
            #         **style)

            # Adding labels for rsub
            ax.axvline(rsub*500/2, ymin=.62, alpha=0.4,
                       solid_capstyle='round', color=style['color'])
            ax.plot([rsub*500/2, np.sqrt(rsub*500/2*.1*500/2)],
                    [2.3e-2, 4e-3],
                    alpha=0.4, solid_capstyle='round',
                    color=style['color'])


        plotter.set_axes('mass', ylim=ylim, xlim=xlim)
        # plotter.set_axes('mass', ylim=(0, .4), xlim=xlim, yscale='linear')

        # # Level radius legend
        # lo_line = Line2D([0], [0], label='LO',
        #                  color=adjust_lightness('darkgrey', 0.9),
        #                  linestyle=(0, (2, 1.3)))
        # ll_line = Line2D([0], [0], label=accuracy.upper(), color='k')
        # level_legend = ax.legend(loc=(level_legend_loc[0]+0.09,
        #                               level_legend_loc[1]+0.01),
        #                          fontsize=14,
        #                          handles=[lo_line, ll_line])

        # Add text
        plotter.process_figure(dict(**EE2HADRON_PARAMS,
                                    shower_model='Fixed-order prediction',
                                    jet_algorithm='theory',
                                    jet_radius=0.5,
                                    subjet_algorithm=None),
                               vline_alpha=0, legend=False)

        ax.text(0.1*500/2, 1.5e-3, r'$r_\text{sub}\sqrt{s}/2$',
                fontsize=14, color='grey', ha='center')

        # Subjet radius legend
        rsub_legend = ax.legend(loc=rsub_legend_loc)
        # ax.add_artist(level_legend)

        # Saving figure
        plotter.savefig('calculation/ee2hadrons_analytic_'
                        f'{accuracy.lower()}.pdf')
