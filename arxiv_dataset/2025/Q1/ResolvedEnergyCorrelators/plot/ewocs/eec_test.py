import numpy as np
from matplotlib.lines import Line2D

from itertools import product

from histogram import HistogramData
from plotter import ewoc_data_dir

from utils.plot_utils import adjust_lightness

from qcd.qcd_basics import alpha_s, CF, CA, N_F

from ewocs.ewoc_plotter import EWOCPlotter, check_normalization
from ewocs.plot_info import EE2HADRON_PARAMS, plot_style
from ewocs.analytic import subjet_eec
from ewocs.montecarlo import MonteCarloEEC


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

ylim, xlim =(1e-3, 1e1), (6e-4, 1.1e0)
level_legend_loc = (0.68, 0.81)

# ==============================================
# Analytic Expressions
# ==============================================
# TODO: Check factor in front of splitting function
def eec_analytic(theta, R_jet, energy, accuracy):
    mu = R_jet*energy/2
    if accuracy.upper() == 'LO':
        # TODO: Check
        return 2*alpha_s(mu)/(np.pi*theta) \
            * np.heaviside(R_jet - theta, 0)
    result = subjet_eec(var=theta, mu=mu,
                        R_jet=None if accuracy.upper()=='LO' else R_jet,
                        r_sub=0, accuracy=accuracy, var_name='theta')
    return result * np.heaviside(R_jet - theta, 0)


if __name__ == "__main__":
    # Transition region
    transition_low, transition_high = 1/250, 5/250
    delta = .0036  # line width for transition region in plot

    """
    # ==============================================
    # Pythia
    # ==============================================
    # Initializing plotter
    plotter = EWOCPlotter('theta', plot_feature_locs,
                          plot_fontsizes, None)
    ax = plotter.axes[0]

    # Plotting data
    levels = ['parton', 'hadron']
    scheme = 'WTA'
    for level in levels:
        # Plotting data
        file_name = f'theta_ee_qcd_{scheme}_{level}_'\
            +f'jet0-50_subjet0-00.py'

        # Getting data
        hist1d = HistogramData(file_name=ewoc_data_dir/file_name,
                               variable_order=['pair_obs'])

        vals, eec = hist1d.centers['pair_obs'], hist1d.hist

        # Plotting data
        label = None
        ax.plot(vals, eec, lw=2.5,
                color='k' if level=='hadron' else 'grey',
                ls='solid' if level=='hadron' else (0, (5,2)),
                zorder=1 if level=='hadron' else 2)

    # Processing figure
    plotter.process_figure(dict(**EE2HADRON_PARAMS,
                                jet_algorithm='ca',
                                jet_radius=0.5,
                                pt_min=100),
                           vline_alpha=0.0, legend=False)
    plotter.set_axes('theta', ylim=ylim, xlim=xlim)
    ax = plotter.axes[0]
    # Labelling transition region
    ax.text(.375, .09, r'Transition', ha='center',
            color='mediumorchid', size=13,
            transform=ax.transAxes)
    ax.text(.375, .04, r'Region', ha='center',
            color='mediumorchid', size=13,
            transform=ax.transAxes)
    rec = ax.axvspan(transition_low, transition_high,
                     alpha=0.3, color='mediumorchid',
                     ymax=.7, zorder=0)
    rec.set_edgecolor(None)
    rec = ax.axvspan(transition_low, transition_high,
                     alpha=0.07, color='mediumorchid',
                     ymin=.7, zorder=0)
    rec.set_edgecolor(None)
    # Boundaries
    ax.axvline(transition_low, ymax=.7-delta, alpha=0.3,
               color='mediumorchid')
    ax.axvline(transition_high, ymax=.7-delta, alpha=0.3,
               color='mediumorchid')
    ax.axvline(transition_low, ymin=.7+delta, alpha=0.07,
               color='mediumorchid')
    ax.axvline(transition_high, ymin=.7+delta, alpha=0.07,
               color='mediumorchid')

    # Custom level legend
    pline = Line2D([0], [0], label='Parton',
                   color=adjust_lightness('darkgrey', 0.9),
                   linestyle=(0, (5, 3)))
    hline = Line2D([0], [0], label='Hadron', color='k')
    level_legend = ax.legend(loc=level_legend_loc,
                             handles=[pline, hline])

    # Saving figure
    plotter.savefig('tests/eec_test_pythia.pdf')
    """


    # ==============================================
    # Analytic
    # ==============================================
    vals_a = np.logspace(-5, 0, 500)

    for accuracy in ['ll']:
        accuracy = accuracy.upper()
        # Initializing plotter
        plotter = EWOCPlotter('theta', plot_feature_locs,
                              plot_fontsizes, None)
        ax = plotter.axes[0]

        # Getting analytic prediction
        accuracy_a = 'll' if accuracy.lower().endswith('ll')\
                     else accuracy
        eec_a = eec_analytic(vals_a, 0.5, 1000, accuracy_a)

        # Plotting analytic result
        ax.plot(vals_a, vals_a*np.log(10)*eec_a, lw=2.5,
                color='grey', ls=(0,(2,1.3)), zorder=2)

        # Getting numerical data
        num_file_name = f'theta_{accuracy.lower()}ewoc_'\
            +'jet0-50_subjet0-00.py'
        try:
            # Loading EEC
            hist1d = HistogramData(file_name=ewoc_data_dir/num_file_name)
            vals_n, eec_n = hist1d.centers['theta'], hist1d.hist
        except:
            E_cm = 1000

            # for the following bins
            edges = np.logspace(-5, np.log10(.6), 80)
            centers = np.sqrt(edges[:-1]*edges[1:])
            # via Monte Carlo integration
            integrator = MonteCarloEEC(initial_flavor=0,
                                     E_cm=E_cm,
                                     R_jet=0.5, r_sub=0,
                                     epsilon=1e-5,
                                     obs_bins=edges,
                                     obs_centers=centers,
                                     accuracy=accuracy)
            integrator.integrate(num_samples=1e8)
            integrator.save()

            # Loading EWOC
            hist1d = HistogramData(file_name=ewoc_data_dir/num_file_name)
            vals_n, eec_n = hist1d.centers['theta'], hist1d.hist

        # Plotting numerical
        ax.plot(vals_n, vals_n*np.log(10)*eec_n, lw=2.5,
                color='k', ls='solid', zorder=1)

        # Post-processing axes
        plotter.set_axes('theta', ylim=ylim, xlim=xlim)
        plotter.process_figure(dict(**EE2HADRON_PARAMS,
                                    shower_model=accuracy.upper()+\
                                                 ' prediction',
                                    jet_algorithm='theory',
                                    jet_radius=0.5,
                                    subjet_algorithm=None),
                               vline_alpha=0, legend=False)
        # Level legend
        a_line = Line2D([0], [0], label='Analytic',
                         color=adjust_lightness('darkgrey', 0.9),
                         linestyle=(0, (2, 1.3)))
        n_line = Line2D([0], [0], label='Numerical', color='k')
        level_legend = ax.legend(loc=level_legend_loc,
                                 fontsize=14,
                                 handles=[a_line, n_line])
        # Labelling transition region
        ax.text(.375, .09, r'Transition', ha='center',
                color='mediumorchid', size=13,
                transform=ax.transAxes)
        ax.text(.375, .04, r'Region', ha='center',
                color='mediumorchid', size=13,
                transform=ax.transAxes)
        rec = ax.axvspan(transition_low, transition_high,
                         alpha=0.3, color='mediumorchid',
                         ymax=.77, zorder=0)
        rec.set_edgecolor(None)
        rec = ax.axvspan(transition_low, transition_high,
                         alpha=0.07, color='mediumorchid',
                         ymin=.77, zorder=0)
        rec.set_edgecolor(None)
        ax.axvline(transition_low, ymax=.77-delta, alpha=0.3,
                   color='mediumorchid', zorder=0)
        ax.axvline(transition_high, ymax=.77-delta, alpha=0.3,
                   color='mediumorchid', zorder=0)
        ax.axvline(transition_low, ymin=.77+delta, alpha=0.07,
                   color='mediumorchid', zorder=0)
        ax.axvline(transition_high, ymin=.77+delta, alpha=0.07,
                   color='mediumorchid', zorder=0)

        # Saving figure
        plotter.savefig(f'tests/eec_test_{accuracy.lower()}.pdf')
