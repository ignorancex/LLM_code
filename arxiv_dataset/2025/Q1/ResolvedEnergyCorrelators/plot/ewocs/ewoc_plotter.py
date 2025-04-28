import time
from pathlib import Path
import warnings

import numpy as np
import numpy.ma as ma

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
import seaborn as sns

from cycler import cycler

from scipy.optimize import curve_fit

# Local imports
from plotter import Plotter, LOGGER
from plotter import ewoc_figure_dir
from ewocs.plot_info import PLOT_DATA, PLOT_RATIO, \
    PLOT_ANALYTIC, DEFAULT_ANALYTIC_ACCURACY, \
    xlabel_latex, diff_ewoc_latex


from utils.plot_utils import get_colors_colorbar
from utils.plot_utils import adjust_lightness
from utils.plot_utils import stamp
from utils.plot_utils import get_full_width
from utils.gen_utils import str_to_bool

# EWOC specific plot utilities
from ewocs.plot_info import lims, \
    obs_title

from qcd.qcd_basics import M_W

from qcd.observable_utils import \
    expected_peaks_scales_colors, \
    pt_fudge, pt_val_from_min, \
    mask_obs_by_region, region_bounds


# ####################################
# Plotting flags
# ####################################
ANALYTIC_LINEWIDTH = 2.5
ANALYTIC_LINESTYLE = 'dashed'
ANALYTIC_COLOR = 'darkgrey'

# TOGGLE: Whether to plot jet mass when plotting pairwise masses
PLOT_JET_MASS = False

# TOGGLE: Different types of power law fits for ``difference'' plots
POWERLAW_TYPE = 2

# Lower y boundary in log-log plot
LOWER_LOGY_BOUNDARY = 2e-4
UPPER_YLOG_BOUNDARY = 2e1


# ####################################
# Misc. Plot Options
# ####################################
# Plot types
PLOT_TYPES = [
    "Default",
    "Money Style",
    "Fix Option Color",
    "Angularity",
    "Black Lines", "Pastel",
    "logy",
    "Linear",
    "Hadron-Parton",
    "MPI-Hadron",
    "Pythia-Analytic",
    "Verbose",
]

# Whether to highlight different regions (non-pert., resummed)
HIGHLIGHT_REGIONS = False

# See https://arxiv.org/pdf/2307.15739#page=21,
# figure 7 is the first example I found/the inspiration
# for this plot style
DIFFERENCE_PLOTS = ["Hadron-Parton", "MPI-Hadron", "Pythia-Analytic"]

# Plot features
plot_fontsizes = {
    'title': 20,
    'process': 16,
    'stamp': 14,
    'peak': 22,
    'main_legend': 13,
    'line_legend': 13,
    'x_axis': 28,
    'y_axis': 28,
}

plot_feature_locs = {
    'title': (0.5, 0.935),
    'process': (0.02, 0.92),
    'stamp': (0.02, 0.91),
    'peak_text': (0.02, 0.93),
    'main_legend': (0.03, 0.4),
    'line_legend': (.72, 0.45)
}


"""
plot_fontsizes = {
    plot_type: {
        'title': 20,
        'process': 16,
        'stamp': 14,
        'peak': 22,
        'main_legend': 13,
        'line_legend': 13,
        'x_axis': 28,
        'y_axis': 28,
    }
    for plot_type in PLOT_TYPES
}

plot_fontsizes["Money Style"] = {
    'title': 18,
    'process': 14,
    'stamp': 12,
    'peak': 20,
    # 'main_legend': 12,  # TOGGLE: For paper
    'main_legend': 14,  # TOGGLE: For pres
    # 'line_legend': 9,  # TOGGLE: For paper
    'line_legend': 12,  # TOGGLE: For pres
    'x_axis': 24,
    'y_axis': 28,
}

# Locations of plot features
plot_feature_locs = {
    plot_type: {
        'title': (0.5, 0.935),
        'process': (0.02, 0.92),
        'stamp': (0.02, 0.91),
        'peak_text': (0.02, 0.93),
        'main_legend': (0.03, 0.4),
        'line_legend': (.72, 0.45)
    }
    for plot_type in PLOT_TYPES
}

plot_feature_locs["Money Style"] = {
    'title': (0.5, 0.945),
    'process': (0.02, 0.93),
    'stamp': (0.02, 0.93),
    'peak_text': (0.02, 0.94),
    'main_legend': (0.67, 0.60),
    # 'line_legend': (.72, 0.45) # TOGGLE: For paper
    'line_legend': (.15, 0.05)   # TOGGLE: For pres
}


plot_feature_locs["Black Lines"]['peak_text'] = \
    (0.3, 0.82)
plot_feature_locs["Black Lines"]['stamp'] = \
    (0.025, 0.6)

plot_feature_locs["Pastel"]['main_legend'] = (0.7, 0.455)
plot_feature_locs["Pastel"]['line_legend'] = (0.7, 0.27)
# TOGGLE: e+e- analytic
plot_feature_locs["Pastel"]['main_legend'] = (0.45, 0.1)
plot_feature_locs["Pastel"]['line_legend'] = (0.75, 0.63)

plot_feature_locs["Angularity"]['main_legend'] = (0.48, 0.1)
plot_feature_locs["Angularity"]['line_legend'] = (0.75, 0.63)


for difference_type in DIFFERENCE_PLOTS:
    plot_feature_locs[difference_type] = {
        'title': (0.5, 0.935),
        'process': (0.02, 0.92),
        'stamp': (0.02, 0.91),
        'peak_text': (0.02, 0.93),
        'main_legend': (0.03, 0.4),
        'line_legend': (.6, 0.45)
    }

plot_feature_locs["Verbose"]['main_legend'] = "center left"
plot_feature_locs["Verbose"]['line_legend'] = "center right"
"""

# Lines and Labels
# for regular plots
line_label = {'dotted':  'Parton',
              'dashdot': 'Hadron',
              'solid':   'Hadron + MPI',
              'dashed':  'Analytic'}
line_label[':'] = line_label['dotted']
line_label['-.'] = line_label['dashdot']
line_label['-'] = line_label['solid']
line_label['--'] = line_label['dashed']

# for ``difference'' plots
difference_line_label = {
    'solid': {
        'Hadron-Parton':   r"Hadron$-$Parton",
        'MPI-Hadron':      r"MPI$-$Hadron",
        'Pythia-Analytic': r"Pythia$-$Analytic"
    }
}

difference_line_label['-'] = difference_line_label['solid']


def ls_key(linestyle):
    linestyle = linestyle[1]
    if line_label[linestyle] == 'Parton':
        return 0
    elif line_label[linestyle] == 'Hadron':
        return 1
    elif line_label[linestyle] == 'Hadron + MPI':
        return 2
    elif line_label[linestyle] == 'Analytic':
        return 3
    raise ValueError(f'Unknown {linestyle=}')


# --------------------------------
# Check normalization for histograms
# --------------------------------
def check_normalization(hist):
    # If we are dealing with an EnC which should be normalized,
    # check for normalization
    hist.integrate_histogram(scheme='log10', outflow_weight=1)

    try:
        weight = hist.metadata['weight']

        if not hasattr(weight, '__iter__'):
            if weight != 1:
                return True
        else:
            weight=np.array(weight)
            if all(weight == 1):
                return True
    except:
        pass


    if not np.isclose(hist.integral, 1.0, rtol=5e-2):
        warnings.warn(f"EWOC does not integrate to 1 "
                      f"({hist.integral=})")
        return False


# --------------------------------
# Color Pre-Sets
# --------------------------------
BASIC_COLORS = {float(0.0):  "#0571b0",
                float(0.03): "#b8e186",
                float(0.1):  "#a6611a",
                float(0.3):  "#fdb863",
                float(0.6):  "#ca0020",}
Z_CUT_COLORS = {}
WEIGHT_COLORS = {}
Z_CUT_WEIGHT_COLORS = {}


def COLOR_PRESET(plot_type, **parameters):
    """Preset colors for preset plot types."""
    obs = parameters['pair_obs']
    subrad = parameters['subjet_radius']
    z_cut = parameters['z_cut']
    weight_1 = parameters['weight_1']
    weight_2 = parameters['weight_2']

    # Configuring parameters to be either floats or NoneType
    if subrad not in [None, 'None']:
        subrad = float(subrad)
    else:
        subrad = None
    if z_cut not in [None, 'None']:
        z_cut = float(z_cut)
    else:
        z_cut = None
    if weight_1 not in [None, 'None']:
        weight_1 = float(weight_1)
    else:
        weight_1 = None
    if weight_2 not in [None, 'None']:
        weight_2 = float(weight_2)
    else:
        weight_2 = None

    # Black lines
    if plot_type == "Black Lines":
        return 'black'

    # Angularity Plot
    if plot_type == "Angularity":
        if obs == 'e1':
            return "#ABDEE6"
        if obs == 'e2':
            return "#CBAACB"
        if obs == 'e4':
            return "#FFCCB6"
        if obs == 'e6':
            return "#F3B0C3"
        return None

    # Pastel cycler handled by axes
    if plot_type == "Pastel":
        return None

    # Non-"option color fixed" plot type
    if plot_type != "Fix Option Color" or \
            (z_cut == 0 and weight_1 == 1 and weight_2 == 1) or \
            (z_cut is None and weight_1 is None and weight_2 is None):
        # Money plot color
        if plot_type == "Money Style" and \
                subrad in [0.0, 0.03, 0.3, 0.6]:
            if subrad == 0.0:
                return 'lightskyblue'
            if subrad == 0.03:
                return 'palegreen'
            if subrad == 0.3:
                return 'maroon'
            if subrad == 0.6:
                return 'mediumpurple'

        # Jet observables in gray  # TOGGLE
        if subrad is None:
            return 'gray'

        # Otherwise, use a default color
        return BASIC_COLORS.setdefault(subrad, '#49bb26')

    # Other option is plot_type == "Fix Option Color"
    # and the weights or zcut are non-standard
    if z_cut == 0.1 and weight_1 == 1 and weight_2 == 1:
        # return '#d01c8b'
        # return "#f02c2c"
        return Z_CUT_COLORS.setdefault(subrad, "#df617f")
    if z_cut == 0.0 and weight_1 == 2 and weight_2 == 2:
        return WEIGHT_COLORS.setdefault(subrad, "#8a9747")
    if z_cut == 0.1 and weight_1 == 2 and weight_2 == 2:
        # return "#e66101"
        # return "#763053"
        # "#bb2693"
        return Z_CUT_WEIGHT_COLORS.setdefault(subrad, "#633075")

    return None

# ====================================
# Code for processing EWOC Data
# ====================================
def partonic_energy(fixed_param_values):
    energy = fixed_param_values.get('energy')
    pid_1 = int(fixed_param_values.get('PID_1'))
    pid_2 = int(fixed_param_values.get('PID_2'))
    for pid in [pid_1, pid_2]:
        if pid == 2212:
            energy /= np.sqrt(13)
    return energy



# =====================================
# EWOC Plotter Class
# =====================================
class EWOCPlotter(Plotter):
    def __init__(self, pair_obs,
                 feature_locs,
                 feature_fontsizes,
                 color_preset,
                 **kwargs):
        """Initializes the HistPlotter."""
        # Configuring the observable to be plotted,
        if pair_obs is None:
            raise ValueError("EWOCPlotter: "
                             "Must specify `pair_obs`.")
        # even as a list if relevant
        if isinstance(pair_obs, str):
            if pair_obs[0] == '[' and pair_obs[-1] == ']':
                # If it is a string that looks like a list
                pair_obs = pair_obs[1:-1].replace('\'', '').\
                    replace('\"', '').\
                    replace(' ', '').split(',')
            elif ' ' in pair_obs:
                pair_obs = pair_obs.split(' ')
            elif ', ' in pair_obs:
                pair_obs = pair_obs.split(', ')
            elif ',' in pair_obs:
                pair_obs = pair_obs.split(',')

        self.pair_obs = pair_obs
        self.obs_key = pair_obs

        if isinstance(pair_obs, list):
            if all(obs[:4] == 'mass' or obs == 'jet_mass'
                   for obs in pair_obs):
                self.obs_key = 'mass'
            elif all(obs[:2] == 'm2' for obs in pair_obs):
                self.obs_key = 'm2'
            elif all(len(obs) == 2 and obs[0] == 'e'
                     for obs in pair_obs):
                self.obs_key = 'angularity'
            else:
                raise ValueError("EWOCPlotter: "
                                 f"Unsupported value for {pair_obs=}.")
        elif pair_obs.startswith('mass'):
            self.obs_key = 'mass'
        elif pair_obs.startswith('m2'):
            self.obs_key = 'm2'
        elif pair_obs[0] == 'e':
            self.obs_key = 'angularity'


        # --------------------------------
        # Color presets for this plot
        # --------------------------------
        self.color_preset = kwargs.pop('color_preset',
                                       None)

        # --------------------------------
        # Subjet algorithm string
        # --------------------------------
        subjet_algorithm = kwargs.pop('subjet_algorithm',
                                      None)
        if subjet_algorithm is not None:
            if subjet_algorithm.startswith('ee_'):
                subjet_algorithm = subjet_algorithm[3:]
            if subjet_algorithm == 'antikt':
                subjet_algorithm = 'akt'
            if subjet_algorithm.lower() == "c/a":
                subjet_algorithm = "ca"
            subjet_alg_str = f"{subjet_algorithm}".lower()\
                                .replace("_", "-")
        else:
            subjet_alg_str = ''

        self.subjet_alg_str = subjet_alg_str


        # --------------------------------
        # Setting up behavior of analytic function
        # --------------------------------
        # Function to which data will be compared (if it exists)
        self.analytic_function = kwargs.pop('analytic_function',
                                            None)
        self.plotted_analytic_params = []

        # --------------------------------
        # Setting up plot style and defaults
        # --------------------------------
        self.feature_locs = feature_locs
        self.feature_fontsizes = feature_fontsizes

        # Additional information for color scheme
        num_colors = kwargs.pop('num_colors', None)
        if num_colors is not None:
            colors = sns.color_palette(palette, num_colors).as_hex()
            kwargs.update({'axes.prop_cycle': colors})

        # Initializing
        self.logger = kwargs.pop('logger', LOGGER)
        kwargs['ratio_plot'] = kwargs.get('ratio_plot', False)

        kwargs['xtick.labelsize'] = 13.5
        kwargs['ytick.labelsize'] = 13.5
        super().__init__(**kwargs)

        self.logger.debug("Initialized EWOCPlotter:\n\t"
                   f"{self.metadata = }\n")


    def to_analytic_params(self, params):
        """Method to discern which parameters have been
        used to produce analytic plots."""
        ignored_params = ['n_events', 'shower_model',
                          'isr', 'fsr', 'mpi', 's_channel',
                          'level', 'pt_min', 'pt_max',
                          'jet_algorithm', 'subjet_algorithm',
                          'left_bin', 'right_bin']
        analytic_params = params.copy()
        for key in ignored_params:
            analytic_params.pop(key)

        return analytic_params


    # =====================================
    # Difference Plot Sketch
    # =====================================
    """
    # Difference plots I

    if self.plot_type == "Pythia-Analytic":
        # new y values = Pythia - Analytic
        ys = ys-analytic
        style.update({'linestyle': 'solid'})

    else:
        # Need to get the two datasets to match up
        if not (len(xs) == len(assoc_xs)):
            make_interpolation = True
        elif not (xs == assoc_xs).all():
            make_interpolation = True

        if make_interpolation:
            assoc_ys = np.interp(xs,  assoc_xs, assoc_ys)

        # Finding difference
        ys = ys - assoc_ys

        # Potentially modifying hist data based on plot type
        if self.plot_type == "Hadron-Parton":
            assert parameters['level'] == 'hadron' and \
                str_to_bool(parameters['mpi']) == False
            style.update({'linestyle': 'solid'})

        elif self.plot_type == "MPI-Hadron":
            assert parameters['level'] == 'hadron' and \
                str_to_bool(parameters['mpi']) == False
            style.update({'linestyle': 'solid'})
    """
    """
    # Then, can imagine doing
    self.plot_data((xs, ys), parameters, **kwargs)
    """
    """
    # Difference plots II
    # Showing a power law fit for non-pert. corrections, etc.
    if self.plot_type in DIFFERENCE_PLOTS and PLOT_DATA:
        # TOGGLE: Different types of power law fits
        if POWERLAW_TYPE == 0:
            param_names = ['Lambda0', 'Lambda1', 'm']
            def powerlaw(x, Lambda0, Lambda1, m):
                return Lambda0 + Lambda1*x**m

        if POWERLAW_TYPE == 1:
            param_names = ['Lambda0', 'Lambda1', 'm1',
                           'Lambda2', 'm2']
            def powerlaw(x, Lambda0, Lambda1, m1,
                         Lambda2, m2):
                return Lambda0 + Lambda1*x**m1 + Lambda2*x**m2

        if POWERLAW_TYPE in [2, 'renormalon']:
            param_names = ['Lambda1']
            def powerlaw(x, Lambda1):
                return Lambda1/x**1.5

        if POWERLAW_TYPE in [3, 'renormalon+']:
            param_names = ['Lambda1', 'Lambda2']
            def powerlaw(x, Lambda1, Lambda2):
                return Lambda1/x**1.5 + Lambda2/x

        # Restrict the x-values for the fit to lie within
        # the resummed region
        fit_mask = mask_obs_by_region(self.pair_obs, xs,
                                      'resummed',
                                      **parameters)
        fit_xs = ma.masked_array(xs, mask=fit_mask)
        fit_ys = ma.masked_array(ys, mask=fit_mask)
        pfit_params, _= curve_fit(powerlaw, fit_xs, fit_ys)

        # Print out the power-law fit parameters
        pfit_message = ''
        for pair in zip(param_names, pfit_params):
            pfit_message += "\t" + str(pair) + "\n"
        self.logger.info("# - - - - - - - - - - - - - - - ")
        self.logger.info("# Power Law Fit:")
        self.logger.info("# - - - - - - - - - - - - - - - ")
        self.logger.info(pfit_message + "\n")

        pfit_ys = powerlaw(xs, *pfit_params)
        self.axes[0].plot(xs, pfit_ys, **analytic_style)

        if self.ratio_plot:
            self.axes[1].plot(xs, np.ones_like(xs),
                         **analytic_style)
            self.axes[1].plot(xs, ys/pfit_ys, **style)
    """


    # =====================================
    # Plotting
    # =====================================
    def plot_data(self, data, parameters,
                  **kwargs):
        """Plots given EWOC data."""
        # ---------------------------------
        # Setup
        # ---------------------------------
        # TODO: setup with filename or module
        # TODO: get parameters from file
        # TODO: Get data from file
        xs, ys = data

        # Not plotting certain datasets
        if parameters['level'] == 'parton' and \
                parameters['mpi'] in ['1', 1, 'True', True]:
            self.logger.warning("EWOCPlotter: "
                           +"Not plotting parton-level "
                           +"EWOC with mpi=True.")
            return

        if parameters['weight_1'] != parameters['weight_2']:
            self.logger.warning("EWOCPlotter: "
                           +"Not plotting EWOC with different "
                           +"energy weights.")
            return

        # - - - - - - - - - - - - - - - - -
        # Plot Setup
        # - - - - - - - - - - - - - - - - -
        # Analytic function
        analytic_function = kwargs.pop('analytic_function',
                                       self.analytic_function)

        # Histogram weights
        hist_weight = kwargs.pop('hist_weight', None)
        if hist_weight is None:
            # Default: no additional weight
            def hist_weight(xs):
                return np.ones(len(xs))

        # - - - - - - - - - - - - - - - - -
        # Histogram Style
        # - - - - - - - - - - - - - - - - -
        # Getting colors
        color = None
        if self.color_preset is not None:
            color = self.color_preset(**parameters)
        if color is None:
            color = next(self._mpl_rc['axes.prop_cycle'])

        # Setting up for legend
        legend_entry = self.legend_prescription(parameters, axes=self.axes[0])

        # Setting up the style
        style = {'linewidth':  self.mpl_rc['lines.linewidth']}
        style.update(kwargs)

        # LOGGER messages
        self.logger.debug("\tPlotting new dataset")
        self.logger.debug(f"\t\t{parameters=}")
        self.logger.debug(f"\t\t{color=}\n")

        # - - - - - - - - - - - - - - - - -
        # Analytic Function Setup:
        # - - - - - - - - - - - - - - - - -
        ratio_plot = self.ratio_plot

        if analytic_function is None:
            ratio_plot = False

        # Analytic curve style
        analytic_style = dict(style,
                              linewidth=ANALYTIC_LINEWIDTH,
                              linestyle=ANALYTIC_LINESTYLE,
                              color=ANALYTIC_COLOR,
                             )


        # Lighter color for analytic plots
        if color is not None:
            # analytic_color = color
            analytic_color = adjust_lightness(color, 1.25)
            if isinstance(analytic_color, tuple):
                # ensuring the color is valid
                analytic_color = tuple(min(c, 1.) for c in analytic_color)

            analytic_style.update({'color': analytic_color})

        # - - - - - - - - - - - - - - - - -
        # Parameter-dependent styles
        # - - - - - - - - - - - - - - - - -
        # Parton level style
        if parameters['level'] == 'parton':
            # Medium color dotted line
            style.update({'linestyle': 'dotted'})

        # Hadron level style
        elif parameters['level'] == 'hadron':
            # Darker color solid line
            if parameters['mpi']:
                style.update({'linestyle': 'solid'})
                color = adjust_lightness(color, 0.7)
            else:
                style.update({'linestyle': 'dashdot'})
                color = adjust_lightness(color, 0.85)

        # Universal color control
        if isinstance(color, tuple):
            # ensuring the color is valid -- no rbg > 1
            color = tuple(min(c, 1.) for c in color)
        style.update({'color': color})

        # - - - - - - - - - - - - - - - - -
        # Getting stored histogram data
        # - - - - - - - - - - - - - - - - -
        ys = ys * hist_weight(xs)
        if self.metadata['x_scale'] == 'log':
            ys = ys * xs

        # TOGGLE: Printing out numerical histogram data
        self.logger.log(5, "Histogram data:")
        self.logger.log(5, f"Numerical EWOC:  {ys}")
        self.logger.log(5, f"Numerical xs: {xs}")

        # - - - - - - - - - - - - - - - - -
        # Getting analytic function
        # - - - - - - - - - - - - - - - - -
        # Here, I get and (usually) plot the analytic function.
        # Doing this before plotting the actual data in case
        # I want to do ``Pythia-Analytic'' difference plots.
        analytic_ys = None
        if analytic_function is not None:
            # - - - - - - - - - - - - - - - - -
            # Getting processing analytic curve
            # - - - - - - - - - - - - - - - - -
            self.logger.debug("Plotting analytic function")
            analytic_ys= analytic_function(xs, **parameters)
            if analytic_ys is None:
                self.logger.debug("analytic_function given to plotter "
                                  "returned None.")

        if analytic_ys is not None:
            analytic_ys= analytic_ys* hist_weight(xs)

            if self.metadata['x_scale'] == 'log':
                analytic_ys= analytic_ys*xs

            # - - - - - - - - - - - - - - - - -
            # Plotting analytic curve
            # - - - - - - - - - - - - - - - - -
            # Avoiding redundant analytic plots
            a_params = self.to_analytic_params(parameters)
            if PLOT_ANALYTIC and a_params not in self.plotted_analytic_params:
                # Plot on main axes
                self.axes[0].plot(xs, analytic_ys, **analytic_style)
                self.plotted_analytic_params.append(a_params)

                # Plot on ratio axes
                if ratio_plot and PLOT_DATA:
                    ratio_analytic_style = analytic_style.copy()

                    # Plotting analytic in grey if
                    # there are multiple analytic curves
                    if len(self.plotted_analytic_params) > 1:
                        ratio_analytic_style['color'] = 'grey'

                    self.axes[1].plot(xs, np.ones(len(xs)),
                                 **ratio_analytic_style,
                                 zorder=0.5)

                # Plotting the ratio for _all_ plots
                if ratio_plot and PLOT_DATA:
                    self.axes[1].plot(xs, ys/analytic_ys,
                                 **style)
        else:
            self.logger.debug("\nNo analytic function to plot!\n")


        # Making plot!
        if PLOT_DATA:
            self.axes[0].plot(xs, ys, label=legend_entry, **style)
        return


    def highlight_regions(parameters, alpha=0.5):
        """Highlighting different regions within the plot
        in which different physics descriptions take over.
        """
        nonpert_bnds = region_bounds(self.pair_obs,
                                     'non-perturbative',
                                     **parameters)
        resummed_bnds = region_bounds(self.pair_obs,
                                     'resummed',
                                     **parameters)

        [ax.axvspan(*nonpert_bnds,
                    color='darkmagenta', alpha=alpha)
         for ax in self.axes]
        [ax.axvspan(*resummed_bnds,
                    color='cornflowerblue', alpha=alpha)
         for ax in self.axes]

        return


    # =====================================
    # Plotting scratch code/reminders
    # =====================================
    """
        # Distinguished plotstyle
        if color == "maroon":
            style['linewidth'] *= 1.4
    """
    """
        add_jet_mass = (
            self.plot_type in ['Default', 'Money Style']
                and
            len(catalogs) == 1
        )

        if (isinstance(parameters['pair_obs'], list)):
            add_jet_mass = add_jet_mass and (
                all([obs.startswith('mass')
                     for obs in parameters['pair_obs']])
            )

        else:
            add_jet_mass = add_jet_mass and (
                parameters['pair_obs'].startswith('mass')
            )

        if add_jet_mass:
            for param_set in distinct_parameters(parameters):
                jet_mass_params = param_set.copy()
                jet_mass_params.update({
                    'pair_obs': 'jet_mass',
                    'z_cut': None, 'weight_1': None, 'weight_2': None,
                    'subjet_radius': None, 'subjet_algorithm': None,
                    'left_bin': 'default', 'right_bin': 'default',
                    'nbins': 100
                })

                tmp_kwargs = kwargs.copy()
                if fig_kwargs is not None:
                    tmp_kwargs.update(
                        fig_kwargs(data_label='hist dict',
                                   params=jet_mass_params))

                file_path = catalogs[0].get_filename('hist dict',
                                                    jet_mass_params)

                data = np.load(file_path, allow_pickle=True)

                self.plot_data(data, analytic_function=None,
                               fig=fig, axes=axes,
                               parameters=jet_mass_params)
        # Saving figure
        if fig_catalog is not None:
            self.logger.debug("\n\t\tSaving figure"
                  f" to {fig_catalog.name()}")
            # Saving figure
            fig_catalog.savefig(fig, fig_label,
                                parameters,
                                nested_folder=fig_label)

        # Closing figure
        plt.close(fig)

        """

    # =====================================
    # Post-processing
    # =====================================
    def ewoc_stamp(self, fixed_param_values: dict,
                   show_w_guess: bool = False,
                   fwhm_symbol: str = None,
                   show_level:   bool = False ,
                   difference:   bool = False):
        """Adds a stamp to the given axes associated
        with the parameters that are fixed for the plot
        on those axes.
        """
        self.logger.debug("\tGenerating stamp for figure")

        # - - - - - - - - - - - - - - - - -
        # Stamp configuration
        # - - - - - - - - - - - - - - - - -
        # Location of the stamps
        title_loc = self.feature_locs['title']
        process_loc = self.feature_locs['process']
        stamp_loc = self.feature_locs['stamp']
        peak_text_loc = self.feature_locs['peak_text']

        # Text for the stamp
        title_text = {}
        process_text = {f'line_{i}': '' for i in range(2)}
        stamp_text = {f'line_{i}': '' for i in range(7)}
        peak_guess_text = {f'line_{i}': '' for i in range(7)}
        line = 0

        # Options for the stamp
        fonts = self.feature_fontsizes

        title_text_options = {
            'fontsize': fonts['title'],
            'ha': 'center',
        }
        process_text_options = {
            'fontsize': fonts['process'],
            'ha': 'left',
        }
        stamp_text_options = {
            'fontsize': fonts['stamp'],
            'ha': 'left',
         }
        peak_text_options = {
            'fontsize': fonts['peak'],
            'ha': 'center',
            'delta_y': 0.13
        }

        # - - - - - - - - - - - - - - - - -
        # Extracting relevant parameters
        # - - - - - - - - - - - - - - - - -
        pair_obs = self.pair_obs

        shower_model = fixed_param_values.get('shower_model',
                                              'pythia')
        pid_1 = int(fixed_param_values.get('PID_1', 0))
        pid_2 = int(fixed_param_values.get('PID_2', 0))
        outstate = fixed_param_values.get('outstate', 0)
        energy = fixed_param_values.get('energy', None)
        energyhat = partonic_energy(fixed_param_values)

        pid_dict = {11: r'e^-', -11: r'e^+',
                    2212: 'p', 1000822080: 'Pb',
                    0: 'X'}
        pidstr_1, pidstr_2 = pid_dict[pid_1], pid_dict[pid_2]
        is_proton_collision = int(pid_1) == 2212 and \
                              int(pid_2) == 2212

        outstate_dict = {'top': r'$t\bar{t}$',
                         'W': r'$W^+W^-$',
                         'w': r'$W^+W^-$',
                         'qcd': 'hadrons',
                         0: 'XX'}
        outstate_str = outstate_dict[outstate]

        pt_min = fixed_param_values.get('pt_min', None)
        pt_max = fixed_param_values.get('pt_max', None)

        n_events = fixed_param_values.get('n_events', None)
        level = fixed_param_values.get('level', None)

        jet_algorithm = fixed_param_values.get('jet_algorithm', None)
        subjet_algorithm = fixed_param_values.get('subjet_algorithm',
                                                  None)
        if subjet_algorithm:
            subjet_algorithm = subjet_algorithm.lower()

        jet_radius = fixed_param_values.get('jet_radius', None)
        subjet_radius = fixed_param_values.get('subjet_radius', None)

        z_cut = fixed_param_values.get('z_cut', None)
        weight_1 = fixed_param_values.get('weight_1', None)
        weight_2 = fixed_param_values.get('weight_2', None)

        isr = fixed_param_values.get('isr', None)
        fsr = fixed_param_values.get('fsr', None)
        mpi = fixed_param_values.get('mpi', None)

        def needs_comma(line_text):
            if line_text != '' and line_text[-2:] != ', ':
                return True
            return False


        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        # Title: Pair Observable
        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        if pair_obs is not None and \
                not isinstance(pair_obs, list):
            obsname = obs_title(pair_obs)
            # if subjet_radius is None or subjet_radius == 0:
            if obsname != 'Subjet EEC':
                obsname = obsname.lstrip('Subjet ')
            line_text = fr"\textbf{{{obsname}}}"
            if difference:
                line_text += ' Difference'
            title_text[f'line_{line}'] = line_text
            line += 1
        elif isinstance(pair_obs, list):
            if all(obs.startswith('mass') or obs.endswith('mass') for obs in pair_obs):
                line_text = fr"\textbf{{{obs_title('mass')}s}}"
            elif all(obs[:2] == 'm2' for obs in pair_obs):
                line_text = fr"\textbf{{{obs_title('m2')}s}}"
            else:
                line_text = fr"\textbf{{Subjet EWOCs}}"
            if difference:
                line_text += ' Difference'
            title_text[f'line_{line}'] = line_text
            line += 1


        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        # W Estimation Information
        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        # Only show money plot info for LHC W pair creation
        # with a single curve
        try:
            z_cut_val = float(z_cut)
        except TypeError:
            z_cut_val = 0

        show_w_guess = is_proton_collision and \
                               outstate == 'w' and \
                               z_cut_val >= 0 and \
                               show_w_guess

        if show_w_guess or fwhm_symbol:
            line_text = ''

            # Getting the peak location for the money plot
            xs, ys = self.axes[0].lines[0].get_data()

            # where the peak should be restricted to be the peak
            # actually shown in the plot
            xmin, xmax = self.axes[0].get_xlim()
            inds = [i for i, x in enumerate(xs) if xmin < x < xmax]
            peak_loc = xs[inds][np.argmax(ys[inds])]

            # Getting the full width at half the peak height
            width, loc_low, loc_high = get_full_width(
                                            xs[inds], ys[inds],
                                            height_ratio=0.5,
                                            return_loc=True)

            def get_mass(val):
                if pair_obs == 'costheta':
                    return energyhat * np.sqrt((1 - val)/8)
                if pair_obs in ['theta']:
                    return energyhat * val / 4
                if pair_obs in ['deltaR']:
                    energyhat = pt_val_from_min(pt_min)
                    return energyhat * val / (2*pt_fudge())
                if pair_obs in ['theta2', 'deltaR2']:
                    return energyhat * np.sqrt(val) / 4
                elif pair_obs == 'z':
                    return energyhat * np.sqrt(val) / 2
                elif pair_obs[:4] == 'mass':
                    return val
                elif pair_obs[:2] == 'm2':
                    return np.sqrt(val)
                self.logger.error("Invalid observable for LHC money plot.")
                return None

            # Writing peak location
            guess = get_mass(peak_loc)
            guess_high = get_mass(loc_high)
            guess_low = get_mass(loc_low)
            delta_guess = guess - M_W
            delta_high = guess_high - guess
            delta_low = guess - guess_low


            if show_w_guess:
                line_text +=\
                    r"\boldmath$m_{\rm Peak} - {m_W} = "\
                    + rf'{delta_guess:.2f}$ \textbf{{GeV}}'
                # TOGGLE: Write width as an "uncertainty"
                # line_text +=\
                #     fr"{{{delta_guess:.2f}}}^{{+{delta_high:.2f}}}_{{-{delta_low:.2f}}}$"
                peak_guess_text[f'line_{line}'] = line_text
                line += 1

            # Showing full-width at half max
            if fwhm_symbol:
                half_max = 0.5 * np.max(ys[inds])
                full_width = guess_high - guess_low

                if fwhm_symbol == '=':
                    line_text = \
                        r'$\textbf{FWHM\,=\,}'\
                        + fr'\textbf{{{full_width:.1f} GeV}}$'
                elif fwhm_symbol == '~':
                    line_text = \
                        r'$\textbf{FWHM} \sim '\
                        + fr'\textbf{{{full_width:.1f} GeV}}$'
                peak_guess_text[f'line_{line}'] = line_text
                if not show_w_guess:
                    line += 1

                fill_xs = np.linspace(loc_low/2, loc_high*2,
                                      1000)
                fill_ys = np.interp(fill_xs, xs, ys)
                fwhm_inds = [i for i, y in enumerate(fill_ys)
                               if y > half_max]

                if pair_obs in ['mass', 'm2']:
                    fill_color = 'firebrick'
                else:
                    fill_color = 'royalblue'

                self.axes[0].fill_between(
                    fill_xs[fwhm_inds], fill_ys[fwhm_inds],
                    np.zeros_like(fwhm_inds),
                    color=fill_color, alpha=0.3,
                zorder=0)


                ylow, yhigh = self.axes[0].get_ylim()
                scale = np.log(yhigh)-np.log(ylow)
                scale = scale/2

                if fwhm_symbol == '=':
                    ar_scale = 1.008
                    l_scale = 1.15
                else:
                    ar_scale = 1.012
                    l_scale = 1.08

                # Add arrow indicating FWHM
                self.axes[0].annotate("", xy=(loc_low/ar_scale,
                                              half_max),
                            xytext=(loc_high*ar_scale, half_max),
                            arrowprops=dict(color=fill_color,
                                            arrowstyle="<|-|>",
                                            linewidth=2.0))
                # with lines on the ends
                for loc in [loc_low, loc_high]:
                    self.axes[0].plot([loc, loc],
                            [half_max/l_scale, half_max*l_scale],
                            color=fill_color,
                            solid_capstyle='round')

            line_text = ''


        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        # Shower Model, Process
        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        line_text = ''

        if PLOT_ANALYTIC and not PLOT_DATA:
            line_text = DEFAULT_ANALYTIC_ACCURACY
            line_text += ", "
        elif shower_model is not None:
            line_text = (r"\texttt{Vincia}"
                    if shower_model == 'vincia'\
                  else (
                    r"\texttt{Dire}"
                      if shower_model == 'dire'\
                    else (r"\texttt{Pythia 8.307}"
                        if shower_model == 'pythia'\
                      else (shower_model)
                         )
                  ))
        else:
            line_text = ""

        # if show_w_guess:
        stamp_text[f'line_{line}'] = line_text
        line_text = ''
        line += 1

        # PIDs
        if needs_comma(line_text):
            line_text += ", "
        line_text += (r"$" +pidstr_1+pidstr_2 +r"$")

        # Outstates
        if outstate is not None:
            line_text += (" to " + outstate_str)

        # Center-of-mass energy
        if energy is not None:
            if needs_comma(line_text):
                line_text += ", "
            line_text += (r"$\sqrt{s}$="
                          +f"{energy/1000:.1f}"
                          +" TeV")

        if PLOT_ANALYTIC and not PLOT_DATA:
            # ---------------------------------
            # Finish up
            # ---------------------------------
            stamp_text[f'line_{line}'] = line_text
            line_text = ''
            line += 1

            # - - - - - - - - - - - - - - - - -
            # Adding stamp to the axes
            # - - - - - - - - - - - - - - - - -
            self.stamp(*title_loc, **title_text,
                       **title_text_options)
            self.stamp(*process_loc, **process_text,
                       **process_text_options)
            self.stamp(*stamp_loc, **stamp_text,
                       **stamp_text_options)
            return

        # ---------------------------------
        # Monte Carlo Information:
        # ---------------------------------
        # Level (Parton, Hadron, Hadron Level + MPI)
        if show_level and level is not None:
            if needs_comma(line_text):
                line_text += ', '
            line_text += f"{level} level ".capitalize()
            if mpi is not None:
                if str_to_bool(mpi):
                    line_text += "+ MPI"

        # ---------------------------------
        # Finish up
        # ---------------------------------
        stamp_text[f'line_{line}'] = line_text
        line_text = ''
        line += 1

        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        # Jet Information:
        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        # ---------------------------------
        # Jet algorithm string
        # ---------------------------------
        if jet_algorithm is not None:
            if jet_algorithm.startswith('ee_'):
                jet_algorithm = jet_algorithm[3:]
            if jet_algorithm == 'antikt':
                jet_algorithm = 'akt'
            if jet_algorithm.lower() == "c/a":
                jet_algorithm = "ca"
            jet_alg_str = f"{jet_algorithm}".upper()\
                                .replace("_", "-")
        else:
            jet_alg_str = ''

        # Deciding whether to print jet info
        # show_jet_info = self.plot_type == "Verbose" and \
        show_jet_info = \
            (jet_algorithm is not None\
            or subjet_algorithm is not None\
            or jet_radius is not None\
            or subjet_radius is not None)

        if show_jet_info:
            if jet_radius == 1000.:
                if line_text == '':
                    line_text = "Full event"
                else:
                    line_text += ", full event"
            # Ex: AKT8 Jets,
            elif jet_algorithm is not None \
                    and jet_radius is not None:
                if needs_comma(line_text):
                    line_text += ", "
                # line_text += \
                #     fr"$R^{{({jet_alg_str})}}_{{\rm jet}}$"\
                #     +fr"$=\,${jet_radius:.0f}"
                jet_text = f'{jet_alg_str}{10*jet_radius:.0f} '\
                            'Jets'
                if jet_algorithm == "ca":
                    line_text += r'C/A Jets, $R_\text{jet} \! =\,$'\
                                f'{jet_radius:.1f}'
                elif jet_algorithm in ["analytic", "theory"]:
                    line_text += r'$R_\text{jet} \!=\,$'\
                                f'{jet_radius:.1f}'
                else:
                    line_text += jet_text
            elif jet_algorithm is not None:
                if needs_comma(line_text):
                    line_text += ", "
                line_text += f"{jet_alg_str}"
                line_text += " jets"
            else:
                assert jet_radius is not None,\
                    "Unsupported use case: "\
                    "jet_algorithm and "\
                    "jet_radius are both None."

            # Ex: kt4 subjets
            if subjet_algorithm is not None\
                    and subjet_radius is not None:
                if needs_comma(line_text):
                    line_text += ", "
                # DEBUG: case-by-case soln for now
                if subjet_radius == 0.0:
                    line_text += 'all particles'
                elif subjet_radius < .1:
                    # line_text += \
                    #     fr"$r^{{({subjet_algorithm})}}_{{\rm sub}}$"\
                    #     +fr"$=\,${subjet_radius:.2f}"
                    line_text += f'{subjet_algorithm}'\
                                 f'{10*subjet_radius:.1f} '\
                                'subjets'
                else:
                    # line_text += \
                    #     fr"$r^{{({subjet_algorithm})}}_{{\rm sub}}$"\
                    #     +fr"$=\,${subjet_radius:.1f}"
                    line_text += f'{subjet_algorithm}'\
                                 f'{10*subjet_radius:.0f} '\
                                'subjets'


        # ---------------------------------
        # p_T Range
        # ---------------------------------
        show_pt_range = False

        if is_proton_collision:
            pt_str = r"$p_{T,\, \mathrm{jet}}$"
        else:
            stamp_text[f'line_{line}'] = line_text
            line_text = ''
            line += 1
            pt_str = r"$E_{\mathrm{jet}}$"

        if pt_min is not None and pt_min > 0:
            if needs_comma(line_text):
                line_text += ', '
            show_pt_range = True
            line_text += (f"{int(pt_min)} GeV "
                        r"$\! < \,$"+pt_str)
        else:
            pt_min = None
        if pt_max is not None and\
                energy is not None\
                and pt_max < energy/2:
            if needs_comma(line_text) and not pt_min:
                line_text += ', '
            show_pt_range = True
            if not pt_min:
                line_text += pt_str
            line_text += (r"$\, < $" + f"{pt_max:.1f} GeV")

        # ---------------------------------
        # Finish up
        # ---------------------------------
        stamp_text[f'line_{line}'] = line_text
        line_text = ''
        line += 1


        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        # Adding stamp to the axes
        # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
        self.stamp(*title_loc, **title_text,
                   **title_text_options)
        self.stamp(*process_loc, **process_text,
                   **process_text_options)
        self.stamp(*stamp_loc, **stamp_text,
                   **stamp_text_options)
        self.stamp(*peak_text_loc, **peak_guess_text,
                   **peak_text_options)


        if weight_1 and weight_1 != 1:
            assert weight_1 == weight_2, \
                f"{weight_1=} must equal {weight_2=}, for now."
            self.stamp(0.65, 0.87,
                   line_0=fr'\textbf{{Energy weight: {weight_1}}}',
                   fontsize=13)
        return


    def legend_prescription(self, params):
        """Generates a legend entry for the dataset
        corresponding to the given parameters.
        """
        params = {key: val for key, val in params.items()
                  if key in self.varied_parameters}
        if len(params) == 0:
            return None

        # Initialization
        legend_entry = ""

        # - - - - - - - - - - - - - - - - -
        # Observable
        # - - - - - - - - - - - - - - - - -
        if 'pair_obs' in params:
            if params['pair_obs'].startswith('jet'):
                legend_entry += obs_title(params['pair_obs'])
            # DEBUG: Angularity plots
            # elif self.plot_type == 'Angularity':
            #     legend_entry += obs_title(params['pair_obs'])[:-5]

        # - - - - - - - - - - - - - - - - -
        # Subjet radii
        # - - - - - - - - - - - - - - - - -
        if params.get('subjet_radius', None) is not None:
            # If the given curve should be characterized by a
            # subjet radius
            if legend_entry != "":
                legend_entry += ", "
            subjet_radius_str = f"{params['subjet_radius']:.2f}"
            if subjet_radius_str.endswith("0") and \
                    subjet_radius_str != '0.0':
                subjet_radius_str = subjet_radius_str[:-1]

            legend_entry += fr"$r^{{({self.subjet_alg_str})}}_{{\rm sub}}=\,$"\
                            + subjet_radius_str

        # - - - - - - - - - - - - - - - - -
        # z_cut and weights
        # - - - - - - - - - - - - - - - - -
        if 'z_cut' in params:
            if params['z_cut'] is not None:
                if legend_entry != "":
                    legend_entry += ", "
                legend_entry += r"$z_{\rm cut} = $ " \
                                + f"{params['z_cut']:.1f}"

        if 'weight_1' in params or 'weight_2' in params:
            if params['weight_1'] is not None and \
                    params['weight_2'] is not None:
                if legend_entry != "":
                    legend_entry += ", "
                legend_entry += "weights: ("
            if 'weight_1' in params:
                if params['weight_1'] is not None:
                    legend_entry += f"{params['weight_1']:.1f}, "
                else:
                    legend_entry += r"\cdot, "
            else:
                legend_entry += r"\cdot, "
            if 'weight_2' in params:
                if params['weight_2'] is not None:
                    legend_entry += f"{params['weight_2']:.1f})"
                else:
                    legend_entry += r"\cdot)"
            else:
                legend_entry += r"\cdot)"

        # - - - - - - - - - - - - - - - - -
        # Returning
        # - - - - - - - - - - - - - - - - -
        if legend_entry == "":
            return None

        self.metadata['showlegend'] = True
        return legend_entry


    def plot_physical_scales(self, fixed_param_values: dict,
                             fraction=0.8, alpha=0.3):
        # Getting peaks that might be expected by the physical scales
        # of the problem
        exp_peaks, scales, peak_colors =\
                expected_peaks_scales_colors(
                    observable=self.pair_obs,
                    **fixed_param_values)

        # Getting the peaks that lie within the x limits
        # of the plot
        try:
            x_min, x_max = self.metadata['xlim']
            _ = x_min < 0 < x_max
        except TypeError:
            x_min, x_max = self.axes[0].get_xlim()

        scales = [scale for scale, peak
                  in zip(scales, exp_peaks)
                  if x_min < peak < x_max]
        peak_colors = [color for color, peak
                       in zip(peak_colors, exp_peaks)
                       if x_min < peak < x_max]
        exp_peaks = [peak for peak in exp_peaks
                     if x_min < peak < x_max]

        # Plotting the lines
        ymax_vline = fraction * self.axes[0].get_ylim()[1]\
            + (1-fraction) * self.axes[0].get_ylim()[0]
        if self.axes[0].get_yscale() == 'log':
            ymax_vline = self.axes[0].get_ylim()[0]**(1-fraction)\
                * self.axes[0].get_ylim()[1]**fraction

        self.axes[0].vlines(exp_peaks, -1, ymax_vline,
                       lw=3.5, colors=peak_colors,
                       zorder=0, alpha=alpha).set_capstyle('round')
        if self.ratio_plot:
            self.axes[1].vlines(exp_peaks, -1, 1e7,
                           lw=3.5, colors=peak_colors,
                           zorder=0, alpha=alpha)


    def set_axes(self, pair_obs, xscale='log', yscale='log',
                 xlim=None, ylim=None):
        if xlim is None:
            xlim = lims[pair_obs][(xscale, yscale)][0]
        if ylim is None:
            ylim = lims[pair_obs][(xscale, yscale)][1]

        self.axes[0].set_xlim(xlim)
        self.axes[0].set_ylim(ylim)

        if xscale == 'log':
            [ax.set_xscale('log') for ax in self.axes]
        if yscale == 'log':
            self.axes[0].set_yscale('log')

        return

    def process_figure(self, fixed_param_values,
                       difference: str = '',
                       legend=True, **kwargs):
        """Processes the figure after plotting
        all relevant data.
        """
        # Getting plot parameters
        fonts = self.feature_fontsizes
        main_legend_loc = self.feature_locs['main_legend']
        line_legend_loc = self.feature_locs['line_legend']

        # params for the vertical line marking physical scales
        vline_fraction = kwargs.pop('vline_fraction', 0.9)
        vline_alpha = kwargs.pop('vline_alpha', 0.3)

        # Axis labels
        [ax.set_xlabel(xlabel_latex[self.obs_key])
         for ax in self.axes]
        self.axes[0].set_ylabel(diff_ewoc_latex[self.obs_key])
        self.axes[0].xaxis.get_label().set_fontsize(fonts['x_axis'])
        self.axes[0].yaxis.get_label().set_fontsize(fonts['y_axis'])

        # Stamp
        self.ewoc_stamp(fixed_param_values, **kwargs)

        # Lines corresponding to physical scales of the scattering
        self.plot_physical_scales(fixed_param_values,
                                  fraction=vline_fraction,
                                  alpha=vline_alpha)

        if not legend:
            plt.tight_layout()
            return

        # Legend
        handles, labels = self.axes[0].get_legend_handles_labels()

        # Main legend
        parameter_labels, label_inds = np.unique(labels,
                                                 return_index=True)
        parameter_handles = [handles[ind] for ind in label_inds]
        parameter_legend = self.axes[0].legend(
                          parameter_handles,
                          parameter_labels,
                          frameon=False,
                          prop={'size': fonts['main_legend']},
                          loc=main_legend_loc)
        # Making the main legend lines solid
        for line in parameter_legend.get_lines():
            line.set_linewidth(3)
            # line.set_linestyle('-')
        self.axes[0].add_artist(parameter_legend)

        # - - - - - - - - - - - - - - - - - - - - - -
        # Legend distinguishing different linestyles
        # - - - - - - - - - - - - - - - - - - - - - -
        linestyles = [line._dash_pattern
                      if line._dashSeq else 'solid'
                      for line in self.axes[0].lines]
        linestyles = np.unique(linestyles)

        unique_linestyles = np.unique(linestyles)

        # Line style legend for regular plots
        if not difference:
            line_labels = [line_label[linestyle]
                           for linestyle in unique_linestyles]

            if '--' in [line.get_linestyle() for line in
                            self.axes[0].get_lines()]\
                    and '--' not in unique_linestyles:
                unique_linestyles = np.array([*unique_linestyles,
                                              'dashed'])
                line_labels.append(line_label['dashed'])

        # Line style legend for ``difference'' plots
        else:
            line_labels = [difference_line_label[linestyle]\
                                [difference]
                           for linestyle in unique_linestyles]

            if '--' in [line.get_linestyle() for line in
                            self.axes[0].get_lines()]\
                    and '--' not in unique_linestyles:
                unique_linestyles = np.array([*unique_linestyles,
                                              'dashed'])
                line_labels.append("Power law fit")

        if len(unique_linestyles) == 0:
            self.logger.warning("EWOCPlotter: "
                                "WARNING: No lines "
                                "found in plot.")
            return False

        line_labels, unique_linestyles = \
            zip(*sorted(zip(line_labels, unique_linestyles),
                        key=ls_key))

        if len(unique_linestyles) > 1:
            line_handles = [
                       Line2D([0], [0], color='k',
                              linestyle=linestyle)
                       for linestyle in unique_linestyles]
            line_legend = self.axes[0].legend(line_handles,
                         line_labels,
                         frameon=False,
                         prop={'size': fonts['line_legend']},
                         loc=line_legend_loc)
            self.axes[0].add_artist(line_legend)

        # Tight layout
        plt.tight_layout()
        return True


    def savefig(self, filename):
        self.fig.tight_layout()
        super().savefig(filename=filename,
                        dirname=ewoc_figure_dir)



if __name__ == "__main__":
    plotter = EWOCPlotter('test', {}, {}, None,
                               xlabel=r'test $x$',
                               ylabel=r'test $y$')
    plotter.savefig('test.pdf')
