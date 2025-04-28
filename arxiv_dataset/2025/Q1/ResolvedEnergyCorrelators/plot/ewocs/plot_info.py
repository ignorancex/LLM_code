from utils.gen_utils import str_to_bool
from utils.plot_utils import adjust_lightness

# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Plotting
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# =====================================
# Flags for Plotting
# =====================================
PLOT_DATA = True                  # TOGGLE: Whether to plot data
PLOT_RATIO = False                # TOGGLE: Whether ratio plot if possible

DEFAULT_ANALYTIC_ACCURACY = 'LO'  # TOGGLE: Accuracy of analytic plots
PLOT_ANALYTIC = False             # TOGGLE: Whether analytic plot if possible

# =====================================
# Flags for analysis
# =====================================
# Flags for modular debugging during histogram creation
REGENERATE_HISTS = False  # TOGGLE: Whether to overwrite histograms

MULTIPROCESS = True       # TOGGLE: whether to use pool.multiprocess
                          # TOGGLE:   (not in use right now)

# Average angle of the decay products of a W in proton-proton at 14TeV
LHC_W_RADIUS = 0.3

# =====================================
# Flags for GUI
# =====================================
MAKE_GUI = True
DEFAULT_VARIED_PARAMETERS = ['subjet_radius', 'level', 'mpi']


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Different parameters for plots
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
LHC_PARAMETERS= {'PID_1': 2212, 'PID_2': 2212,
                 'energy': 14000}
LHC_W_PARAMS = LHC_PARAMETERS.copy()
LHC_W_PARAMS.update(
    {'outstate': 'w'}
)

EE_PARAMETERS= {'PID_1': 11, 'PID_2': -11,
                 'energy': 1000}
EE2HADRON_PARAMS = EE_PARAMETERS.copy()
EE2HADRON_PARAMS.update(
    {'outstate': 'qcd'}
)


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Different parameters for plots
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #

color_map_pp = {0.00: 'darkturquoise',
                0.03: 'mediumseagreen',
                0.1:  'goldenrod',
                0.3:  'indianred',
                0.6:  'mediumpurple',
                2.03: 'yellowgreen',
                3.03: 'palegreen',
                2.3:  adjust_lightness('#f99244', 0.9),
                3.3:  adjust_lightness('#ff980e', 0.9),
                }
color_map_ee = {0.05: adjust_lightness('#a2d2ff', 0.9),
                0.1:  adjust_lightness('#d1bdff', 0.9),
                0.2:  '#fcb274'}

linestyle_map_pp = {(False,False): (0, (2, 1.3)),
                    (True, False): (0, (5,2)),
                    (True, True) : 'solid'}
linestyle_map_ee = {'LO': (0, (2, 1.3)),
                    'parton': (0, (5, 2)),
                    'hadron': 'solid',
                    'LL': 'solid',}



def plot_style(collision_type='pp', **kwargs):
    if collision_type == 'pp':
        return plot_style_pp(**kwargs)
    if collision_type == 'ee':
        return plot_style_ee(**kwargs)

def plot_style_pp(**kwargs):
    plot_style = {}

    # Line style
    plot_style['linestyle'] = linestyle_map_pp[
        (kwargs['level']=='hadron', str_to_bool(kwargs['mpi']))]

    # Color
    if 'sub_rad' in kwargs.keys() and \
            float(kwargs['sub_rad']) in color_map_pp.keys():
        if kwargs.get('weight', 1) == 1:
            plot_style['color'] = color_map_pp[kwargs['sub_rad']]
        else:
            # Assuming sub_rad = 0.3
            plot_style['color'] = color_map_pp[kwargs.get('weight')+
                                            kwargs['sub_rad']]
    if str_to_bool(kwargs.get('mpi', None)):
        assert kwargs.get('level') == 'hadron', \
            f'Hadron-level required for MPI={kwargs.get("mpi")}, '\
            f'but level={kwargs.get("level")}.'
        plot_style['zorder'] = 1
        if kwargs.get('adjust_lightness', True):
            plot_style['color'] = adjust_lightness(plot_style['color'],
                                                   0.6)
    elif kwargs.get('level') == 'parton':
        plot_style['zorder'] = 3
        plot_style['color'] = adjust_lightness(plot_style['color'],
                                               1.1)
        if kwargs.get('weight', 1) != 1:
            plot_style['color'] = adjust_lightness(
                                    plot_style['color'], 1.25)
    else:
        plot_style['zorder'] = 2

    return plot_style


def plot_style_ee(**kwargs):
    plot_style = {}

    level = kwargs.get('level')

    # Line style
    plot_style['linestyle'] = linestyle_map_ee.get(level)

    # Color
    if 'sub_rad' in kwargs.keys() and \
            float(kwargs['sub_rad']) in color_map_ee.keys():
        plot_style['color'] = color_map_ee[kwargs['sub_rad']]
    if level in ['parton', 'LO']:
        plot_style['zorder'] = 3
        plot_style['color'] = adjust_lightness(plot_style['color'],
                                               1.15)
    else:
        plot_style['zorder'] = 2

    return plot_style


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# EWOC-Specific Parameters
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Types of EWOC parameters
EWOC_TYPES = {
    'pair_obs': 'str',
    'energy_weight': 'float',
}

# Default EWOC parameters
EWOC_DEFAULTS = {
    'pair_obs': 'mass',
    'energy_weight': 1.0,
}


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Plot Parameters
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# =====================================
# Plot Limits
# =====================================
lims = {'costheta': {('linear', 'linear'):
                        ((0.6, 1.0), (0, 10)),
                      ('linear', 'log'):
                        ((.5, 1.0), (.001, 10)),
                      ('log', 'linear'):
                        ((0.0, 0.5), (0, 30)),
                      ('log', 'log'):
                        ((.65, 1.0), (.1, 50))
                      },
         'theta'   : {('linear', 'linear'):
                        ((0, .25), (0, 20)),
                      ('linear', 'log'):
                        ((0, .25), (1e-4, 1e2)),
                      ('log', 'linear'):
                        ((3e-5, 1e0), (0, 0.2)),
                      ('log', 'log'):
                          ((1e-4, 1e-2), (1e-4, 1e0))
                      },
         'deltaR'   : {('linear', 'linear'):
                        ((0, 2e0), (0, 20)),
                      ('linear', 'log'):
                        ((0, 2e0), (1e-4, 1e2)),
                      ('log', 'linear'):
                        ((3e-5, 2e0), (0, 1.4)),
                      ('log', 'log'):
                          ((1e-4, 2e0), (1e-4, 3e1))
                      },
         'theta2'  : {('linear', 'linear'):
                        ((0, .25), (0, 20)),
                      ('linear', 'log'):
                        ((0, .25), (1e-4, 1e2)),
                      ('log', 'linear'):
                        ((3e-5, 1e0), (0, 0.2)),
                      ('log', 'log'):
                          ((1e-4, 1e-2), (1e-4, 1e0))
                      },
        'z'       : {('linear', 'linear'):
                        ((0, .25), (0, 20)),
                      ('linear', 'log'):
                        ((0, .25), (1e-4, 1e2)),
                      ('log', 'linear'):
                        ((3e-5, 1e0), (0, 0.2)),
                      ('log', 'log'):
                          ((1e-4, 1e-2), (1e-4, 1e0))
                      },
         'mass'   : {('linear', 'linear'):
                        ((0, 1e3), (0, 1.7e-2)),
                      ('linear', 'log'):
                        ((0, 150), (1e-4, .12)),
                      ('log', 'linear'):
                        ((20, 150), (0, 4.0)),  # Better w/jet mass
                      ('log', 'log'):
                        ((10, 500), (1e-4, 1e2))
                      },
         'm2'     : {('linear', 'linear'):
                        ((0, 1e6), (0, 1.7e-4)),
                      ('linear', 'log'):
                        ((0, 1e4), (1e-4, .12)),
                      ('log', 'linear'):
                        ((1e-2, 1e7), (0, .9)),
                      ('log', 'log'):
                        ((3e-4, 2e6), (1e-8, 5e2))
                      },
         'formtime': {('linear', 'linear'):
                        ((0, 1e4), (0, 1e-4)),
                      ('linear', 'log'):
                        ((0, 150), (1e-4, .12)),
                      ('log', 'linear'):
                        ((1e-3, 1e3), (0, 4e-1)),
                      ('log', 'log'):
                        ((1.0, 1e3), (1e-4, .7))
                      },
         'kt'   : {('linear', 'linear'):
                        ((0, 1e3), (0, 1.7e-2)),
                      ('linear', 'log'):
                        ((0, 150), (1e-4, .12)),
                      ('log', 'linear'):
                        ((2e-3, 1e3), (0, .9)),
                      ('log', 'log'):
                        ((1e-1, 1e3), (1e-7, 1e-1))
                      },
         'e1'       : {('linear', 'linear'):
                        ((0, .25), (0, 20)),
                      ('linear', 'log'):
                        ((0, .25), (.1, 50)),
                      ('log', 'linear'):
                        ((3e-5, 1e0), (0, .5)),
                      ('log', 'log'):
                        ((1e-8, 1e2), (1e-10, 1e2))
                      },
        }

lims['mass_onshell'] = lims['mass']
lims['mass_approx'] = lims['mass']
lims['mass_tot'] = lims['mass']

lims['m2_onshell'] = lims['m2']
lims['m2_approx'] = lims['m2']
lims['m2_tot'] = lims['m2']

lims['angularity'] = lims['e1']
lims['e2'] = lims['e1']
lims['e4'] = lims['e1']
lims['e6'] = lims['e1']

lims['splitangularity'] = lims['e1']
lims['splite1'] = lims['e1']
lims['splite2'] = lims['e1']
lims['splite4'] = lims['e1']
lims['splite6'] = lims['e1']

lims['jet_mass'] = lims['mass']


# =====================================
# Typesetting
# =====================================
obs_latex = {'costheta': r'${\cos\theta}$',
             'theta': r'$\theta$',
             'deltaR': r'$R$',
             'theta2': r'${\theta^2}$',
             'z': r'$z = (1-\cos\theta)/2$',
             'm2_onshell': r'On-Shell ${m^2}$',
             'mass_onshell': r'On-Shell $m$',
             'm2_approx': r'Approx. ${m^2}$',
             'mass_approx': r'Approx. $m$',
             'm2_tot': r'${m^2}$',
             'mass_tot': r'$m$',
             'm2': r'${m^2}$',
             'mass': r'$m$',
             'formtime_onshell': r'On-Shell $\tau$',
             'formtime_approx': r'Approx. $\tau$',
             'formtime_tot': r'Full $\tau$',
             'formtime': r'$\tau$',
             'kt': r'${k_t}$',
             'angularity': r'${e_\varsigma}$',
             'e1': r'${e_1}$',
             'e2': r'${e_2}$',
             'e4': r'${e_4}$',
             'e6': r'${e_6}$',
             'splite1': r'${e_1}$',
             'splite2': r'${e_2}$',
             'splite4': r'${e_4}$',
             'splite6': r'${e_6}$',
             # Jet observables
             'mmdt':      'mMDT Mass',
             'jet_mass':      'Jet Mass',
             'jet_pt':       r'$p_T$',
             'jet_eta':      r'$\eta$',
             # Others
             'mass':         r'$m$'
         }
units_label = {'costheta': '',
               'theta': '',
               'deltaR': '',
               'theta2': '',
               'z': '',
               'm2_onshell': r' (GeV$^2$)',
               'mass_onshell': ' (GeV)',
               'm2_approx': r' (GeV$^2$)',
               'mass_approx': ' (GeV)',
               'm2_tot': r' (GeV$^2$)',
               'mass_tot': ' (GeV)',
               'm2': r' (GeV$^2$)',
               'mass': ' (GeV)',
               'formtime_onshell': r' (GeV$^{-1}$)',
               'formtime_approx': r' (GeV$^{-1}$)',
               'formtime_tot': r' (GeV$^{-1}$)',
               'formtime': r' (GeV$^{-1}$)',
               'kt': ' (GeV)',
               'angularity': '',
               'e1': '',
               'e2': '',
               'e4': '',
               'e6': '',
               'splite1': '',
               'splite2': '',
               'splite4': '',
               'splite6': '',
               # Jet observables
               'mmdt':     ' (GeV)',
               'jet_mass':     ' (GeV)',
               'jet_pt':       ' (GeV)',
               'jet_eta':      '',
               # Others
               'mass':         ' (GeV)',
               }

xlabel_latex = {}
for key in obs_latex:
    xlabel_latex[key] = obs_latex[key] + units_label[key]


# TOGGLE: Linear vs. logarithmically normalized histogram y-axis labels
# diff_ewoc_latex = {'costheta': r'$\frac{d\,\Sigma}{d\,\cos\theta}$',
#                    'z': r'$\frac{d\,\Sigma}{d\,z}$',
#                    'm2': r'$\frac{d\,\Sigma}{d\,{m^2}}$',
#                    'mass': r'$\frac{d\,\Sigma}{d\,m}$',
#                    'formtime': r'$\frac{d\,\Sigma}{d\,\tau}$',
#                    'kt': r'$\frac{d\,\Sigma}{d\,k_t}$',
#                    'e1': r'$\frac{d\,\Sigma}{d\,e_1}$'
#                    }
diff_ewoc_latex = {'costheta':
                       r'$\frac{d\,\Sigma}{d\,\log {\cos\theta}}$',
                   'z': r'$\frac{d\,\Sigma}{d\,\log z}$',
                   'theta': r'$\frac{d\,\Sigma}{d\,\log \theta}$',
                   'deltaR':
                       r'$\frac{\text{d}\,\Sigma_\text{\small{EEC}}}{\text{d}\,\log R}$',
                   'theta2': r'$\frac{d\,\Sigma}{d\,\log {\theta^2}}$',
                   'm2': r'$\frac{d\,\Sigma}{d\,\log {m^2}}$',
                   'mass': r'$\frac{\text{d}\,\Sigma_m}{\text{d}\,\log m}$',
                   'formtime':
                       r'$\frac{d\,\Sigma}{d\,\log \tau}$',
                   'kt': r'$\frac{d\,\Sigma}{d\,\log {k_t}}$',
                   'angularity':
                       r'$\frac{d\,\Sigma}{d\,\log {e_\varsigma}}$',
                   'e1': r'$\frac{d\,\Sigma}{d\,\log {e_1}}$',
                   'e2': r'$\frac{d\,\Sigma}{d\,\log {e_2}}$',
                   'e4': r'$\frac{d\,\Sigma}{d\,\log {e_4}}$',
                   'e6': r'$\frac{d\,\Sigma}{d\,\log {e_6}}$',
                   'splite1': r'$\frac{d\,\Sigma}{d\,\log {e_1}}$',
                   'splite2': r'$\frac{d\,\Sigma}{d\,\log {e_2}}$',
                   'splite4': r'$\frac{d\,\Sigma}{d\,\log {e_4}}$',
                   'splite6': r'$\frac{d\,\Sigma}{d\,\log {e_6}}$',
                   # Jet observables
                   'mmdt':     r'$\frac{d\,\Sigma}{d\,\log m}$',
                   'jet_mass': r'$\frac{d\,\Sigma}{d\,\log m}$',
                   'jet_pt':   r'$\frac{d\,\Sigma}{d\,\log p_T}$',
                   'jet_eta':  r'$\frac{d\,\Sigma}{d\,\log \eta}$',
                   }

cuml_ewoc_latex = {'costheta': r'$\Sigma({\cos\theta})$',
                   'theta': r'$\Sigma(\theta)$',
                   'deltaR': r'$\Sigma(\Delta R)$',
                   'theta2': r'$\Sigma({\theta^2})$',
                   'z': r'$\Sigma(z)$',
                   'm2': r'$\Sigma({m^2})$',
                   'mass': r'$\Sigma(m)$',
                   'formtime': r'$\Sigma(\tau)$',
                   'kt': r'$\Sigma({k_t})$',
                   'angularity':
                       r'$\Sigma({e_\varsigma})$',
                   'e1': r'$\Sigma({e_1})$',
                   'e2': r'$\Sigma({e_2})$',
                   'e4': r'$\Sigma({e_4})$',
                   'e6': r'$\Sigma({e_6})$',
                   'splite1': r'$\Sigma({e_1})$',
                   'splite2': r'$\Sigma({e_2})$',
                   'splite4': r'$\Sigma({e_4})$',
                   'splite6': r'$\Sigma({e_6})$',
                   # Jet observables
                   'jet_mass':    r'$\Sigma(m)$',
                   'jet_pt':      r'$\Sigma(p_T)$',
                   'jet_eta':     r'$\Sigma(\eta)$',
                  }

for qualifier in ['onshell', 'approx', 'tot']:
    diff_ewoc_latex['m2_' + qualifier] = diff_ewoc_latex['m2']
    diff_ewoc_latex['mass_' + qualifier] = diff_ewoc_latex['mass']
    diff_ewoc_latex['formtime_' + qualifier] = diff_ewoc_latex['formtime']
    cuml_ewoc_latex['m2_' + qualifier] = cuml_ewoc_latex['m2']
    cuml_ewoc_latex['mass_' + qualifier] = cuml_ewoc_latex['mass']
    cuml_ewoc_latex['formtime_' + qualifier] = cuml_ewoc_latex['formtime']


def obs_title(observable):
    """Returns a title for plotting the given observable
    in LaTeX form.
    """
    if observable in ['costheta', 'z', 'theta', 'deltaR', 'theta2']:
        # title = r'$\cos(\theta)$ EWOC'
        # title = r'$z$ EWOC'
        title = 'EEC'
    elif observable == 'mass':
        title = r'Mass EWOC'
    elif observable == 'mass_onshell':
        title = r'On-Shell Mass EWOC'
    elif observable == 'mass_approx':
        title = r'Approx. Mass EWOC'
    elif observable == 'mass_tot':
        # title = r'Full Mass EWOC'
        title = r'Mass EWOC'
    elif observable == 'm2':
        title = r'Mass-Squared EWOC'
    elif observable == 'm2_onshell':
        title = r'On-Shell Mass-Squared EWOC'
    elif observable == 'm2_approx':
        title = r'Approx. Mass-Squared EWOC'
    elif observable == 'm2_tot':
        title = r'Full Mass-Squared EWOC'
    elif observable == 'formtime':
        title = r'Formation Time EWOC'
    elif observable == 'kt':
        title = r'$k_T$ EWOC'
    elif observable == 'e1':
        title = r'$e^{(1)} = k_T/p_T$ EWOC'
        # TOGGLE: e+e- angularity defn
        title = r'$e^{(1)} = k_T/E_{\rm jet}$ EWOC'
    elif observable == 'e2':
        title = r'$e^{(2)} = m^2/p_T^2$ EWOC'
        # TOGGLE: e+e- angularity defn
        title = r'$e^{(2)} = m^2/E_{\rm jet}^2$ EWOC'
    elif observable == 'e4':
        title = r'$e^{(4)}$ EWOC'
    elif observable == 'e6':
        title = r'$e^{(6)}$ EWOC'
    elif observable == 'angularity':
        title = 'Angularity EWOC'
    elif observable == 'splite1':
        title = r'$e^{(1)} = k_T/p_T$ EWOC'
    elif observable == 'splite2':
        title = r'$e^{(2)} = m^2/p_T^2$ EWOC'
    elif observable == 'splite4':
        title = r'$e^{(4)}$ EWOC'
    elif observable == 'splite6':
        title = r'$e^{(6)}$ EWOC'
    elif observable == 'angularity':
        title = 'Angularity EWOC'
    # Jet Observables
    elif observable == 'mmdt':
        title = 'mMDT Mass'
    elif observable == 'jet_mass':
        title = 'Jet Mass'
    elif observable == 'jet_pt':
        title = r'Jet $p_T$'
    elif observable == 'jet_eta':
        title = r'Jet $\eta$'
    else:
        raise AssertionError(f"Invalid {observable = }")

    return title
