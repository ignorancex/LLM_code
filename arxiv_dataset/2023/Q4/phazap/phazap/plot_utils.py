import os
import numpy as np

import getdist
from getdist import plots
import matplotlib.pyplot as plt

from .phazap import phases_events
from . import utils
from . import tension_utils as tension

_default_plot_filename_str = "phazap_{}_{}_fbest_{}_fhigh_{}_flow_{}.pdf"

def phazap_plot(event1_postprocessed_phase, event2_postprocessed_phase, output_dir="./", output_filename=None, use_latex=True):
    """
    Plot the postprocessed phase for two events

    Parameters
    ----------
    event1_postprocessed_phase: PostprocessedPhase
        PostprocessedPhase instance for event 1
    event2_postprocessed_phase: PostprocessedPhase
        PostprocessedPhase instance for event 2
    output_dir: str, optional
        Output directory
    output_filename: str, optional
        Output filename
    use_latex: bool, optional
        Use LaTeX for the labels
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    
    """
    if use_latex:
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'serif'

    parameters_1, parameters_2, det_phases_1, det_phases_2, tau_phases_1, tau_phases_2, Dphi_f_1, Dphi_f_2, above_below = phases_events(event1_postprocessed_phase, event2_postprocessed_phase)

    event_name_1 = "event_1"
    if event1_postprocessed_phase.superevent_name is not None:
        event_name_1 = event1_postprocessed_phase.superevent_name
    event_name_2 = "event_2"
    if event2_postprocessed_phase.superevent_name is not None:
        event_name_2 = event2_postprocessed_phase.superevent_name

    nphases = 3
    length = np.shape(parameters_1)[1]
    
    #Compute volume phases     
    vol_phases_1, vol_phases_2 = tension.volume_only_phases(det_phases_1,det_phases_2)

    #Compute distances
    #We divide the samples in two groups, above and below the plane
    #only if the fraction of samples in each group is between 0.05 and 0.95
    frac_samples_below = len(above_below[above_below<0]) / len(above_below)
    frac_limit = 0.05
    if (frac_samples_below>frac_limit) & (frac_samples_below < 1 - frac_limit):
        #All phases
        dist_M = tension.distance_phases_with_shift(parameters_1,parameters_2[above_below<0],nphases)
        dist_P = tension.distance_phases_with_shift(parameters_1,parameters_2[above_below>0],nphases)
        dist_all = np.minimum(dist_M,dist_P)
    else:
        dist_all = tension.distance_phases_with_shift(parameters_1,parameters_2,nphases)

    dist = dist_all

    phase_shifts = np.array([0,1,2,-1])*np.pi/2
    phase_shift = phase_shifts[np.argmin(dist)]*np.hstack((np.ones(nphases),np.zeros(length-nphases)))

    #Event 2 is shifted by the phase shift that minimizes the distance
    parameters_2_shifted = parameters_2 + phase_shift
    parameters_2_shifted[:,:nphases] = np.mod(parameters_2_shifted[:,:nphases],2*np.pi)

    parameters_1_wrap, parameters_2_wrap = utils.wrap_phases(parameters_1,parameters_2_shifted,nphases)

    #labels
    labels = [
        r'\phi_\mathrm{H}',
        r'\phi_\mathrm{L}',
        r'\phi_\mathrm{V}',
        r'\tau_\mathrm{HL}',
        r'\tau_\mathrm{HV}',
        r'\Delta\phi_f',
    ]

    label_1 = event_name_1
    best_phase_shift = utils.format_pretty_phase_shift(phase_shifts[np.argmin(dist)]).replace('0', '0π').replace('π', '\pi').replace('±', '\pm')
    label_2 = fr"{event_name_2} (shifted by ${best_phase_shift}$)"

    color_1 = 'blue'
    color_2 = 'green'

    # Get the getdist MCSamples objects for the samples, specifying same parameter
    # names and labels; if not specified weights are assumed to all be unity
    names = ["x%s"%i for i in range(length)]
    #labels =  ["x_%s"%i for i in range(ndim)]
    samples_1 = getdist.MCSamples(samples=parameters_1_wrap,names=names, labels = labels,label=label_1)
    samples_2 = getdist.MCSamples(samples=parameters_2_wrap,names=names, labels = labels,label=label_2)


    # Start plotting
    g = plots.get_subplot_plotter(width_inch=6, scaling=False, rc_sizes=True)

    # Settings
    g.settings.figure_legend_frame = False

    # Triangle plot
    g.triangle_plot([samples_1, samples_2], 
                    filled=True, 
                    colors = [color_1, color_2],
                   line_args = [{'color':color_1},{'color':color_2}])

    plt.suptitle(r'Minimum distance = %s' % np.format_float_positional(np.min(dist), precision=2, fractional=False)+r', volume = %s' % np.format_float_positional(np.min(vol_phases_1), precision=2, fractional=False), va='bottom')

    if output_filename is None:
        output_filename = _default_plot_filename_str.format(
            event_name_1,
            event_name_2,
            event1_postprocessed_phase.fbest,
            event1_postprocessed_phase.fhigh,
            event1_postprocessed_phase.flow,
        )

    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight', transparent=True)

    return plt.gcf()