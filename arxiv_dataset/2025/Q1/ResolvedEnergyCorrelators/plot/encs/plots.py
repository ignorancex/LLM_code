from pathlib import Path
import subprocess
import warnings
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

import matplotlib.ticker as mticker

# Histogram imports
from histogram import HistogramData
from histogram import HistogramData1D
from histogram import HistogramData2D

# Plot imports
from plotter import Plotter
from utils.plot_utils import adjust_lightness
from plotter import enc_data_dir, enc_figure_dir
from utils.postprocess import collision_stamp


# =====================================
# Plotting flags
# =====================================
density_colormap = {
    'opendata': 'magma_r',
    'qcd': 'YlGnBu',
    'w'  : 'PuRd',
    'top': 'YlOrBr',
    None : None
}

oldenglish_path = "/usr/share/fonts/oldenglishtextmt.ttf"
oeboots_path = "/usr/share/fonts/Old_Englished_Boots.ttf"

# Add the font to matplotlib
bl_prop = fm.FontProperties(fname=oeboots_path)


# =====================================
# Utility functions
# =====================================
def check_normalization(hist):
    # If we are dealing with an EnC which should be normalized,
    # check for normalization
    hist.integrate_histogram(scheme='log10', outflow_weight=1)
    weight = hist.metadata['weight']

    if not hasattr(weight, '__iter__'):
        if weight != 1:
            return True
    else:
        weight=np.array(weight)
        if all(weight == 1):
            return True

    if not np.isclose(hist.integral, 1.0, rtol=5e-2):
        warnings.warn(f"E^nC has weights {hist.metadata['weight']} "
                      "but does not integrate to 1 "
                      f"({hist.integral=})")
        return False


# =====================================
# Plotting functions
# =====================================
def plot_1d_enc(file_name=None, hist_data=None,
                primary_observable='theta1',
                variable_order=None,
                integration_scheme=None,
                cumulative=False,
                **kwargs):
    # If no variable order is given, assume data is 1d
    if variable_order is None:
        variable_order = [primary_observable]
    if integration_scheme is None:
        integration_scheme = {var: 'linear'
                              for var in variable_order}

    # Loading and validating histogram
    hist1d = HistogramData(file_name=file_name,
                           hist_data=hist_data,
                           variable_order=variable_order)

    # Integrating over non-primary observables
    for var in variable_order:
        if var == primary_observable:
            continue
        hist1d = hist1d.integrate_over_variable(var,
                            scheme=integration_scheme[var],
                            outflow_weight=1)

    hist1d = HistogramData1D(hist_data=hist1d)
    check_normalization(hist1d)

    if cumulative:
        scheme = integration_scheme[primary_observable]
        hist1d = hist1d.cumulative(outflow_weight=1, scheme=scheme,
                                   **kwargs)

    # Plotting test
    if kwargs:
        # Preparing file name for figure
        if file_name:
            save_name = Path(file_name).stem
        else:
            save_name = '1d_histogram'
        save_name = kwargs.pop('save', save_name)

        # Making plot
        hist1d.make_plot(**kwargs)

        # Saving
        if save_name:
            hist1d.plot.savefig(save_name+'.pdf',
                                enc_figure_dir)

    return hist1d


def plot_2d_density(file_name=None,
                    hist_data=None,
                    log_normalize=True,
                    **kwargs):
    try:
        hist3d = HistogramData(file_name, hist_data,
                               variable_order=['theta1',
                                       'theta2_over_theta1',
                                       'phi'])
        integration_scheme_3d = {'theta1': 'log10',
                                 'theta2_over_theta1': 'linear',
                                 'phi': 'linear'}
        t1_key = 'theta1'
        t2_t1_key = 'theta2_over_theta1'
    except:
        hist3d = HistogramData(file_name, hist_data,
                               variable_order=['thetaL',
                                         'thetaS_over_thetaL',
                                         'phi'])
        integration_scheme_3d = {'thetaL': 'log10',
                                 'thetaS_over_thetaL': 'linear',
                                 'phi': 'linear'}
        t1_key = 'thetaL'
        t2_t1_key = 'thetaS_over_thetaL'

    # Normalization test
    check_normalization(hist3d)

    # Testing integration over phi:
    hist2d = HistogramData2D(
                hist_data=hist3d.integrate_over_variable('phi'))

    # Testing new normalization
    integration_scheme_2d = integration_scheme_3d.copy()
    integration_scheme_2d.pop('phi')

    # Normalization test
    check_normalization(hist2d)


    # Changing variables
    if not log_normalize:
        bin1_centers = hist2d.centers[t1_key]
        hist2d.hist = np.array([
            hist2d.hist[i1] / t1
            for i1, t1 in
                enumerate(bin1_centers)
        ])


    # Testing plots:
    if kwargs:
        # Remove outflow bins for theta1 and validate
        finite_theta_edges = hist2d.edges[t1_key][1:-1]
        finite_theta_centers = hist2d.centers[t1_key][1:-1]
        finite_hist2d = [np.array(hist1d[1:-1])
                         for hist1d in hist2d.hist.T]
        hist2d.hist = np.array(finite_hist2d).T
        hist2d.edges[t1_key] = finite_theta_edges
        hist2d.centers[t1_key] = finite_theta_centers
        hist2d.validate()

        # Name for file saving
        if file_name:
            save_name = Path(file_name).stem
        else:
            save_name = '2d_histogram'
        save_name = kwargs.pop('save', save_name)

        # Log-normalized plot
        log_colorbar = kwargs.pop('log_colorbar', False)
        lin_colorbar = kwargs.pop('lin_colorbar', True)
        if log_colorbar:
            hist2d.make_plot('density', log_norm=True,
                             **kwargs)
            if save_name:
                hist2d.density.savefig(
                    save_name+'_lognorm_density.pdf')

        # Linear-normalized plot
        if lin_colorbar:
            hist2d.make_plot('density', log_norm=False,
                             **kwargs)
            if save_name:
                hist2d.density.savefig(
                    save_name+'_linnorm_density.pdf')

    return hist2d


def plot_2d_bullseye(file_name=None, hist_data=None,
                     theta1=None, **kwargs):
    try:
        hist3d = HistogramData(file_name, hist_data,
                               variable_order=['theta1',
                                   'theta2_over_theta1',
                                   'phi'])
        old_defn = False
        t1_key = 'theta1'
        t2_t1_key = 'theta2_over_theta1'
        hist3d.edges[t1_key]
    except:
        hist3d = HistogramData(file_name, hist_data,
                               variable_order=['thetaL',
                                   'thetaS_over_thetaL',
                                   'phi'])
        old_defn = True
        t1_key = 'thetaL'
        t2_t1_key = 'thetaS_over_thetaL'
        hist3d.edges[t1_key]

    # If not given any theta1 values, plot all
    if theta1 is None:
        theta1 = hist3d.centers['theta1']

    # If only given a single theta1 value, make an associated plot
    if not hasattr(theta1, '__iter__'):
        try:
            t1_ind = np.digitize([theta1], hist3d.edges[t1_key])[0] - 1
            theta1 = hist3d.centers[t1_key][t1_ind]
            t1_bin = (hist3d.edges[t1_key][t1_ind],
                      hist3d.edges[t1_key][t1_ind+1])
        except:
            return
        finally:
            if theta1 == 0 or not np.isfinite(theta1):
                return

        # Get sub-histogram for this value of theta1 and normalize
        bullseye2d = hist3d.get_sub_histogram(t1_key, theta1)

        # If only nans, 0s, or infinities, continue
        nonzero_hist = bullseye2d.hist[bullseye2d.hist != 0]
        if not np.isfinite(nonzero_hist).any():
            return

        bullseye2d = HistogramData2D(hist_data=bullseye2d)

        # Normalize correctly (currently, the histogram is
        # normalized to 1 when integrated dlogt1 d(t2/t1) dphi
        bin2_centers = bullseye2d.centers[t2_t1_key]
        bin2_edges = bullseye2d.edges[t2_t1_key]
        theta2_centers = bin2_centers * theta1
        bullseye2d.hist = np.array([
            bullseye2d.hist[i2] / ((theta1+1e-30)**2
                                   * (t2+1e-30))
            for i2, t2 in enumerate(theta2_centers)
        ])
            # bullseye2d.hist[i2] / (theta2_centers[i2] * t1**2)
            # for i2 in range(len(theta2_centers))])

        # Update variables used for plotting:
        # theta2 goes from 0 to theta1
        bullseye2d.edges[t2_t1_key] = bin2_edges*theta1
        bullseye2d.centers[t2_t1_key] = theta2_centers

        # Plot bullseye and save
        if kwargs:
            kwargs = kwargs.copy()

            # Setting up filename for saving figure
            if file_name:
                save_name = Path(file_name).stem
            else:
                save_name = '2d_histogram'

            save_name = kwargs.pop('save', save_name)

            kwargs.update({'ylim': (0, theta1)})

            weights = (1, *bullseye2d.metadata['weight'])

            # Creating plot
            bullseye2d.make_plot('bullseye', log_norm=True,
                                 radial_coordinate=t2_t1_key,
                                 # cbar_label=cbar_label,
                                 **kwargs)

            # Set labels and limits
            ax = bullseye2d.bullseye.axes[0]
            # ax.set_ylim(0, t1_bin[1])

            # if using old defn, adding phase space boundaries
            if old_defn:
                # in terms of thetaL, thetaS
                phi_bound = np.linspace(-np.pi/2, np.pi/2, 1000)
                r_bound = np.minimum(2*np.cos(phi_bound),
                                     1 / (2 * np.cos(phi_bound)))
                ax.plot(phi_bound, theta1*r_bound, lw=4,
                        color='cornflowerblue', alpha=0.8)

            # # Drawing curves and arrows
            # rad_end = 1.35
            # phi_end = 0.8
            # # Angular arrow
            # phis = np.arange(0, phi_end, .01)
            # rads = np.ones_like(phis)*theta1*1.1
            # ax.plot(phis, rads, lw=2, color='dimgrey', clip_on=False)
            # ax.arrow(phis[-1], theta1*1.1, 1e-10*theta1, 0,
            #          clip_on=False, width=0.0,
            #          facecolor='dimgrey', edgecolor='dimgrey',
            #          head_width=0.05*theta1,
            #          head_length=0.2*theta1)
            # # Radial arrow
            # ax.plot([-0.85, -0.85], [theta1, theta1*rad_end],
            #         lw=2, color='dimgrey', clip_on=False)
            # ax.arrow(-0.85, theta1*rad_end, 0, 1e-10*theta1,
            #          clip_on=False, width=0.0,
            #          facecolor='dimgrey', edgecolor='dimgrey',
            #          head_width=0.11*theta1,
            #          head_length=0.07*theta1)

            # Saving
            if save_name:
                bullseye2d.bullseye.savefig(
                    save_name+(f"{str(theta1).replace('.','-')}"
                              "_bullseye.pdf"),
                    enc_figure_dir)

        return bullseye2d

    # Otherwise, looping over many theta1 values
    all_hists = []
    for it, t1 in enumerate(theta1):
        all_hists.append(
        plot_2d_bullseye(hist_data=hist3d, theta1=t1,
                         **kwargs)
        )
        plt.close()
    return all_hists


def bullseye_inset(hist_data, inner_rad,
                   fig=None, ax=None, **kwargs):
    # Creating an inset plot
    if fig is None:
        fig = hist_data.bullseye.fig
    if ax is None:
        ax  = fig.add_axes([0.00, 0.76, 0.2, 0.2])

    ax.set_aspect('equal')

    # Setting axes to be bounded by the jet radius
    jet_radius = float(hist_data.metadata['jet_rad'])
    plot_rad = max(jet_radius, inner_rad)

    ax.axis('off')
    ax.set_xlim(-plot_rad*1.1, plot_rad*1.1)
    ax.set_ylim(-plot_rad*1.1, plot_rad*1.1)

    # if is_ax:   # Could change ax below to ax
    # ax.set_ylim((0, jet_radius))
    # Drawing a dotted gray circle at the jet boundary
    jet_centerx = max(0, inner_rad-jet_radius)
    ax.add_patch(
        plt.Circle((jet_centerx, 0.0), jet_radius,
                   ls="--", lw=2.3,
                   color="gray", fill=False,
                   transform=ax.transData._b,
                   )
        )
    # Setting up particle colors
    colors = ['black', 'royalblue', 'firebrick',
              'mediumorchid', 'sienna']
    inset_pts = kwargs.pop('inset_pts', [])
    ind_bound = 1 + len(inset_pts)
    boundpt_col  = colors[ind_bound]
    bulls_edge   = adjust_lightness(boundpt_col, 0.5)

    # Drawing a colored circle for the particle within the bullseye
    ax.add_patch(
        plt.Circle((0.0, 0.0), inner_rad,
                   ls="-", lw=1.0, alpha=0.25,
                   color=colors[ind_bound+1],
                   transform=ax.transData._b,)
        )

    # Drawing a dark circle for the bullseye boundary
    ax.add_patch(
        plt.Circle((0.0, 0.0), inner_rad,
                   ls="-", lw=2.3,
                   color=bulls_edge, fill=False,
                   transform=ax.transData._b,)
        )

    # Drawing points for the special particle
    special_loc = (0, 0)
    ax.plot(0, 0, color='black', marker='o')
    # and the innermost non-special particle
    bound_loc = (inner_rad, 0)
    ax.plot(inner_rad, 0, color=boundpt_col,
               marker='o')
    # and any additional particles
    for i, (R, phi) in enumerate(inset_pts):
        prev_col   = colors[ind_bound-i]
        insetpt_col  = colors[ind_bound-i-1]

        alpha = 0.09*(len(inset_pts) - i)/len(inset_pts)
        # ax.add_patch(
        #     plt.Circle((0.0, 0.0), R, color=prev_col,
        #                alpha=alpha, lw=0,
        #                transform=ax.transData._b,)
        #     )
        ax.plot(R*np.cos(phi), R*np.sin(phi),
                alpha=alpha*9, color=insetpt_col, marker='o',
                markeredgewidth=0)

    return ax


def plot_runtime(file_name=None, hist_data=None,
                 plotter=None, **kwargs):
    # Loading and validating histogram
    hist = HistogramData(file_name=file_name,
                         hist_data=hist_data)

    # Preparing file name for figure
    if file_name:
        save_name = Path(file_name).stem
    else:
        save_name = 'runtime'
    save_name = kwargs.pop('save', save_name)

    # Setting up plotter
    if plotter is None:
        plotter = Plotter(xlabel='Number of Particles',
                          ylabel=r'Runtime (ms)',
                          **kwargs)

        # Putting stamp on histogram
        for key, val in zip(['jet_alg', 'jet_rad', 'pt_min', 'pt_max'],
                            ['akt', 0.5, 500, 550]):
            if key not in hist.metadata:
                hist.metadata[key] = val

    # Getting runtime data
    means = np.array(hist.metadata['runtime_means'])/1000
    stds  = np.array(hist.metadata['runtime_stds'])/1000
    nans  = np.isnan(means) * np.isnan(stds)

    # Removing nans
    means = np.ma.masked_where(nans, means)
    stds  = np.ma.masked_where(nans, stds)
    nums  = np.ma.masked_where(nans, np.arange(0, len(means)))

    # Plotting runtime data
    color = kwargs.pop('color', 'cornflowerblue')
    label = kwargs.pop('label', None)
    ls    = kwargs.pop('linestyle', '-')
    alpha = kwargs.pop('alpha', 0.0)

    plotter.axes[0].plot(nums, means, color=color,
                         label=label, ls=ls)
    if alpha > 0:
        stdev_scale = kwargs.get('stdev_scale', 1.0)
        plotter.axes[0].fill_between(nums,
                means-stdev_scale*stds, means+stdev_scale*stds,
                color=color, alpha=alpha)

    # Saving
    if save_name:
        plotter.axes[0].legend()
        plotter.savefig(save_name+'.pdf', enc_figure_dir)

    return hist, plotter


def logarithmic_axes(ax):
    """Logarithmic axes for 3d plots."""

    def log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_formatter(
            mticker.FuncFormatter(log_tick_formatter))
        axis.set_major_locator(
            mticker.MaxNLocator(integer=True))


def density_4d(hist_data, colormap, **kwargs):
    fig = plt.figure()

    ax = fig.add_subplot(projection = '3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Getting X, Y, and Z values
    (X, Y, Z) = (np.log10(hist_data.centers[var])
                for var in hist_data.variable_order)
    X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

    # Getting histogram
    data = hist_data.hist

    scale = data.flatten()/np.max(data)
    mask = data > 0  #10.01

    # Making plot
    ax.scatter(X, Y, Z,
               c=scale,
               alpha=0.01,
               s=10.0 * mask,
               edgecolor="face",
               marker="o",
               cmap=colormap, linewidth=0)
    logarithmic_axes(ax)

    fig.tight_layout()


    return fig, ax
