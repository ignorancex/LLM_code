import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from histogram import HistogramData
from plotter import ewoc_data_dir, property_data_dir
from ewocs.ewoc_plotter import EWOCPlotter
from ewocs.plot_info import LHC_W_PARAMS, plot_style

from ewocs.nonpert_comparison import plot_feature_locs, plot_fontsizes

from qcd.qcd_basics import M_W
from utils.plot_utils import get_full_width


high_res = True


def get_mass_from_obs(val, pair_obs, energyhat):
    if pair_obs == 'costheta':
        return energyhat * np.sqrt((1 - val)/8)
    if pair_obs in ['theta', 'deltaR']:
        return energyhat * val / 4
    if pair_obs in ['theta2', 'deltaR2']:
        return energyhat * np.sqrt(val) / 4
    elif pair_obs == 'z':
        return energyhat * np.sqrt(val) / 2
    elif pair_obs[:4] in ['mass', 'mmdt']:
        return val
    elif pair_obs[:2] == 'm2':
        return np.sqrt(val)
    raise RuntimeError("Invalid observable for LHC money plot.")
    return None


def get_mass_from_ax(pair_obs, ax, line_index, return_width=False):
    # Getting the peak location for the money plot
    xs, ys = ax.lines[line_index].get_data()

    # where the peak should be restricted to be the peak
    # actually shown in the plot
    xmin, xmax = ax.get_xlim()
    inds = [i for i, x in enumerate(xs) if xmin < x < xmax]
    peak_loc = xs[inds][np.argmax(ys[inds])]

    # Getting the full width at half the peak height
    width, loc_low, loc_high = get_full_width(
                                    xs[inds], ys[inds],
                                    height_ratio=0.5,
                                    return_loc=True)

    energyhat = 14000/13

    if return_width:
        return get_mass_from_obs(loc_high, pair_obs, energyhat) \
            - get_mass_from_obs(loc_low, pair_obs, energyhat)

    guess = get_mass_from_obs(peak_loc, pair_obs, energyhat)
    return guess


def get_ewoc_peak(pair_obs, r_sub, ptmin=500,
                  level='hadron', mpi=True,
                  weight=1, return_width=False,
                  charged=False, smeared=False,
                  plot=True, xmin=None, xmax=None,
                  use_fit=True):
    assert level == 'hadron' or not(mpi),\
        f"If {level=}, mpi must be false."
    variable = 'pair_obs'

    file_dir = ewoc_data_dir
    if high_res:
        file_name = f'{pair_obs}_lhc_w_highres120bins_'\
                    +('parton_' if level == 'parton' else '')\
                    +('nompi_' if (level == 'hadron' and not mpi)
                      else '')\
                    +(f'weight{weight:.1f}_'.replace('.', '-')
                      if weight != 1 else '')\
                    +(f'charged_' if charged else '')\
                    +(f'onesmeared_' if smeared else '')\
                    +f'jet0-80_subjet{r_sub:.2f}'.replace('.', '-')\
                    +'.py'
    else:
        file_name = f'{pair_obs}_lhc_w_ptmin{ptmin}_nomax_'\
                    +(f'weight{weight:.1f}_'.replace('.', '-') if weight != 1 else '')\
                    +(f'charged_' if charged else '')\
                    +(f'onesmeared_' if smeared else '')\
                    +f'jet0-80_subjet{r_sub:.2f}'.replace('.', '-')\
                    +'.py'

    if pair_obs.lower() in ['mmdt', 'sd2']:
        file_dir = property_data_dir
        file_name = f'groomed_{pair_obs.lower()}_zc1_lhc_w_'\
            +('highres120bins_' if high_res else '')\
            +('parton_' if level == 'parton' else '')\
            +('nompi_' if(level == 'hadron' and not mpi) else '')\
            +(f'charged_' if charged else '')\
            +(f'onesmeared_' if smeared else '')\
            +f'mass.py'
        variable = 'mass'

    hist1d = HistogramData(file_name=file_dir/file_name,
                           variable_order=[variable])

    vals, ewoc = hist1d.centers[variable], hist1d.hist

    # Initializing plotter
    if plot:
        plotter = EWOCPlotter(pair_obs,
                              plot_feature_locs,
                              plot_fontsizes, None)
        ax = plotter.axes[0]

        try:
            style = plot_style(**hist1d.metadata)
        except:
            style = {}

        ax.plot(vals, ewoc, label=rf'$r_\text{{sub}}={r_sub}$',
                **style)

        plotter.process_figure(LHC_W_PARAMS, show_w_guess=False,
                               legend=False, vline_fraction=0.8)

        xlim = (1e-2, 1e0) if pair_obs == 'deltaR' else (60, 110)
        ylim = (1e-1 if pair_obs == 'deltaR' else 1e-1,
                2 if pair_obs == 'deltaR' else 1e2)
        plotter.set_axes(pair_obs, xlim=xlim, ylim=ylim)

    if use_fit:
        # Plotting fit
        if pair_obs == 'deltaR':
            # lowval, highval
            l, h = 0.285, 0.315
        else:
            l, h = 79.0, 82.0

        x = vals[(vals > l) & (vals < h)]
        y = ewoc[(vals > l) & (vals < h)]
        try:
            peak, (a,b,c) = quadratic_peak_fit(x, y)
            energyhat = 14000/13
            peak = get_mass_from_obs(peak, pair_obs, energyhat)
            if plot:
                vals = np.linspace(l/2, 2*h, 1000)
                ax.plot(vals, a*vals**2+b*vals+c, alpha=0.3, **style)
                plt.show()
            plt.close()
            return peak
        except Exception as e:
            print(e)
            plt.close()
            return None

    try:
        peak = get_mass_from_ax(pair_obs, ax, 0,
                                return_width=return_width)
    except:
        peak = get_mass_from_ax('mass', ax, 0,
                                return_width=return_width)

    plt.show()
    plt.close()

    return peak


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
    for pair_obs, r_sub in zip(['mass', 'deltaR', 'mmdt'],
                               [0.3, 0.0, 0.0]):
        print("############################")
        print(f"# {pair_obs}")
        print("############################")
        smearshift = get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                   mpi=True, weight=1,
                                   smeared=True)\
                   - \
                   get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                 mpi=True, weight=1,
                                 smeared=False)
        print(f"{smearshift=}")


        hadshift = get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                 mpi=False, weight=1)\
                   - \
                   get_ewoc_peak(pair_obs, r_sub, level='parton',
                                 mpi=False, weight=1)
        print(f"{hadshift=}")

        mpishift = get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                 mpi=True, weight=1)\
                   - \
                   get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                 mpi=False, weight=1)
        print(f"{mpishift=}")

    pair_obs = 'mass'
    r_sub = 0.3
    for weight in [1, 2, 3]:
        print("############################")
        print(f"# mass, {weight=}")
        print("############################")
        smearshift = get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                   mpi=True, weight=weight,
                                   smeared=True)\
                   - \
                   get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                 mpi=True, weight=weight,
                                 smeared=False)
        print(f"{smearshift=}")

        hadshift = get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                 mpi=False, weight=weight)\
                   - \
                   get_ewoc_peak(pair_obs, r_sub, level='parton',
                                 mpi=False, weight=weight)
        print(f"{hadshift=}")

        mpishift = get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                 mpi=True, weight=weight)\
                   - \
                   get_ewoc_peak(pair_obs, r_sub, level='hadron',
                                 mpi=False, weight=weight)
        print(f"{mpishift=}")
