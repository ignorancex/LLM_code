#!/usr/bin/env python3
"""Plot the values of all existing measurements of ACP(D0 -> KS KS)."""

from argparse import ArgumentParser
import logging
from math import log
import matplotlib.pyplot as plt
import os
from utils import Measurement


def get_measures(comb):
    """Get the list of measurements to be shown for a given summary plot.

    :param comb: Identifier for a given summary plot.
    :type comb: str
    """
    measures = [
        Measurement("CLEO", "hep-ex/0012054", -23e-2, 19e-2),
        Measurement("LHCb 2015", "1508.06087", -2.9e-2, 5.2e-2, 2.2e-2),
        # Measurement("Belle", "1705.05966", -0.02e-2, 1.53e-2, 0.02e-2, 0.17e-2),
        Measurement("LHCb 2021", "2105.01565", -3.1e-2, 1.2e-2, 0.4e-2, 0.2e-2),
        Measurement("CMS", "2405.11606", 6.2e-2, 3.0e-2, 0.2e-2, 0.8e-2),
        Measurement("Belle + Belle II 2024", "2411.00306", -1.4e-2, 1.3e-2, 0.1e-2),
        # Measurement("World average 2024", None, -1.21e-2, 0.89e-2, 0.26e-2),
        Measurement("World average 2024", None, -1.67e-2, 0.84e-2, 0.22e-2),
    ]
    return measures


def get_units_label(units):
    raw_exp = log(units) / log(10)
    exp = round(raw_exp)
    if abs(exp - raw_exp) > 1e-2:
        raise RuntimeError(f'Units {units} not supported')
    return r'\%' if exp == -2 else f'10^{{{exp}}}'


def plot_average(date, out_dir):
    measures = get_measures(date)
    n_meas = len(measures)
    if n_meas == 0:
        logging.warning(f'No measurements available for combination {date}. Will not produce the plot')
        return

    # Axes and labels
    fig, ax = plt.subplots(figsize=(6, 6))
    (x_min, x_max), units = (-0.45, 0.6), 1e-2
    y_min, y_max = -0.5, n_meas - 0.5
    plt.xlim(x_min / units, x_max / units)
    plt.ylim(y_min, y_max)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    plt.xlabel(r'$A_{CP}(D^0 \to K^0_S K^0_S)$ $[\%]$', fontsize=24, ha='center')

    # Plot the measures and their numerical values
    x_text = max([meas.m + meas.err() for meas in measures]) + 0.05 * (x_max - x_min)
    for i in range(n_meas):
        meas = measures[i]
        plt.errorbar(meas.m / units,
                     n_meas - 1 - i,
                     xerr=meas.err() / units,
                     fmt='.',
                     markersize=8,
                     capsize=7,
                     color=meas.color)
        plt.errorbar(meas.m / units, n_meas - 1 - i, xerr=meas.stat / units, capsize=5, color=meas.color)
        y_text = n_meas - i - 1 - 0.25 * (n_meas / 6)
        plt.text(x_text / units, y_text, f'{meas.label}\n{meas.result_str(units)}', fontsize=18, color=meas.color)

    # Vertical and horizontal lines for world average
    if "World" in measures[-1].label:
        plt.plot([measures[-1].m / units, measures[-1].m / units], [y_min, y_max],
                 linestyle='--',
                 linewidth=1,
                 color='k')
        plt.plot([x_min / units, x_max / units], [0.5, 0.5], linestyle='-', linewidth=1, color='k')

    # Save figure
    os.makedirs(out_dir, exist_ok=True)
    fig_name = f'acp-d0-to-ksks-{date}'
    for ext in ['pdf']:
        plt.savefig(f'{out_dir}/{fig_name}.{ext}')


def parse_args():
    cwd = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()
    parser.add_argument('-c', '--date', type=str, default='2024', help='Date of the combination', choices=['2024'])
    parser.add_argument('-o',
                        '--outdir',
                        type=str,
                        default=os.path.join(cwd, 'figs'),
                        help='Output directory for saving the plots')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    cwd = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(cwd, 'lhcb.mplstyle'))
    args = parse_args()
    plot_average(args.date, args.outdir)
