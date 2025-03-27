#!/usr/bin/env python3
"""Plot the values of all existing measurements of ACP and BF of D(s)+ -> eta(') h+ decays,
plus the PDG 2024 average.
"""

from argparse import ArgumentParser
import itertools
import logging
from math import log
import matplotlib.pyplot as plt
import os
from utils import Measurement


def get_measures(meas_type, decay, date):
    """Get the list of measurements to be shown for a given summary plot.

    :param meas_type: 'BF' or 'ACP'
    :param decay: Decay identifier.
    :param date: Date identifier for a given summary plot.
    """

    measures = {
        'BF': {
            'dp-to-etapi': [
                Measurement(
                    'CLEO', '0906.3198', 3.63e-3, 0.08e-3, 0.18e-3, 0.06e-3
                ),  # Using PDG 2024 for D+ -> K- pi+ pi+; original values (3.54e-3, 0.08e-3, 0.18e-3, 0.08e-3)
                Measurement('BES III', '1802.03119', 3.790e-3, 0.070e-3, 0.068e-3),
                Measurement('PDG 2024', None, 3.77e-3, 0.09e-3),
            ],
            'dp-to-etappi': [
                Measurement(
                    'CLEO', '0906.3198', 4.80e-3, 0.16e-3, 0.23e-3, 0.08e-3
                ),  # Using PDG 2024 for D+ -> K- pi+ pi+; original values (4.68e-3, 0.16e-3, 0.23e-3, 0.10e-3)
                Measurement('BES III', '1802.03119', 5.12e-3, 0.14e-3,
                            0.02e-3),  # TODO typo in the PDG result (one missing zero in sys. unc.)
                Measurement('PDG 2024', None, 4.97e-3, 0.19e-3),
            ],
            'dp-to-etak': [
                Measurement('Belle', '1107.0553', 1.15e-4, 0.16e-4, 0.05e-4, 0.03e-4),
                Measurement('BES III', '1802.03119', 1.51e-4, 0.25e-4, 0.14e-4),
                Measurement('PDG 2024', None, 1.25e-4, 0.16e-4),
            ],
            'dp-to-etapk': [
                Measurement('Belle', '1107.0553', 1.87e-4, 0.19e-4, 0.05e-4,
                            0.07e-4),  # Using PDG 2024 for D+ -> eta' pi
                Measurement('BES III', '1802.03119', 1.64e-4, 0.51e-4, 0.24e-4),
                Measurement('PDG 2024', None, 1.85e-4, 0.20e-4),
            ],
            'ds-to-etapi': [
                Measurement('CLEO', 'hep-ex/9705006', 1.2e-2, 0.3e-2, 0.2e-2,
                            0.2e-2),  # Using PDG 2024 for D+ -> omega pi+
                Measurement('CLEO', '1306.5363', 1.67e-2, 0.08e-2, 0.06e-2),
                Measurement('Belle', '1307.6240', 1.82e-2, 0.14e-2, 0.07e-2),
                Measurement(
                    'BES III', '2005.05072', 1.715e-2, 0.018e-2, 0.026e-2, 0.032e-2
                ),  # Using PDG 2024 for D+ -> K- K+ pi+; original values (1.741e-2, 0.018e-2, 0.027e-2, 0.054e-2)
                Measurement(
                    'Belle', '2103.09969', 1.874e-2, 0.010e-2, 0.058e-2, 0.051e-2
                ),  # Using PDG 2024 for D+ -> phi pi+; original values (1.900e-2, 0.010e-2, 0.059e-2, 0.068e-2)
                Measurement('PDG 2024', None, 1.67e-2, 0.09e-2),
            ],
            'ds-to-etappi': [
                Measurement('CLEO', '1306.5363', 3.94e-2, 0.15e-2, 0.20e-2),
                Measurement('BES III', '2005.05072', 37.8e-3, 0.4e-3, 2.1e-3, 1.2e-3),
                Measurement('PDG 2024', None, 3.94e-2, 0.25e-2),
            ],
            'ds-to-etak': [
                Measurement('CLEO', '0906.3198', 1.76e-3, 0.33e-3, 0.09e-3, 0.10e-3),
                Measurement('BES III', '2005.05072', 1.62e-3, 0.10e-3, 0.03e-3, 0.05e-3),
                Measurement(
                    'Belle', '2103.09969', 1.73e-3, 0.05e-3, 0.05e-3,
                    0.05e-3),  # Using PDG 2024 for D+ -> phi pi+; original values (1.75e-3, 0.05e-3, 0.05e-3, 0.06e-3)
                Measurement('PDG 2024', None, 1.73e-3, 0.08e-3),
            ],
            'ds-to-etapk': [
                Measurement('CLEO', '0906.3198', 1.8e-3, 0.5e-3, 0.1e-3, 0.1e-3),
                Measurement('BES III', '2005.05072', 2.68e-3, 0.17e-3, 0.17e-3, 0.08e-3),
                Measurement('PDG 2024', None, 2.64e-3, 0.24e-3),
            ],
        },
        'ACP': {
            'dp-to-etapi': [
                Measurement('CLEO', '0906.3198', -2.0e-2, 2.3e-2, 0.3e-2),
                Measurement('Belle', '1107.0553', 1.74e-2, 1.13e-2, 0.19e-2),
                Measurement(r'LHCb $\eta\to\gamma\gamma$, 6 fb$^{-1}$', '2103.11058', -0.2e-2, 0.8e-2, 0.4e-2),
                Measurement(r'LHCb $\eta\to\pi^+\pi^-\gamma$, 6 fb$^{-1}$', '2204.12228', 3.4e-3, 6.6e-3, 1.6e-3,
                            0.5e-3),
                Measurement('PDG 2024', None, 3e-3, 5e-3),
            ],
            'dp-to-etappi': [
                Measurement('CLEO', '0906.3198', -4.0e-2, 3.4e-2, 0.3e-2),
                Measurement('Belle', '1107.0553', -0.12e-2, 1.12e-2, 0.17e-2),
                Measurement(r'LHCb $\eta^{\prime}\to\pi^+\pi^-\gamma$, 3 fb$^{-1}$', '1701.01871', -6.1e-3, 7.2e-3,
                            5.3e-3, 1.2e-3),
                Measurement(r'LHCb $\eta^{\prime}\to\pi^+\pi^-\gamma$, 6 fb$^{-1}$', '2204.12228', 4.9e-3, 1.8e-3,
                            0.6e-3, 0.5e-3),
                Measurement('PDG 2024', None, 4.1e-3, 2.3e-3),
            ],
            'dp-to-etak': [
                Measurement(r'LHCb $\eta\to\gamma\gamma$, 6 fb$^{-1}$', '2103.11058', -6e-2, 10e-2, 4e-2),
                Measurement('PDG 2024', None, -6e-2, 11e-2),
            ],
            'dp-to-etapk': [],
            'ds-to-etapi': [
                Measurement('CLEO', '1306.5363', 1.1e-2, 3.0e-2, 0.8e-2),
                Measurement('Belle', '2103.09969', 2e-3, 3e-3, 3e-3),
                Measurement(r'LHCb $\eta\to\gamma\gamma$, 6 fb$^{-1}$', '2103.11058', 0.8e-2, 0.7e-2, 0.5e-2),
                Measurement(r'LHCb $\eta\to\pi^+\pi^-\gamma$, 6 fb$^{-1}$', '2204.12228', 3.2e-3, 5.1e-3, 1.2e-3),
                Measurement('PDG 2024', None, 3.2e-3, 3.1e-3),
            ],
            'ds-to-etappi': [
                Measurement('CLEO', '1306.5363', -2.2e-2, 2.2e-2, 0.6e-2),
                Measurement(r'LHCb $\eta^{\prime}\to\pi^+\pi^-\gamma$, 3 fb$^{-1}$', '1701.01871', -8.2e-3, 3.6e-3,
                            2.2e-3, 2.7e-3),
                Measurement(r'LHCb $\eta^{\prime}\to\pi^+\pi^-\gamma$, 6 fb$^{-1}$', '2204.12228', 0.1e-3, 1.2e-3,
                            0.8e-3),
                Measurement('PDG 2024', None, -0.6e-3, 2.2e-3),
            ],
            'ds-to-etak': [
                Measurement('CLEO', '0906.3198', 9.3e-2, 15.2e-2, 0.9e-2),
                Measurement('Belle', '2103.09969', 2.1e-2, 2.1e-2, 0.4e-2),
                Measurement(r'LHCb $\eta\to\gamma\gamma$, 6 fb$^{-1}$', '2103.11058', 0.9e-2, 3.7e-2, 1.1e-2),
                Measurement('PDG 2024', None, 1.8e-2, 1.9e-2),
            ],
            'ds-to-etapk': [
                Measurement('CLEO', '0906.3198', 6.0e-2, 18.9e-2, 0.9e-2),
                Measurement('PDG 2024', None, 6e-2, 19e-2),
            ],
        },
    }

    if date == '2024':
        return measures[meas_type][decay]
    else:
        raise RuntimeError(f'The combination for {date} is not supported')


def get_decay_label(decay):
    dp = 'D^+'
    ds = 'D^+_s'
    eta = r'\eta'
    etap = r'\eta^{\prime}'
    pi = r'\pi^+'
    k = 'K^+'

    lab = dp if 'dp' in decay else ds
    lab += r' \to '
    lab += etap if any(e in decay for e in ['etapp', 'etapk']) else eta
    lab += ' '
    lab += k if 'k' in decay else pi
    return lab


def get_xrange_and_units(meas_type, decay, date):
    """Get the x range and the units for a given summary plot."""
    xranges = {
        'BF': {
            'dp-to-etapi': (3.3e-3, 4.9e-3),
            'dp-to-etappi': (4.4e-3, 6.5e-3),
            'dp-to-etak': (0.8e-4, 3.2e-4),
            'dp-to-etapk': (1e-4, 3.8e-4),
            'ds-to-etapi': (0.7e-2, 3.6e-2),
            'ds-to-etappi': (3.4e-2, 6e-2),
            'ds-to-etak': (1.3e-3, 3.3e-3),
            'ds-to-etapk': (1e-3, 5.5e-3),
        },
        'ACP': {
            'dp-to-etapi': (-5e-2, 15e-2),
            'dp-to-etappi': (-8e-2, 13e-2),
            'dp-to-etak': (-20e-2, 30e-2),
            'dp-to-etapk': (-10e-2, 10e-2),
            'ds-to-etapi': (-2.5e-2, 13.5e-2),
            'ds-to-etappi': (-5e-2, 7.5e-2),
            'ds-to-etak': (-8e-2, 50e-2),
            'ds-to-etapk': (-15e-2, 55e-2),
        },
    }
    units = {
        'BF': {
            'dp-to-etapi': 1e-3,
            'dp-to-etappi': 1e-3,
            'dp-to-etak': 1e-4,
            'dp-to-etapk': 1e-4,
            'ds-to-etapi': 1e-2,
            'ds-to-etappi': 1e-2,
            'ds-to-etak': 1e-3,
            'ds-to-etapk': 1e-3,
        },
        'ACP': {
            'dp-to-etapi': 1e-2,
            'dp-to-etappi': 1e-2,
            'dp-to-etak': 1e-2,
            'dp-to-etapk': 1e-2,
            'ds-to-etapi': 1e-2,
            'ds-to-etappi': 1e-2,
            'ds-to-etak': 1e-2,
            'ds-to-etapk': 1e-2,
        },
    }
    return xranges[meas_type][decay], units[meas_type][decay]


def get_units_label(units):
    raw_exp = log(units) / log(10)
    exp = round(raw_exp)
    if abs(exp - raw_exp) > 1e-2:
        raise RuntimeError(f'Units {units} not supported')
    return r'\%' if exp == -2 else f'10^{{{exp}}}'


def plot_average(meas_type, decay, date, out_dir):

    measures = get_measures(meas_type, decay, date)
    n_meas = len(measures)
    if n_meas == 0:
        logging.warning(f'No measurements available for {meas_type} of {decay}. Will not produce the plot')
        return

    # Axes and labels
    fig, ax = plt.subplots(figsize=(6, 6))
    (x_min, x_max), units = get_xrange_and_units(meas_type, decay, date)
    y_min, y_max = -0.5, n_meas - 0.5
    plt.xlim(x_min / units, x_max / units)
    plt.ylim(y_min, y_max)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    meas_label = 'BF' if meas_type == 'BF' else 'A_{CP}'
    plt.xlabel(rf'${meas_label}({get_decay_label(decay)})$ $[{get_units_label(units)}]$', fontsize=24, ha='center')

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

    # Vertical line for world average
    plt.plot([measures[-1].m / units, measures[-1].m / units], [y_min, y_max], linestyle='--', linewidth=1, color='k')

    # Horizontal line to divide world average from other measurements
    plt.plot([x_min / units, x_max / units], [0.5, 0.5], linestyle='-', linewidth=1, color='k')

    # Save figure
    os.makedirs(out_dir, exist_ok=True)
    fig_name = f'{meas_type.lower()}-{decay}-{date}'
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

    for m, d, e, h in itertools.product(['BF', 'ACP'], ['dp', 'ds'], ['eta', 'etap'], ['pi', 'k']):
        plot_average(m, f'{d}-to-{e}{h}', args.date, args.outdir)
