#!/usr/bin/env python3
"""Produce the plot with the world average of the DeltaY parameter."""

from argparse import ArgumentParser
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
from utils import Measurement

measures = {
    "babar-2012": {
        "all": Measurement("BaBar 2012", "1209.3896", -8.8, 25.5, 5.8)
    },
    "cdf-2014": {
        "all": Measurement("CDF 2014", "1410.5435", 12., 12.),
        "kk": Measurement("CDF 2014", "1410.5435", 19., 15., 4.),
        "pipi": Measurement("CDF 2014", "1410.5435", 1., 18., 3.)
    },
    "belle-2016": {
        "all": Measurement("Belle 2016", "1509.08266", 3.0, 20.0, 7.0)
    },
    "lhcb-2015": {
        "all": Measurement(r"LHCb 2015 $\mu^{-}$ tag (3 fb$^{-1}$)", "1501.06777", 12.5, 7.3),
        "kk": Measurement(r"LHCb 2015 $\mu^{-}$ tag (3 fb$^{-1}$)", "1501.06777", 13.4, 7.7, 2.6),
        "pipi": Measurement(r"LHCb 2015 $\mu^{-}$ tag (3 fb$^{-1}$)", "1501.06777", 9.2, 14.5, 2.5),
    },
    "lhcb-2017": {
        "all": Measurement("LHCb 2017 $D^{*+}$ tag (3 fb$^{-1}$)", "1702.06490", 1.3, 2.8, 1.0),
        "kk": Measurement("LHCb 2017 $D^{*+}$ tag (3 fb$^{-1}$)", "1702.06490", 3.0, 3.2, 1.0),
        "pipi": Measurement("LHCb 2017 $D^{*+}$ tag (3 fb$^{-1}$)", "1702.06490", -4.6, 5.8, 1.2),
    },
    "lhcb-2019": {  # CONF note
        "all": Measurement("LHCb 2019 $D^{*+}$ tag (2 fb$^{-1}$)", "", -3.4, 3.1, 0.6),
        "kk": Measurement("LHCb 2019 $D^{*+}$ tag (2 fb$^{-1}$)", "", -1.3, 3.5, 0.7),
        "pipi": Measurement("LHCb 2019 $D^{*+}$ tag (2 fb$^{-1}$)", "", -11.3, 6.9, 0.8),
    },
    "lhcb-2020": {
        "all": Measurement(r"LHCb 2020 $\mu^{-}$ tag (5.4 fb$^{-1}$)", "1911.01114", 3.3, 3.6, 0.3),
        # "all": Measurement(r"LHCb 2020 $\mu^{-}$ tag (5.4 fb$^{-1}$)", "1911.01114", 2.9, 3.2, 0.5),
        "kk": Measurement(r"LHCb 2020 $\mu^{-}$ tag (5.4 fb$^{-1}$)", "1911.01114", 4.8, 4.0, 0.3),
        "pipi": Measurement(r"LHCb 2020 $\mu^{-}$ tag (5.4 fb$^{-1}$)", "1911.01114", -2.5, 7.8, 0.3),
    },
    "lhcb-2021": {
        "all": Measurement("LHCb 2021 $D^{*+}$ tag (6 fb$^{-1}$)", "2105.09889", -2.70, 1.34, 0.30),
        "kk": Measurement("LHCb 2021 $D^{*+}$ tag (6 fb$^{-1}$)", "2105.09889", -2.32, 1.52, 0.32),
        "pipi": Measurement("LHCb 2021 $D^{*+}$ tag (6 fb$^{-1}$)", "2105.09889", -4.01, 2.81, 0.39),
    },
    "lhcb-2024": {
        "all": Measurement(r"LHCb 2024 $D^0\to\pi^{+}\pi^{-}\pi^{0}$ (6 fb$^{-1}$)", "2405.06556", -1.3, 6.3, 2.4)
    },
    "lhcb-avg-2020": {
        "all": Measurement("LHCb average", None, -1.0, 1.1, 0.3)
    },
    "lhcb-avg-2021": {
        "all": Measurement(r"LHCb 2024 $D^0\to\pi^{+}\pi^{-}\pi^{0}$ (6 fb$^{-1}$)", "2405.06556", -1.3, 6.3, 2.4)
    },
    "lhcb-avg-2024": {
        "all": Measurement("LHCb average", None, -1.09, 1.11, 0.32),
        "kk": Measurement("LHCb average", None, -0.35, 1.28, 0.32),
        "pipi": Measurement("LHCb average", None, -3.61, 2.38, 0.40),
    },
    "wa-2020": {
        "all": Measurement("World average", None, 1.9, 1.6, 0.5)
    },
    "wa-2021": {
        "all": Measurement("World average", None, -0.9, 1.1, 0.3)
    },
    "wa-2024": {
        "all": Measurement("World average", None, -0.97, 1.11, 0.32),
        "kk": Measurement("World average", None, -0.20, 1.28, 0.32),
        "pipi": Measurement("World average", None, -3.53, 2.36, 0.39),
    },
}


def get_measures(comb, final_state="all"):
    """Get the list of measurements to be shown for a given summary plot.

    :param comb: Identifier for a given summary plot.
    :type comb: str
    :type final_state: str
    """
    comb_measures = []
    if "lhcb" not in comb:
        if final_state == "all":
            comb_measures.append(measures["babar-2012"]["all"])
        comb_measures.append(measures["cdf-2014"][final_state])
    comb_measures.append(measures["lhcb-2015"][final_state])
    if "lhcb" not in comb and final_state == "all":
        comb_measures.append(measures["belle-2016"]["all"])
    comb_measures.extend([
        measures["lhcb-2017"][final_state],
        measures["lhcb-2020"][final_state],
        measures["lhcb-2021"][final_state],
    ])
    if "2024" in comb and final_state == "all":
        comb_measures.append(measures["lhcb-2024"]["all"])
    comb_measures.append(measures[comb][final_state])  # World/LHCb average
    return comb_measures


def get_xy_ranges(comb, final_state, dy_notation):
    """Get the x and y ranges for a given summary plot."""
    n_meas = len(get_measures(comb, final_state))
    if "lhcb" in comb:
        x_min_lhcb = {
            "all": -10. if dy_notation else -23,
            "kk": -10. if dy_notation else -25,
            "pipi": -16. if dy_notation else -27,
        }
        x_max_lhcb = {
            "all": 70 if dy_notation else 60,
            "kk": 60 if dy_notation else 40,
            "pipi": 68 if dy_notation else 60,
        }
        x_min = x_min_lhcb[final_state]
        x_max = x_max_lhcb[final_state]
    else:
        x_min_wa = {
            "all": -40. if dy_notation else -27,
            "kk": -10. if dy_notation else -40,
            "pipi": -20. if dy_notation else -27,
        }
        x_max_wa = {
            "all": (120 if dy_notation else 120) if "2024" in comb else (90 if dy_notation else 70),
            "kk": 80 if dy_notation else 55,
            "pipi": 73 if dy_notation else 65,
        }
        x_min = x_min_wa[final_state]
        x_max = x_max_wa[final_state]
    y_min, y_max = -0.5, n_meas - 0.5
    return (x_min, x_max), (y_min, y_max)


def get_x_text(comb, final_state, dy_notation):
    if dy_notation:
        if "lhcb" in comb:
            return 30 if final_state == "pipi" else 26
        else:
            return 40 if final_state == "kk" else 30
    else:
        x_text_lhcb = {"all": 15, "kk": 10, "pipi": 20}
        x_text_wa = {"all": 40, "kk": 10, "pipi": 22}
        return x_text_lhcb[final_state] if "lhcb" in comb else x_text_wa[final_state]


def make_plot(comb, final_state, dy_notation, out_dir):
    if comb == "lhcb":
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
    observable = r"\Delta Y" if dy_notation else r"A_{\Gamma}"
    decay_label = "" if final_state == "all" else r"(D^0 \to K^+ K^-)" if final_state == "kk" else r"(D^0 \to \pi^+ \pi^-)"
    plt.xlabel(f"${observable}{decay_label}\\;[10^{{-4}}]$", fontsize=24, ha='center')
    (x_min, x_max), (y_min, y_max) = get_xy_ranges(comb, final_state, dy_notation)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)

    measures = get_measures(comb, final_state)
    n_meas = len(measures)
    for i in range(n_meas):
        meas = measures[i]
        if not dy_notation:
            meas.m = -meas.m
        plt.errorbar(meas.m,
                     n_meas - 1 - i,
                     xerr=meas.err(),
                     fmt='.',
                     markersize=8,
                     capsize=7,
                     color=meas.color if comb != "lhcb" else 'k')
        plt.errorbar(meas.m, n_meas - 1 - i, xerr=meas.stat, capsize=7, color=meas.color if comb != "lhcb" else 'k')
        x_text = get_x_text(comb, final_state, dy_notation)
        y_text = n_meas - i - 1.25
        plt.text(x_text, y_text, f"{meas.label}\n{meas.result_str()}", color=meas.color if comb != "lhcb" else 'k')
        if i < n_meas - 1 and False:  # TODO
            arxiv_fontsize = 15
            default_fontsize = matplotlib.rcParams["font.size"]
            plt.text(x_text,
                     y_text - 0.006 * (default_fontsize + arxiv_fontsize),
                     f"arXiv:{meas.arxiv}",
                     fontsize=arxiv_fontsize,
                     color="deepskyblue")

    # Vertical line for world average
    plt.plot([measures[-1].m, measures[-1].m], [y_min, y_max], linestyle='--', linewidth=1, color='k')

    # Horizontal line to divide world average from other measurements
    plt.plot([x_min, x_max], [0.5, 0.5], linestyle='-', linewidth=1, color='k')

    # Save figure
    os.makedirs(out_dir, exist_ok=True)
    obs = 'dy' if dy_notation else 'agamma'
    fig_name = f'{obs}-{comb}'
    if final_state != "all":
        fig_name += f"-{final_state}"
    for ext in ['pdf']:
        plt.savefig(f'{out_dir}/{fig_name}.{ext}')


def parse_args():
    cwd = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()
    parser.add_argument('-c',
                        '--comb',
                        type=str,
                        default='wa-2024',
                        help='Combination id',
                        choices=['wa-2020', 'wa-2021', 'lhcb-avg-2021', 'wa-2024', 'lhcb-avg-2024'])
    parser.add_argument('--fs',
                        type=str,
                        default="all",
                        help="Final state for the combination",
                        choices=["all", "kk", "pipi"])
    parser.add_argument('--agamma',
                        action='store_false',
                        dest='dy_notation',
                        help='Use the A_Gamma notation rather than DeltaY')
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
    make_plot(args.comb, args.fs, args.dy_notation, args.outdir)
