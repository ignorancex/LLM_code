"""
Plot test-statistic distribution
================================
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

from stats import pdf_half_chi2
from data import n_channels


if __name__ == "__main__":

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette = itertools.cycle(colors)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', **{'family':'serif', 'size': 18})
    plt.subplots(figsize=(7, 5))

    test_statistic_global = np.load("data/test_statistic_global.npy")
    test_statistic_local = np.load("data/test_statistic_local.npy")

    c = next(palette)
    plt.hist(test_statistic_local, bins='auto', density=True,
             histtype="stepfilled", alpha=0.5, color=c,
             label=r"Fixed mass, $m=151\,\text{GeV}$")
    plt.hist(test_statistic_global, bins='auto', density=True,
             histtype="step", color=next(palette), lw=3,
             label="Floating mass")

    x = np.linspace(0, 25, 10000)
    plt.plot(x, pdf_half_chi2(x, 1), color=next(palette), lw=2,
             label=r"$\frac12 \chi^2$ --- used in ref.~[1]", zorder=-1)
    plt.plot(x, pdf_half_chi2(x, n_channels), lw=2, color=c,
             label=r"$\frac12 \chi^2_6$", zorder=-1)

    plt.xlabel("Test-statistic, $\lambda$")
    plt.ylabel("PDF")
    plt.yticks([])
    plt.ylim(0, 0.5)
    plt.xlim(-1, 20)

    plt.legend(fancybox=False)
    plt.tight_layout()
    plt.savefig("dist.pdf")
