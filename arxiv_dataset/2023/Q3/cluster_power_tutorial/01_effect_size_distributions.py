#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy
import matplotlib
from matplotlib import pyplot


# # # # #
# SETTINGS

n_repeats = 100
d_min = 0.0
d_max = 2.4
delta_min = 4.0
w_range = [0.75, 1.5, 3.0, 6.0, 12.0]
w_label = { \
    0.75: "(wildly optimistic)", \
    1.5: r"($\approx$Szucs & Ioannidis, 2017)", \
    3.0: "(none to very large)", \
    6.0: "(none to large)", \
    12.0: "(none to medium)", \
    }

# Files and folders.
this_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(this_dir, "output")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


# # # # #
# EFFECT SIZE FIGURE

# Probability over effect sizes roughly matches the empirical distribution in
# Szucs D, Ioannidis JPA (2017) Empirical assessment of published effect sizes 
#   and power in the recent cognitive neuroscience and psychology literature. 
#   PLOS Biology 15(3): e2000797. https://doi.org/10.1371/journal.pbio.2000797
#
# Create an effect size array.
d_step = 0.01
d = numpy.arange(d_min, d_max+d_step, d_step)
# Create an array with an increasing number of parameters.
p = numpy.logspace(0, 3.5, 31, base=10).astype(numpy.int32)
p = numpy.unique(p)
# Create empty arrays for the probability density of effect sizes in each
# context, and for the expected subgroup effect size in each context.
pdf = numpy.zeros((len(w_range), d.shape[0]), dtype=numpy.float64)
delta = numpy.zeros((len(w_range), p.shape[0]), dtype=numpy.float64)

# Loop through all contexts.
for wi, w in enumerate(w_range):
    # Probability density function for an exponential distribution.
    pdf[wi,:] = w * numpy.exp(-w*d)
    # The mean feature effect size is given by 1/lambda, which we can use to
    # compute the expected cluster effect size.
    delta[wi, :] = numpy.sqrt(p * (1.0/w)**2)

# Create a colour map.
cmap = matplotlib.cm.get_cmap("viridis_r")
w_min = numpy.log(numpy.min(w_range))
w_max = numpy.log(numpy.max(w_range))
vmin = w_min - (w_max-w_min) * 0.1
vmax = w_max + (w_max-w_min) * 0.1
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# Create a new figure.
fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16.0,6.0), dpi=300.0)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.1, top=0.92, wspace=0.2)

# Draw the minimally required cluster separation.
axes[1].plot(p, numpy.ones(p.shape)*delta_min, ":", alpha=0.5, color="#000000")
axes[1].annotate( \
    "Minimally required cluster separation\n(Dalmaijer et al., 2022)", \
    (p[0]*1.1, delta_min*0.8), fontsize=12, alpha=0.5, \
    color="#000000")

# Loop through all potential effect size distributions.
for wi, w in enumerate(w_range):
    
    # Get the current colour.
    col = cmap(norm(numpy.log(w)))
    # Construct a label for the current line.
    lbl = r"$\lambda=" + str(round(w, 2)).ljust(3, "0") + r"$"
    if w in w_label.keys():
        lbl += " " + w_label[w]
    
    # Plot the effects size distribution.
    axes[0].plot(d, pdf[wi,:], "-", lw=3, color=col, label=lbl)
    # Finish the plot.
    axes[0].legend(loc="upper right", fontsize=12)
    axes[0].set_title("Distribution of underlying effect sizes ($\delta$)", \
        fontsize=18)
    axes[0].set_xlabel(r"Single-feature effect size, $|\delta|$", fontsize=16)
    axes[0].set_ylabel("Probability density", fontsize=16)
    axes[0].set_xlim(numpy.min(d), numpy.max(d))
    xticks = [0.0, 0.2, 0.5, 0.8, 1.2, 2.0]
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(numpy.round(xticks,1), fontsize=12)
    axes[0].set_ylim(0.0, 10.0)
    yticks = numpy.linspace(0, 10, 6)
    axes[0].set_yticks(yticks)
    # axes[0].set_yticklabels(numpy.round(yticks,2), fontsize=12)
    axes[0].set_yticklabels(yticks.astype(numpy.int64), fontsize=12)

    # Plot the cluster separation as a function of variable.
    axes[1].plot(p, delta[wi,:], lw=3, color=col)
    # Finish the plot.
    axes[1].set_title("Cumulative effect size ($\Delta$)", fontsize=18)
    axes[1].set_xlabel("Number of included variables", fontsize=16)
    axes[1].set_xscale("log")
    axes[1].set_xlim(numpy.min(p), numpy.max(p))
    xticks = [1, 10, 25, 50, 100, 250, 500, 1000, 2500]
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(list(map(str, xticks)), fontsize=12)
    axes[1].set_ylabel(r"Centroid separation, $\Delta$", fontsize=16)
    ylim = (0, 10)
    axes[1].set_ylim(ylim)
    yticks = numpy.linspace(ylim[0], ylim[1], 6)
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(numpy.round(yticks,2), fontsize=12)

# Save and close the figure.
fig.savefig(os.path.join(out_dir, \
    "fig-02_single-feature_and_cluster_effect_size.png"))
pyplot.close(fig)
    