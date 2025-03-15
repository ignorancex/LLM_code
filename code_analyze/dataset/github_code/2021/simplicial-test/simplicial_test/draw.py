#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simplicial_test -- a python module to realize simplicial degree-size sequences
#
# Copyright (C) 2020-2021 Tzu-Chi Yen <tzuchi.yen@colorado.edu>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from .utils import flatten
import numpy as np

import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw_block(facets, limits=None, output=None, dpi=300, figsize=(8, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)

    if limits is not None:
        limit_sizes, limit_degs = limits
        g = np.zeros([limit_sizes, limit_degs])
    else:
        limit_sizes = limit_degs = np.infty
        g = np.zeros([len(facets), max(flatten(facets)) - min(flatten(facets)) + 1])

    for idx, facet in enumerate(facets):
        for vid in facet:
            if idx <= limit_sizes - 1 and vid <= limit_degs - 1:
                g[idx][vid] = 1

    plt.imshow(g, cmap='Greys', interpolation='nearest')
    plt.xlabel("Vertex index")
    plt.ylabel("Facet index")

    ax.tick_params(axis="y", direction="in", length=8)
    ax.tick_params(axis="x", direction="in", length=8)
    rc('xtick', labelsize=20)
    rc('ytick', labelsize=20)

    font = {'family': 'Sans',
            'weight': 'normal',
            'size': 22}

    rc('font', **font)

    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)


def draw_landscape(mat, output=None, dpi=300, ):
    max_sizes_ind = max_degs_ind = int(mat.size ** 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # setup the plot

    colors_undersea = plt.cm.tab20c(np.linspace(0, 0.1999, 256))
    colors_land = plt.cm.Wistia(np.linspace(0., 1., 256))
    all_colors = np.vstack((colors_undersea, colors_land))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('noname', all_colors)

    divnorm = colors.TwoSlopeNorm(vmin=min(mat.flatten()) - 0.5, vcenter=0, vmax=max(mat.flatten()) + 0.5)

    ims = ax.imshow(mat, norm=divnorm, cmap=cmap, origin='lower', extent=[1, max_sizes_ind, 1, max_degs_ind],
                    rasterized=True)
    plt.xlabel("Size sequence")
    plt.ylabel("Degree sequence")

    # scaled colorbar that aligns with the frame
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(ims, cax=cax, label="Failed attempts", shrink=0.6)
    ax.tick_params(axis="y", direction="in", length=8)
    ax.tick_params(axis="x", direction="in", length=8)
    ax.xaxis.set_ticks_position("bottom")

    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    # ax.set_xticks(np.arange(1.5, max_sizes_ind + 1, 5))
    # ax.set_yticks(np.arange(1.5, max_degs_ind + 1, 5))
    # ax.set_xticklabels(np.arange(0, max_sizes_ind + 1, 5))
    # ax.set_yticklabels(np.arange(0, max_degs_ind + 1, 5))
    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)
