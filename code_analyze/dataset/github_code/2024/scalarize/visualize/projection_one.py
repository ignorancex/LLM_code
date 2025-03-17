#!/usr/bin/env python3

r"""
A streamlit app for navigating a Pareto front using one-dimensional projections.
"""

import os

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import streamlit as st
import torch
from botorch.utils.transforms import normalize
from matplotlib.path import Path
from projection_utils import compute_optimal_points, textify_vector
from scalarize.utils.scalarization_functions import LengthScalarization


matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams[
    "text.latex.preamble"
] = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{color}"

tkwargs = {"dtype": torch.double, "device": "cpu"}

# Load the data-set.
# This is a dictionary containing:
# - data["Y"] : A (num_samples, num_points, M)-dim Tensor.
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data.pt")
data = torch.load(data_path)

#################################################################################
# Organise the data
#################################################################################
st.subheader(r"One-dimensional projection")

Y = data["Y"]

Y_max = Y.max(dim=0).values.max(dim=0).values
Y_min = Y.min(dim=0).values.min(dim=0).values
Y_range = Y_max - Y_min
bounds = torch.row_stack([Y_min, Y_max])

ref_point = Y_min - 0.2 * Y_range

normalized_Y = normalize(Y, bounds=bounds)
normalized_ref_point = normalize(ref_point, bounds=bounds)

num_samples = Y.shape[-3]
num_points = Y.shape[-2]
num_objectives = Y.shape[-1]

#################################################################################
# Generate the sliders
#################################################################################
st.sidebar.subheader("Weight selection")
weights = []
for m in range(num_objectives):
    w_m = st.sidebar.slider(
        rf"Weight {m+1}: ($w^{{({m+1})}} > 0$)",
        min_value=0.001,
        max_value=0.999,
        value=0.5,
        step=0.01,
    )
    weights += [w_m]

internal_weight = torch.tensor(weights, **tkwargs)
internal_weight = internal_weight / torch.norm(internal_weight)

st.sidebar.subheader("Positive unit vector")
st.sidebar.write(
    rf"$\boldsymbol{{\lambda}}"
    rf"= {textify_vector(internal_weight, 2)} \in \mathcal{{S}}_+^{{M-1}}$"
)
#################################################################################
# Compute sample statistics
#################################################################################
s_fn = LengthScalarization(weights=internal_weight, ref_points=normalized_ref_point)
normalized_lengths = s_fn(normalized_Y).max(dim=-2).values.squeeze(-1)

mean_vector = compute_optimal_points(
    length=normalized_lengths.mean(),
    ref_point=normalized_ref_point,
    weight=internal_weight,
    bounds=bounds,
)

qs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
quantile_colors = pl.cm.viridis(torch.linspace(0, 1, len(qs)))

quantiles = []
for q in qs:
    q_vector = compute_optimal_points(
        length=normalized_lengths.quantile(q),
        ref_point=normalized_ref_point,
        weight=internal_weight,
        bounds=bounds,
    ).squeeze(0)
    quantiles += [q_vector]

#################################################################################
# Plot parallel coordinates
#################################################################################
fig, ax_main = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=1)

y_labels = [rf"$y^{{({i+1})}}$" for i in range(num_objectives)]
vectors = torch.row_stack(quantiles + [mean_vector])
colours = [q for q in quantile_colors] + ["crimson"]

# Use the sample range as the objective scales
category = [i + 1 for i in range(len(qs) + 1)]
y_lower = Y_min - Y_range * 0.25
y_upper = Y_max + Y_range * 0.25
y_range = y_upper - y_lower

# Set-up axis.
axes = [ax_main] + [ax_main.twinx() for i in range(num_objectives - 1)]
for i, ax in enumerate(axes):
    ax.set_ylim(y_lower[i], y_upper[i])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if ax != ax_main:
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_ticks_position("right")
        ax.spines["right"].set_position(("axes", i / (num_objectives - 1)))

ax_main.set_xlim(0, num_objectives - 1)
ax_main.set_xticks(range(num_objectives))
ax_main.set_xticklabels(y_labels, fontsize=20)
ax_main.tick_params(axis="x", which="major", pad=7)
ax_main.spines["right"].set_visible(False)

# Transform all data to be compatible with the main axis.
transformed_vectors = torch.zeros_like(vectors)
transformed_vectors[:, 0] = vectors[:, 0]
transformed_vectors[:, 1:] = (
    y_lower[0] + (vectors[:, 1:] - y_lower[1:]) / y_range[1:] * y_range[0]
)

colors = plt.cm.tab10.colors
for j in range(len(transformed_vectors)):
    t = torch.linspace(0, num_objectives - 1, num_objectives * 3 - 2)
    vertices = list(
        zip([x for x in t], torch.repeat_interleave(transformed_vectors[j, :], 3)[1:-1])
    )
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(vertices) - 1)]
    path = Path(vertices, codes)
    patch = patches.PathPatch(
        path, facecolor="none", lw=3, edgecolor=colours[category[j] - 1]
    )
    ax_main.add_patch(patch)

expectation_text = (
    r"Expectation: $\mathbb{{E}}_{{\boldsymbol{{\omega}}}}"
    r"[\mathbf{{y}}_{\boldsymbol{\eta}, \boldsymbol{\lambda}}^*"
    r"(\boldsymbol{{\omega}})] \in \mathbb{R}^M$"
)

quantile_text = (
    r"Quantiles: $\mathcal{Q}_{{\boldsymbol{{\omega}}}}"
    r"[\mathbf{{y}}_{\boldsymbol{\eta}, \boldsymbol{\lambda}}^*"
    r"(\boldsymbol{{\omega}}), \alpha] \in \mathbb{R}^M$"
)

red_line = mlines.Line2D(
    [], [], color="crimson", linestyle="-", linewidth=3, label=expectation_text
)
green_line = mlines.Line2D(
    [], [], color=quantile_colors[5], linestyle="-", linewidth=3, label=quantile_text
)

ax_main.legend(
    handles=[red_line, green_line],
    ncol=2,
    fontsize=20,
    loc=(-0.025, -0.25),
    facecolor="k",
    framealpha=0.02,
)

plt.tight_layout()
st.pyplot(fig)
