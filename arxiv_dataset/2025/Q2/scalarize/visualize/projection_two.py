#!/usr/bin/env python3

r"""
A streamlit app for navigating a Pareto front using two-dimensional projections.
"""

import os

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import streamlit as st
import torch
from botorch.utils.transforms import normalize
from projection_utils import (
    compute_optimal_points,
    compute_projected_vector,
    textify_vector,
)
from scalarize.utils.scalarization_functions import LengthScalarization
from scalarize.utils.scalarization_parameters import UnitVector


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
st.subheader(r"Two-dimensional projection")

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

st.sidebar.subheader("Objective pair")

# Pairwise dropdowns.
objective_number = {}
for m in range(num_objectives):
    label = f"Objective {m+1}"
    objective_number[label] = m
all_objectives = list(objective_number.keys())

dropdown_one = st.sidebar.selectbox(
    "First objective",
    all_objectives,
)

remaining_objectives = []
for obj in all_objectives:
    if obj != dropdown_one:
        remaining_objectives += [obj]

dropdown_two = st.sidebar.selectbox(
    "Second objective",
    remaining_objectives,
)

# Partial vector.
st.sidebar.subheader("Positive unit vector")
st.sidebar.write(
    rf"$\boldsymbol{{\lambda}}"
    rf"= {textify_vector(internal_weight, 2)} \in \mathcal{{S}}_+^{{M-1}}$"
)

variable_indices = []
fixed_indices = []
for m in range(num_objectives):
    label = all_objectives[m]
    if label != dropdown_one and label != dropdown_two:
        fixed_indices += [m]
    else:
        variable_indices += [m]
fixed_vector = compute_projected_vector(internal_weight, indices=fixed_indices).squeeze(
    0
)

st.sidebar.subheader("Fixed vector")
st.sidebar.write(
    rf"$\mathbf{{v}}"
    rf"= {textify_vector(fixed_vector, 2)} \in \mathbb{{R}}_{{>0}}^{{M-2}}$"
)
#################################################################################
# Compute sample statistics
#################################################################################

# Get two-dimensional weights.
num_weights = 201
t = torch.linspace(0, 1, num_weights, **tkwargs).unsqueeze(-1)
unit_vector = UnitVector(num_objectives=2, transform_label="polar")
weights = unit_vector(t)

# Concatenate the weights using two pointers.
all_weights = []
ones = torch.ones(num_weights, **tkwargs)
c = torch.sqrt(1 - torch.sum(fixed_vector**2))
i = 0
j = 0
for m in range(num_objectives):
    if m in fixed_indices:
        all_weights += [fixed_vector[..., i] * ones]
        i += 1
    else:
        all_weights += [weights[..., j] * c]
        j += 1
all_weights = torch.column_stack(all_weights)

# Compute statistics.
s_fn = LengthScalarization(weights=all_weights, ref_points=normalized_ref_point)
normalized_lengths = s_fn(normalized_Y).max(dim=-2).values.squeeze(-1)

mean_front = compute_optimal_points(
    length=normalized_lengths.mean(dim=0),
    ref_point=normalized_ref_point,
    weight=all_weights,
    bounds=bounds,
)

mean_front = compute_projected_vector(mean_front, indices=variable_indices)

qs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
quantile_colors = pl.cm.viridis(torch.linspace(0, 1, len(qs)))

quantiles = []
for q in qs:
    q_front = compute_optimal_points(
        length=normalized_lengths.quantile(q, dim=0),
        ref_point=normalized_ref_point,
        weight=all_weights,
        bounds=bounds,
    )
    q_front = compute_projected_vector(q_front, indices=variable_indices)
    quantiles += [q_front]

#################################################################################
# Plot 2D curves
#################################################################################
fig, ax = plt.subplots(figsize=(10, 10))

for i, q_front in enumerate(quantiles):
    ax.plot(q_front[:, 0], q_front[:, 1], color=quantile_colors[i], linewidth=3)

ax.plot(mean_front[:, 0], mean_front[:, 1], color="crimson", linewidth=3, zorder=3)
ax.tick_params(axis="both", which="major", labelsize=15)


expectation_text = (
    r"Expectation: $\mathbb{{E}}_{{\boldsymbol{{\omega}}}}"
    r"[\mathcal{P}_{I, \mathbf{v}} "
    r"[Y_{\boldsymbol{\eta}, f}^*(\boldsymbol{{\omega}})]"
    r"] \subset \mathbb{R}^2$"
)

quantile_text = (
    r"Quantiles: $\mathcal{Q}_{{\boldsymbol{{\omega}}}}"
    r"[\mathcal{P}_{I, \mathbf{v}} "
    r"[Y_{\boldsymbol{\eta}, f}^*(\boldsymbol{{\omega}})], \alpha"
    r"] \subset \mathbb{R}^2$"
)

red_line = mlines.Line2D(
    [], [], color="crimson", linestyle="-", linewidth=3, label=expectation_text
)
green_line = mlines.Line2D(
    [], [], color=quantile_colors[5], linestyle="-", linewidth=3, label=quantile_text
)

ax.legend(
    handles=[red_line, green_line],
    ncol=1,
    fontsize=20,
    loc=(0.175, -0.3),
    facecolor="k",
    framealpha=0.02,
)
ax.set_xlabel(rf"$y^{{({(objective_number[dropdown_one] + 1)})}}$", fontsize=20)
ax.set_ylabel(rf"$y^{{({(objective_number[dropdown_two] + 1)})}}$", fontsize=20)
plt.show()
st.pyplot(fig)
