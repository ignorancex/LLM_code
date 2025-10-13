# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
#
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified figure styling for all plots in the da-gp package."""

import matplotlib.pyplot as plt

from da_gp.logging_setup import get_logger

logger = get_logger(__name__)

# Publication-ready plot settings - JMLR 2001 style
try:
    from tueplots import bundles

    plt.rcParams.update(bundles.jmlr2001(nrows=1, ncols=2))
except ImportError:
    logger.warning("tueplots not available, using fallback matplotlib settings")
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 3.0),  # JMLR-style figure size
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.usetex": False,  # Avoid LaTeX dependency issues
            "font.family": "serif",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "grid.alpha": 0.3,
        }
    )

# Backend styling - consistent colors and markers across all plots
BACKEND_STYLES = {
    "sklearn": {"marker": "o", "color": "C0", "label": "Sklearn GP", "linestyle": "-"},
    "dapper_enkf": {"marker": "s", "color": "C1", "label": "EnKF", "linestyle": "-"},
    "dapper_letkf": {"marker": "^", "color": "C2", "label": "LETKF", "linestyle": "-"},
}

# Color-blind friendly palette (optional)
COLORBLIND_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def apply_colorblind_palette():
    """Apply color-blind friendly palette to backend styles."""
    for i, (backend, style) in enumerate(BACKEND_STYLES.items()):
        if i < len(COLORBLIND_COLORS):
            style["color"] = COLORBLIND_COLORS[i]


def setup_figure_style(colorblind_friendly=False):
    """Setup figure style with optional color-blind friendly palette.

    Args:
        colorblind_friendly: If True, use color-blind friendly colors
    """
    if colorblind_friendly:
        apply_colorblind_palette()

    return BACKEND_STYLES
