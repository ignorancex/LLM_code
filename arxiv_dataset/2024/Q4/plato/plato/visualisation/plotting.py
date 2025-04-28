from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar


def contour_plot(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    colorbar: bool = True,
    contour_kwargs: dict[str, Any] = {},
    contourf_kwargs: dict[str, Any] = {},
) -> tuple[Figure, Axes, Colorbar]:
    """
    Simple function to plot contours of a 2D grid.

    Parameters
    ----------
    x : np.ndarray
        The x-axis values, from a meshgrid.
    y : np.ndarray
        The y-axis values, from a meshgrid.
    z : np.ndarray
        The z-axis values, applied to the meshgrid.
    colorbar : bool, optional
        Whether to plot a colorbar, by default True.
    contour_kwargs : dict[str, Any], optional
        Arguments to pass to the contour plot, by default {}.
    contourf_kwargs : dict[str, Any], optional
        Arguments to pass to the contourf plot, by default {}.

    Returns
    -------
    tuple[Figure, Axes, Colorbar | None]
        The figure, axes, and colorbar objects.
    """

    fig, ax = plt.subplots()
    contour = ax.contourf(
        x,
        y,
        z,
        **contourf_kwargs,
    )
    ax.contour(
        x,
        y,
        z,
        **contour_kwargs,
    )

    cbar = fig.colorbar(contour, ax=ax)

    if not colorbar:
        cbar.remove()

    return fig, ax, cbar
