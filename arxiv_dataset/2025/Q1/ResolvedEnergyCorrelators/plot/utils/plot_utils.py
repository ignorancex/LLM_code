import numpy as np

from matplotlib.path import Path
from matplotlib.patches import PathPatch

import colorsys
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Logging
import logging
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler())

# TOGGLE: plot_utils has its own LOGGER
LOGGER.setLevel(logging.WARN)
# LOGGER.setLevel(logging.DEBUG)


# --------------------------------
# Histogram analysis
# --------------------------------
def get_full_width(x: np.ndarray, y: np.ndarray,
                   height_ratio: float = 0.5,
                   return_loc=False) -> float:
    height_half_max = np.max(y) * height_ratio
    index_max = np.argmax(y)
    x_low = np.interp(height_half_max, y[:index_max+1], x[:index_max+1])
    x_high = np.interp(height_half_max, np.flip(y[index_max:]), np.flip(x[index_max:]))

    if return_loc:
        return x_high - x_low, x_low, x_high

    return x_high - x_low


# ---------------------------------------------------
# Putting text on figures:
# ---------------------------------------------------
def stamp(left_x, top_y, axes,
          delta_y=0.075,
          textops_update=None,
          **kwargs):
    """Adds a stamp to figures"""
    # text options
    textops = {'horizontalalignment': 'left',
               'verticalalignment': 'center',
               'fontsize': 8.5,
               'transform': axes.transAxes}
    if isinstance(textops_update, dict):
        textops.update(textops_update)

    # add text line by line
    for i in range(len(kwargs)):
        y_val = top_y - i*delta_y
        text = kwargs.get('line_' + str(i))
        if text is not None:
            axes.text(left_x, y_val, text,
                    **textops)

def latex_scientific(number, decimals=2):
    mantissa, exponent = f"{number:.{decimals}e}".split("e")
    exponent = int(exponent)
    return f"{mantissa} \\times 10^{{{exponent}}}"

# ---------------------------------------------------
# Colors
# ---------------------------------------------------
def adjust_lightness(color, amount=0.5):
    """
    Adjusts the lightness of the given color by multiplying
    (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> adjust_lightness('g', 0.3)
    >> adjust_lightness('#F034A3', 0.6)
    >> adjust_lightness((.3,.55,.1), 0.5)

    From https://stackoverflow.com/a/49601444
    """
    try:
        col = mcolors.cnames[color]
    except:
        col = color
    col = colorsys.rgb_to_hls(*mcolors.to_rgb(col))
    return colorsys.hls_to_rgb(col[0], max(0, min(1, amount * col[1])), col[2])


def get_colors_colorbar(vals, colormap=cm.jet, norm='linear'):
    """
    Sets up a colorbar for a set of values to be plotted;
    See, e.g., https://stackoverflow.com/a/30781043.

    Parameters
    ----------
        vals : Values to be plotted

    Returns
    -------
        Colors for each value and a

    """
    # See, e.g., https://stackoverflow.com/a/30781043
    # setup the norm and the colormap
    if norm in ['lin', 'linear']:
        normalize = mcolors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    elif norm in ['log', 'logarithmic']:
        if np.min(vals) > 0:
            normalize = mcolors.LogNorm(vmin=np.min(vals), vmax=np.max(vals))
        else:
            normalize = mcolors.SymLogNorm(linthresh=1e-3,
                                           vmin=np.min(vals),
                                           vmax=np.max(vals))
    else:
        raise NotImplementedError(f"{norm=} is not an implemented"
                                  " normalization.")

    if colormap == 'jet':
        colormap = cm.jet
    elif colormap == 'rainbow':
        colormap = cm.rainbow
    elif colormap == 'gist_rainbow':
        colormap = cm.gist_rainbow
    elif colormap == 'brg':
        colormap = cm.brg
    elif colormap == 'viridis':
        colormap = cm.viridis
    elif colormap == 'cool':
        colormap = cm.cool
    elif colormap == 'plasma':
        colormap = cm.plasma
    elif colormap == 'tab20c':
        colormap = cm.tab20c

    # List of colors
    colors = [colormap(normalize(rad)) for rad in vals]

    # Associated colorbar (usage: plt.colorbar(scalarmappaple))
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(vals)

    return colors, scalarmappaple
