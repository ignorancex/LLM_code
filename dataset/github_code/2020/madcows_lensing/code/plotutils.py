from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
"""
Plot utilities from Cristobal Sifon
"""
import numpy as np
from cycler import cycler
from matplotlib import (cm, colors as mplcolors, pyplot as plt,
                        rcParams)
import six


def colorscale(array=None, vmin=None, vmax=None, n=0, cmap='viridis'):
    """
    Returns a set of colors and the associated colorscale, to be
    passed to `plt.colorbar()`

    Optional parameters
    -------------------
    array : array-like of floats, shape (N,)
        values to which colors will be assigned.
    vmin : float
        minimum value for the color scale
    vmax : float, vmin < vmax
        maximum value for the color scale
    n : int
        number of regular samples to draw in the range
        `[vmin,vmax]`. Ignored if `array` is defined.
    cmap : str or `matplotlib.colors.ListedColormap` instance
        colormap to be used (or its name).

    Returns
    -------
    ** If neither `array` nor `n` are defined **
    colormap : `matplotlib.colors.ListedColormap` instance
        colormap, normalized to `vmin` and `vmax`.

    ** If either `array` or `n` are defined **
    colors : array-like, shape (4,N)
        array of RGBA colors
    colormap : `matplotlib.colors.ListedColormap` instance
        colormap, normalized to `vmin` and `vmax`.

    Example
    -------
    # color-code particles according to velocity, normalizing the
    # colorbar to [0,0.8] (e.g., to compare with other similar maps)
    >>> x, y, v = np.random.random((3,100))
    >>> colors, colormap = colorscale(array=v, vmin=0, vmax=0.8)
    >>> plt.scatter(x, y, c=colors, s=360, marker='o')
    >>> plt.colorbar(colormap)
    """
    # just in case
    assert n >= 0, 'Number `n` of samples must be >= 0'
    # find colormap (I don't think this is necessary)
    if isinstance(cmap, six.string_types):
        cmap = getattr(cm, cmap)
    elif type(cmap) != mplcolors.ListedColormap:
        msg = 'argument cmap must be a string or' \
              ' a matplotlib.colors.ListedColormap instance'
        raise TypeError(msg)
    # Default values for vmin and vmax. Set to (0,1) if `array` is not
    # provided; set to `(min(array),max(array))` otherwise.
    if array is None:
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 1
    else:
        array = np.array(array)
        if vmin is None:
            vmin = array.min()
        if vmax is None:
            vmax = array.max()
    assert vmin < vmax, 'Please ensure `vmin < vmax`'

    # define normalization for colormap
    cnorm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
    colorbar = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    # this is necessary for the colorbar to be interpreted by
    # plt.colorbar()
    colorbar._A = []
    if array is None and n == 0:
        return colorbar
    # now get the colors
    if array is None:
        array = np.linspace(vmin, vmax, n)
    colors = colorbar.to_rgba(array)
    return colors, colorbar


def savefig(output, fig=None, close=True, verbose=True, name='',
            tight=True, **kwargs):
    """
    Wrapper to save figures

    Parameters
    ----------
        output  : str
                  Output file name (including extension)

    Optional parameters
    -------------------
        fig     : pyplot.figure object
                  figure containing the plot.
        close   : bool
                  Whether to close the figure after saving.
        verbose : bool
                  Whether to print the output filename on screen
        name    : str
                  A name to identify the plot in the stdout message.
                  The message is always "Saved {name} to {output}".
        tight   : bool
                  Whether to call `tight_layout()`
        kwargs : dict
                  keyword arguments to be passed to `tight_layout()`

    """
    if fig is None:
        fig = plt
    if tight:
        if 'pad' not in kwargs:
            kwargs['pad'] = 0.4
        fig.tight_layout(**kwargs)
    fig.savefig(output)
    if close:
        plt.close()
    if verbose:
        print('Saved {1} to {0}'.format(output, name))
    return


def update_rcParams(dict={}):
    """
    Update matplotlib's rcParams with any desired values. By default,
    this function sets lots of parameters to my personal preferences,
    which basically involve larger font and thicker axes and ticks,
    plus some tex configurations.

    Returns the rcParams object.

    """
    default = {}
    for tick in ('xtick', 'ytick'):
        default['{0}.major.size'.format(tick)] = 8
        default['{0}.minor.size'.format(tick)] = 4
        default['{0}.major.width'.format(tick)] = 2
        default['{0}.minor.width'.format(tick)] = 2
        default['{0}.minor.visible'.format(tick)] = True
        default['{0}.labelsize'.format(tick)] = 20
        default['{0}.direction'.format(tick)] = 'in'
    default['xtick.top'] = True
    default['ytick.right'] = True
    default['axes.linewidth'] = 2
    default['axes.labelsize'] = 22
    default['font.family'] = 'sans-serif'
    default['font.size'] = 22
    default['legend.fontsize'] = 18
    default['lines.linewidth'] = 2
    default['text.latex.preamble']=['\\usepackage{amsmath}']
    # Matthew Hasselfield's color-blind-friendly style
    default['axes.prop_cycle'] = \
        cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd',
                      '#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
    for key in default:
        # some parameters are not valid in different matplotlib functions
        try:
            rcParams[key] = default[key]
        except KeyError:
            pass
    # if any parameters are specified, overwrite anything previously
    # defined
    for key in dict:
        try:
            rcParams[key] = dict[key]
        except KeyError:
            pass
    return

