"""
Author: Renato Durrer
Created: 18.05.2019


"""
import numpy as np
import matplotlib

def latexify(fig_width=14.8, fig_height=None, columns=1, textsize=10):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    fig_width =fig_width / 2.54
    if fig_height is not None:
        fig_height = fig_height / 2.54
    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns ==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5 ) -1.0 ) /2.0    # Aesthetic ratio
        fig_height = fig_width *golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': textsize, # fontsize for x and y labels (was 10)
              'axes.titlesize': textsize,
              'legend.fontsize': textsize, # was 10
              'xtick.labelsize': textsize,
              'ytick.labelsize': textsize,
              'text.usetex': True,
              'figure.figsize': [fig_width ,fig_height],
              'font.family': 'sans-serif',
              'lines.linewidth': 0.8,
              'figure.dpi': 600.0
              }
    # set colormap
    matplotlib.pyplot.viridis()
    matplotlib.rcParams.update(params)