from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def generate_kde(x, y, ax, bin_num, label, color, line_style, fw, x_low, x_high):
    """
    Plots a 1D histogram with a kernel density estimation (KDE) overlay.
    
    Parameters:
    -----------
    x : array-like
        Input data for KDE. Can contain NaNs or infinite values.
    
    y : array-like
        Weights for `x`, used only if `fw` is True.
    
    ax : matplotlib.axes.Axes
        Axis object where the plot is drawn.
    
    bin_num : int
        Number of bins for KDE evaluation.
    
    label : str
        Legend label for the plot.
    
    color : str or color
        Color for the KDE plot line.
    
    line_style : str
        Line style for the KDE plot line.
    
    fw : bool
        Flag indicating whether to use weights for KDE.
    
    x_low : float
        Lower bound of the KDE range.
    
    x_high : float
        Upper bound of the KDE range.
    
    Returns:
    --------
    None
    """
    # Remove non-finite values from x (and y if weighted KDE is used)
    mask = np.isfinite(x) & (np.isfinite(y) if fw else True)
    x, y = x[mask], y[mask]

    # Create the kernel density estimation (weighted if fw=True)
    kernel_1 = stats.gaussian_kde(x, weights=y if fw else None)

    # Plot the KDE on the provided axis
    bins = np.linspace(x_low, x_high, bin_num)
    ax.plot(bins, kernel_1(bins), color=color, lw=2, ls=line_style, label=label)


def solve_cubic(a2, a1, a0):
    """
    Solves the cubic equation of the form: x^3 + a2*x^2 + a1*x + a0 = 0.

    Parameters:
    a2 (float): Coefficient of x^2.
    a1 (float): Coefficient of x.
    a0 (float): Constant term.

    Returns:
    tuple: Three real roots of the cubic equation.
    """

    # Calculate the reduced coefficients
    p = (3 * a1 - a2 ** 2) / 3  # Depressed cubic's linear coefficient
    q = (9 * a1 * a2 - 27 * a0 - 2 * a2 ** 3) / 27  # Depressed cubic's constant term

    # Intermediate terms for solving the cubic equation
    Q = p / 3
    R = q / 2
    discriminant = Q ** 3 + R ** 2  # Discriminant of the cubic equation

    # Calculate the angle theta for real root solutions
    theta = np.arccos(R / np.sqrt(-Q ** 3))

    # Compute the three real roots
    root1 = 2 * np.sqrt(-Q) * np.cos(theta / 3) - a2 / 3
    #root2 = 2 * np.sqrt(-Q) * np.cos((theta + 2 * np.pi) / 3) - a2 / 3 
    #root3 = 2 * np.sqrt(-Q) * np.cos((theta + 4 * np.pi) / 3) - a2 / 3
    # The discriminant, D = Q3 + R2, in this case is less than zero, which means there are three different real solutions.   
    # However, only one solution is positive: root1
    
    return root1


def do_hist2d(x, y, w, ax, bin_width, aspect, t_min, t_max, v_min, v_max, cmap):
    """
    Generates a 2D histogram (heatmap) using the provided data and plots it on the specified axis.
    
    Parameters:
    x (array): Data for the x-axis.
    y (array): Data for the y-axis.
    w (array): Weights for the histogram.
    ax (matplotlib axis): Axis on which to plot the histogram.
    bin_width (float): bin width of bins for the histogram.
    aspect (float): Aspect ratio for the plot.
    t_min, t_max (float): Range for x-axis bins.
    v_min, v_max (float): Range for y-axis bins.
    cmap (str): Colormap for the plot.

    Returns:
    im1: The image object created by `imshow` for the heatmap.
    """

    # Remove non-finite values from x, y, w
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[valid_mask], y[valid_mask], w[valid_mask]

    # Define bin edges
    bins_x = np.arange(t_min, t_max + bin_width, bin_width)
    bins_y = np.arange(v_min, v_max + bin_width, bin_width)

    # Compute weighted and unweighted 2D histograms
    heatmap_weighted, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y], weights=w)

    # Define the extent for the plot
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Plot the heatmap with logarithmic normalization
    im = ax.imshow(heatmap_weighted.T / w.sum(), extent=extent, interpolation=None, 
                    aspect=aspect, origin='lower', cmap=cmap, 
                    norm=LogNorm(vmin=1e-4, vmax=1e-1))

    return im

