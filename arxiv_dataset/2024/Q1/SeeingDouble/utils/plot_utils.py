from matplotlib import pyplot as plt
from matplotlib import colors
import scienceplots
import matplotlib as mpl
import numpy as np

# ######################
# ##### Color Maps #####
# ######################

def initialize():
    fig, axes = plt.subplots(figsize=(8,8))
    plt.close()



# Constants
DPI = 72
FULL_WIDTH_PX = 510
COLUMN_WIDTH_PX = 245

FULL_WIDTH_INCHES = FULL_WIDTH_PX / DPI
COLUMN_WIDTH_INCHES = COLUMN_WIDTH_PX / DPI

GOLDEN_RATIO = 1.618

def newplot(scale, subplot_array = None, width = None, height = None, golden_ratio = False, stamp = None, stamp_kwargs = None, **kwargs):


    plt.rcParams['figure.autolayout'] = True

    # Determine plot aspect ratio
    aspect_ratio = 1
    if golden_ratio:
        aspect_ratio = GOLDEN_RATIO

    # Determine plot size if not directly set
    if scale is None:
        plot_scale = "column"
    if scale == "full":
        fig_width = FULL_WIDTH_INCHES
        fig_height = FULL_WIDTH_INCHES * aspect_ratio
        plt.style.use('utils/rikab_full.mplstyle')

    elif scale == "column":
        fig_width = COLUMN_WIDTH_INCHES
        fig_height = COLUMN_WIDTH_INCHES * aspect_ratio
        plt.style.use('utils/rikab_column.mplstyle')
    else:
        raise ValueError("Invalid scale argument. Must be 'full' or 'column'.")


    if width is not None:
        fig_width = width
    if height is not None:
        fig_height = height

    if subplot_array is not None:
        fig, ax = plt.subplots(subplot_array[0], subplot_array[1], figsize=(fig_width, fig_height), **kwargs)

    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), **kwargs)
        ax.minorticks_on()


    # Plot title
    stamp_kwargs_default = {"style" : 'italic', "horizontalalignment" : 'right', "verticalalignment" : 'bottom', "transform" : ax.transAxes, "fontsize" : 10}
    if stamp_kwargs is not None:
        stamp_kwargs_default.update(stamp_kwargs)

    if stamp is not None:
        # Text in the top right corner, right aligned:
        plt.text(1, 1, stamp, **stamp_kwargs_default)



    return fig, ax



# ######################


def plot_event(event, R, ax = None, filename=None, color="red", title="", show=True):

    # plot the two events
    plt.rcParams.update({'font.size': 18})
    if ax is None:
        newplot

    pts, ys, phis =event[:,0], event[:, 1], event[:, 2]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts), color=color, lw=0, zorder=10, label="Event")

    # Legend
    # legend = plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)
    # legend.legendHandles[0]._sizes = [150]

    # plot settings
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    plt.xlabel('Rapidity')
    plt.ylabel('Azimuthal Angle')
    plt.title(title)
    plt.xticks(np.linspace(-R, R, 5))
    plt.yticks(np.linspace(-R, R, 5))

    ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.show()
        plt.close()
        return ax
    elif show:
        plt.show()
        return ax
    else:
        return ax
    


    # Function to take a list of points and create a histogram of points with sqrt(N) errors, normalized to unit area
def hist_with_errors(ax, points, bins, range, weights = None, show_zero = False, label = None, **kwargs):

    if weights is None:
        weights = np.ones_like(points)

    hist, bin_edges = np.histogram(points, bins = bins, range = range, weights = weights)
    errs2 = np.histogram(points, bins = bins, range = range, weights = weights**2)[0]

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    hist_tot = np.sum(hist * bin_widths)
    hist = hist / hist_tot
    errs2 = errs2 / (hist_tot**2)

    ax.errorbar(bin_centers[hist > 0], hist[hist > 0], np.sqrt(errs2[hist > 0]), xerr = bin_widths[hist > 0] / 2, fmt = "o", label = label, **kwargs)



def hist_with_outline(ax, points, bins, range, weights = None, color = "purple", alpha_1 = 0.25, alpha_2 = 0.75, label = None,  **kwargs):
    
    if weights is None:
        weights = np.ones_like(points)

    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = alpha_1, histtype='stepfilled', **kwargs)
    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = alpha_2, histtype='step', label = label, **kwargs)


    # # # Dummy plot for legend
    # if label is not None:

    #     edgecolor = mpl.colors.colorConverter.to_rgba(color, alpha=alpha_2)
    #     ax.hist(points, bins = bins, range = range, weights = weights * -1, color = color, alpha = alpha_1, lw = lw*2, label = label, edgecolor = edgecolor, **kwargs)






def function_with_band(ax, f, range, params, pcov = None, color = "purple", alpha_line = 0.75, alpha_band = 0.25, lw = 3,  **kwargs):

    x = np.linspace(range[0], range[1], 1000)

    if pcov is not None:

        # Vary the parameters within their errors
        n = 1000
        temp_params = np.random.multivariate_normal(params, pcov, n)
        y = np.array([f(x, *p) for p in temp_params])

        # Plot the band

        y_mean = np.mean(y, axis = 0)
        y_std = np.std(y, axis = 0) 

        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color = color, alpha = alpha_band, **kwargs)


    y = f(x, *params)
    ax.plot(x, y, color = color, alpha = alpha_line, lw = lw, **kwargs)