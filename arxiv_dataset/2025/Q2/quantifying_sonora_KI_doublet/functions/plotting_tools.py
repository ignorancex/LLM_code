import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
import numpy as np
from scipy import stats
import matplotlib.ticker as mticker

# color mapping
fsed_colors = pl.cm.viridis
logg_colors = pl.cm.plasma
rainbow = plt.cm.Spectral_r
bw_cmap = plt.cm.binary

fsed_num = [1, 2, 3, 4, 8]
fsed_ticks = ["\n Cloudier", '2', '3', '4', 'Less \nCloudy\n 8']
fsed_bounds = [0.5, 1.5, 2.5, 3.5, 6, 9]

fsed_num_w_noclouds = [1, 2, 3, 4, 8, 10]
fsed_ticks_w_noclouds = ["\n Cloudier", '2', '3', '4', '8', 'No \nClouds']
fsed_bounds_w_noclouds = [0.5, 1.5, 2.5, 3.5, 6, 9, 11]

logg_num = [3.5, 4, 4.5, 5, 5.5]
logg_ticks = ['Less \ndense', '4', '4.5', '5', 'Denser']
logg_bounds = [3.25, 3.75, 4.25, 4.75, 5.25, 5.75]


# Filippazzo 2015 spectral type to temperature conversion
# https://iopscience.iop.org/article/10.1088/0004-637X/810/2/158/pdf
# integer values 6-29 correspond to spectral types M6-T9
# L0:10, L1:11, L2:12, L3:13, L4:14, L5:15, L6:16, L7:17, L8:18, L9:19
Teff = lambda x: 4.747e3 -7.005e2*x + 1.155e2*(x**2) - 1.191e1*(x**3) +6.318e-1*(x**4) -1.606e-2*(x**5) +1.546e-4*(x**6)

temp_num =   [ 1600, 1800, 2000, 2200, 2400]
temp_ticks = ['1600 K', '1800 K', '2000 K', '2200 K','2400 K']
temp_bounds = [1550, 1650, 1750, 1850, 1950, 2050, 2150, 2250, 2350, 2450]
spec_types = ["L5", "L4", "L3", "L2", "L1", "L0", " M9"]
spec_ticks = [Teff(type) for type in range(15, 9-1, -1)]

#normalizing colorbars for each parameter
norm_f_w_noclouds = mpl.colors.BoundaryNorm(fsed_bounds_w_noclouds, fsed_colors.N, extend='neither')
norm_f = mpl.colors.BoundaryNorm(fsed_bounds, fsed_colors.N, extend='neither')
norm_g = mpl.colors.BoundaryNorm(logg_bounds, logg_colors.N, extend='max')
norm_temp = mpl.colors.BoundaryNorm(temp_bounds, rainbow.N, extend='neither')
norm_bw_logg = mpl.colors.BoundaryNorm(logg_bounds, logg_colors.N, extend='min')

def temp_color(temp):
    return rainbow(norm_temp(temp))

def logg_bw_color(logg):
    return bw_cmap(norm_bw_logg(logg))

def fsed_color(fsed):
    return fsed_colors(norm_f(fsed))

def logg_color(logg):
    return logg_colors(norm_g(logg))


def fsed_colorbar(fig, cax = None, ax = None, orientation='vertical',
                   shrink=1.0, aspect=20, pad=.14, fontsize = 12):

    cbar = fig.colorbar(pl.cm.ScalarMappable(norm=norm_f, cmap=fsed_colors),
                cax=cax, ax = ax, orientation= orientation, 
                extend='neither', spacing='proportional',
                shrink=shrink, aspect=aspect, pad=pad)
    if orientation == 'vertical':
        cbar.ax.yaxis.set_ticks(fsed_num)
        cbar.ax.yaxis.set_ticklabels(fsed_ticks, fontsize = fontsize)
    else:
        cbar.ax.xaxis.set_ticks(fsed_num)
        cbar.ax.xaxis.set_ticklabels(fsed_ticks, fontsize = fontsize)
    return cbar

def fsed_colorbar_w_noclouds(fig, cax = None, ax = None, orientation='vertical',
                   shrink=1.0, aspect=20, pad=.14,  fontsize = 12):

    cbar = fig.colorbar(pl.cm.ScalarMappable(norm=norm_f_w_noclouds, cmap=fsed_colors),
                cax=cax, ax = ax, orientation= orientation, 
                extend='neither', spacing='proportional',
                shrink=shrink, aspect=aspect, pad=pad)
    if orientation == 'vertical':
        cbar.ax.yaxis.set_ticks(fsed_num_w_noclouds)
        cbar.ax.yaxis.set_ticklabels(fsed_ticks_w_noclouds, fontsize = fontsize)
    else:
        cbar.ax.xaxis.set_ticks(fsed_num_w_noclouds)
        cbar.ax.xaxis.set_ticklabels(fsed_ticks_w_noclouds, fontsize = fontsize)
    return cbar
    
def logg_colorbar(fig, cax = None, ax = None, orientation='vertical',
                  shrink=1.0, aspect=20, pad=.14, fontsize = 12):
    cbar = fig.colorbar(pl.cm.ScalarMappable(norm=norm_g, cmap=logg_colors),
                cax=cax, ax = ax, orientation= orientation,
                extend='neither', spacing='proportional',
                shrink=shrink, aspect=aspect, pad=pad)
    if orientation == 'vertical':
        cbar.ax.yaxis.set_ticks(logg_num)
        cbar.ax.yaxis.set_ticklabels(logg_ticks, fontsize = fontsize)
    else:
        cbar.ax.xaxis.set_ticks(logg_num)
        cbar.ax.xaxis.set_ticklabels(logg_ticks, fontsize = fontsize)

    return cbar


def logg_bw_colorbar(fig, cax = None, ax = None, orientation='vertical',
                 shrink=1.0, aspect=20, pad=.14, fontsize = 12):

    cbar = fig.colorbar(pl.cm.ScalarMappable(norm=norm_bw, cmap=bw_cmap),
                cax=cax, ax = ax, orientation= orientation,
                extend='neither', spacing='proportional',
                shrink=shrink, aspect=aspect, pad=pad)
    cbar.ax.yaxis.set_ticks(logg_num)
    cbar.ax.yaxis.set_ticklabels(logg_ticks, fontsize = fontsize)

    return cbar



def temp_colorbar(fig, cax = None, ax = None, orientation='vertical',
                  shrink=1.0, aspect=20, pad=.14, fontsize = 12, labels = False, labelpad = 0.1):
    cbar = fig.colorbar(pl.cm.ScalarMappable(norm=norm_temp, cmap=rainbow),
                cax=cax, ax = ax, orientation= orientation,
                extend='neither', spacing='proportional',
                shrink=shrink, aspect=aspect, pad=pad)
    
    
    if orientation == 'vertical':
        # Add ticks and labels to both sides of the color bar
        cbar.ax.yaxis.set_ticks_position('both')

        # Set ticks and labels for the left side
        left_ticks = spec_ticks
        left_tick_labels = spec_types
        cbar.ax.yaxis.set_ticks(left_ticks)
        cbar.ax.yaxis.set_ticklabels(left_tick_labels, fontsize = fontsize)
        cbar.ax.yaxis.set_label_position('left')
        if labels:
            cbar.set_label("Spectral Type", fontsize = fontsize + 1, labelpad=labelpad)

        # Create secondary axis for the right side ticks
        cbar_ax_secondary = cbar.ax.twinx()
        cbar_ax_secondary.set_ylim(cbar.ax.get_ylim())


        # Set ticks and labels for the right side
        right_ticks = temp_num 
        right_tick_labels = temp_num
        cbar_ax_secondary.set_yticks(right_ticks)
        cbar_ax_secondary.set_yticklabels(right_tick_labels, fontsize = fontsize -1)
        cbar_ax_secondary.yaxis.set_label_position('right')
        if labels:
            cbar_ax_secondary.set_ylabel("Temperature (K)", fontsize = fontsize + 1)
    else:
        print("Horizontal colorbar not implemented yet")

    return cbar
    return cbar
    
def plot_parameter_vs_logg(ax1, parameter, logg, fsed, param_name, lines = False):
    """
    Plots a given parameter against two variables, `logg` and `fsed`, on separate matplotlib axes.

    Parameters:
    ax1 (matplotlib.axes._subplots.AxesSubplot): The first axis to plot `parameter` against `logg`.
    parameter (array-like): The parameter values to be plotted.
    logg (array-like): The log(g) values to be plotted.
    fsed (array-like): The log(g) values to be plotted.
    param_name (str): The name of the parameter to be used as the y-axis label.
    lines (bool, False): If True a line will be plotted for each individual fsed and logg grouping.

    This function performs the following tasks:
    1. Plots scatter plots of `parameter` against `logg` on `ax1`.
    2. Colors the points on `ax1` based on `fsed` values.
    3. Fits and plots linear regression lines for the data on both axes.
    4. Calculates and annotates Pearson correlation coefficients on both axes.
    5. Sets custom x-ticks and x-tick labels for both axes.
    6. Sets the y-axis label of `ax1` to `param_name`.

    Example usage:
    ```
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    plot_parameter(ax1, df['parameter'], 'Parameter Name')
    plt.show()
    ```
    """

    fsed_values = list(set(fsed))

    # plotting each point individually to assign the correct color
    for i in range(len(logg)):
        ax1.scatter(logg[i], parameter[i],
                    color=fsed_colors(norm_f(fsed[i])), marker='x')
        
    # creating line on the first plot
    m, b = np.polyfit(logg, parameter, 1)
    ax1.plot(logg, logg*m + b, color='k',
             linewidth=.7, alpha=1, label=f'{m:.2}x + {b:.2}')
    # writing pearson r value of the line at the top of each graph
    r1 = stats.pearsonr(logg, parameter)[0]
    ax1.annotate(f"r = {r1:.2}", xy=(1, 0.99),
                 ha='right', va='top',
                 fontsize=10,
                 xycoords='axes fraction', color='k')

    
    # plotting the lines for each logg and fsed value grouping 
    if lines:
        for f in fsed_values:
            m, b = np.polyfit(logg[fsed == f], parameter[fsed == f] , 1)
            ax1.plot(logg, logg*m + b, color=fsed_colors(norm_f(f)),
                        linewidth=.7, alpha=.5)

    # creating custom x-ticks 
    ax1.set_xticks(logg_num)
    ax1.set_xticklabels([None for i in logg_num])

    ax1.set_ylabel(param_name)

def plot_parameter_vs_fsed(ax2, parameter, logg, fsed, param_name, lines = False):
    """
    Plots a given parameter against two variables, `logg` and `fsed`, on separate matplotlib axes.

    Parameters:
    ax1 (matplotlib.axes._subplots.AxesSubplot): The first axis to plot `parameter` against `logg`.
    ax2 (matplotlib.axes._subplots.AxesSubplot): The second axis to plot `parameter` against `fsed`.
    parameter (array-like): The parameter values to be plotted.
    logg (array-like): The log(g) values to be plotted.
    fsed (array-like): The log(g) values to be plotted.
    param_name (str): The name of the parameter to be used as the y-axis label.
    lines (bool, False): If True a line will be plotted for each individual fsed and logg grouping.

    This function performs the following tasks:
    1. Plots scatter plots of  `parameter` against `fsed` on `ax2`.
    2. Colors the points on `ax2` based on `logg` values.
    3. Fits and plots linear regression lines for the data on both axes.
    4. Calculates and annotates Pearson correlation coefficients on both axes.
    5. Sets custom x-ticks and x-tick labels for both axes.
    6. Sets the y-axis label of `ax2` to `param_name`.

    Example usage:
    ```
    fig, (ax2) = plt.subplots(1, 1, figsize=(6, 6))
    plot_parameter(ax1, ax2, df['parameter'], 'Parameter Name')
    plt.show()
    ```
    """

    logg_values = list(set(logg))

    # plotting each point individually to assign the correct color
    for i in range(len(logg)):
        ax2.scatter(fsed[i], parameter[i],
                    color=logg_colors(norm_g(logg[i])), marker='x')
        
    # creating line on the second plot
    m, b = np.polyfit(fsed, parameter, 1)
    ax2.plot(fsed, fsed*m + b, color='k',
             linewidth=.7, alpha=0.5, label=f'{m:.2}x + {b:.2}')
    # writing pearson r value of the line at the top of each graph
    r2 = stats.pearsonr(fsed, parameter)[0]
    ax2.annotate(f"r = {r2:.2}", xy=(1, 0.99),
                 ha='right', va='top',
                 fontsize=10,
                 xycoords='axes fraction', color='k')
    
    # plotting the lines for each logg and fsed value grouping 
    if lines:
        for grav in logg_values:
            m, b = np.polyfit(fsed[logg == grav], parameter[logg == grav] , 1)
            ax2.plot(fsed, fsed*m + b, color=logg_colors(norm_g(grav)),
                        linewidth=.7, alpha=.5)


    # creating custom x-ticks 
    ax2.set_xticks(fsed_num)
    ax2.set_xticklabels([None for i in fsed_num])

    ax2.set_ylabel(param_name)

def plot_parameter_vs_logg_fsed(ax1, ax2, parameter, logg, fsed, param_name, lines = False):
    """
    Plots a given parameter against two variables, `logg` and `fsed`, on separate matplotlib axes.

    Parameters:
    ax1 (matplotlib.axes._subplots.AxesSubplot): The first axis to plot `parameter` against `logg`.
    ax2 (matplotlib.axes._subplots.AxesSubplot): The second axis to plot `parameter` against `fsed`.
    parameter (array-like): The parameter values to be plotted.
    logg (array-like): The log(g) values to be plotted.
    fsed (array-like): The log(g) values to be plotted.
    param_name (str): The name of the parameter to be used as the y-axis label.
    lines (bool, False): If True a line will be plotted for each individual fsed and logg grouping.

    This function performs the following tasks:
    1. Plots scatter plots of `parameter` against `logg` on `ax1` and `parameter` against `fsed` on `ax2`.
    2. Colors the points on `ax1` based on `fsed` values and on `ax2` based on `logg` values.
    3. Fits and plots linear regression lines for the data on both axes.
    4. Calculates and annotates Pearson correlation coefficients on both axes.
    5. Sets custom x-ticks and x-tick labels for both axes.
    6. Sets the y-axis label of `ax1` to `param_name`.

    Example usage:
    ```
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_parameter(ax1, ax2, df['parameter'], 'Parameter Name')
    plt.show()
    ```
    """

    logg_values = list(set(logg))
    fsed_values = list(set(fsed))

    # plotting each point individually to assign the correct color
    for i in range(len(logg)):
        ax1.scatter(logg[i], parameter[i],
                    color=fsed_colors(norm_f(fsed[i])), marker='x')
        ax2.scatter(fsed[i], parameter[i],
                    color=logg_colors(norm_g(logg[i])), marker='x')
        
    # creating line on the first plot
    m, b = np.polyfit(logg, parameter, 1)
    ax1.plot(logg, logg*m + b, color='k',
             linewidth=.7, alpha=1, label=f'{m:.2}x + {b:.2}')
    # writing pearson r value of the line at the top of each graph
    r1 = stats.pearsonr(logg, parameter)[0]
    ax1.annotate(f"r = {r1:.2}", xy=(1, 0.99),
                 ha='right', va='top',
                 fontsize=10,
                 xycoords='axes fraction', color='k')

    # creating line on the second plot
    m, b = np.polyfit(fsed, parameter, 1)
    ax2.plot(fsed, fsed*m + b, color='k',
             linewidth=.7, alpha=0.5, label=f'{m:.2}x + {b:.2}')
    # writing pearson r value of the line at the top of each graph
    r2 = stats.pearsonr(fsed, parameter)[0]
    ax2.annotate(f"r = {r2:.2}", xy=(1, 0.99),
                 ha='right', va='top',
                 fontsize=10,
                 xycoords='axes fraction', color='k')
    
    # plotting the lines for each logg and fsed value grouping 
    if lines:
        for grav in logg_values:
            m, b = np.polyfit(fsed[logg == grav], parameter[logg == grav] , 1)
            ax2.plot(fsed, fsed*m + b, color=logg_colors(norm_g(grav)),
                        linewidth=.7, alpha=.5)
        for f in fsed_values:
            m, b = np.polyfit(logg[fsed == f], parameter[fsed == f] , 1)
            ax1.plot(logg, logg*m + b, color=fsed_colors(norm_f(f)),
                        linewidth=.7, alpha=.5)

    # creating custom x-ticks 
    ax1.set_xticks(logg_num)
    ax1.set_xticklabels([None for i in logg_num])

    ax2.set_xticks(fsed_num)
    ax2.set_xticklabels([None for i in fsed_num])

    ax1.set_ylabel(param_name)



def all_parameter_plot(parameter_df, lines_TF = False, title =  'P-Voigt Parameters vs. Gravity and Clouds'):
    """
    Plots Pseudo-Voigt parameters against gravity (logg) and cloud sedimentation efficiency (f_sed).

    This function creates a multi-panel plot with the following subplots:
    1. Average absorption depth (A) vs. logg and f_sed.
    2. Average Full Width at Half Maximum (FWHM) vs. logg and f_sed.
    3. Average mixing parameter (η) vs. logg and f_sed.

    Parameters:
    -----------
    parameter_df : pandas.DataFrame
        DataFrame containing the Pseudo-Voigt parameters with columns:
        'A1', 'A2', 'FWHM1', 'FWHM2', 'nu1', 'nu2', 'logg', 'clouds'.
    lines_TF : bool, optional
        If True, lines are added to the plots for each color grouping in each plot providing multiple lines
        (default is False).
    title : str, optional
        Title of the entire plot (default is 'P-Voigt Parameters vs. Gravity and Clouds').

    Returns:
    None
    """

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)

    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.05], width_ratios=[1, 1])

    ax = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(3)]
    ########################################################################################
    # Plotting each data points
    # A
    plot_parameter_vs_logg_fsed(ax[0][0], ax[0][1], -(parameter_df.A1 + parameter_df.A2)/2,
                parameter_df.logg, parameter_df.clouds, r"$avg(A)$, Max Depth", lines = lines_TF)

    # FWHM
    plot_parameter_vs_logg_fsed(ax[1][0], ax[1][1], (parameter_df.FWHM1 + parameter_df.FWHM2)/2,
                parameter_df.logg, parameter_df.clouds, r"$avg(\sigma)$, FWHM", lines = lines_TF)

    # mixing parameter
    plot_parameter_vs_logg_fsed(ax[2][0], ax[2][1], (parameter_df.nu1 + parameter_df.nu2)/2,
                parameter_df.logg, parameter_df.clouds, r"$avg(η)$, Mixing Parameter", lines = lines_TF)


    ########################################################################################
    # labels along the bottom
    ax[2][0].set_xlabel("Gravity")
    ax[2][0].set_xticklabels(logg_ticks)

    ax[2][1].set_xlabel(r"$f_{sed}$")
    ax[2][1].set_xticklabels(fsed_ticks)

    ########################################################################################
    # labels along y axis
    # Max depth
    yticks = ax[0][0].get_yticks()
    ylabel = list(yticks.copy()/1e11)
    ylabel[-1] = "Stronger\nabsorption\n"
    ylabel[0] = "Weaker\nabsorption"
    ax[0][0].set_yticks(yticks)
    ax[0][0].set_yticklabels(ylabel)
    ax[0][1].set_yticks(yticks)
    ax[0][1].set_yticklabels([None for i in ylabel])

    # FWHM
    yticks = ax[1][0].get_yticks()
    ylabel = list(np.around(yticks.copy()/1e-3, 1))
    ylabel[-1] = "Broader"
    ylabel[0] = "Narrower"
    ax[1][0].set_yticks(yticks)
    ax[1][0].set_yticklabels(ylabel)
    ax[1][1].set_yticks(yticks)
    ax[1][1].set_yticklabels([None for i in ylabel])

    # Mixing parameter
    yticks = ax[2][1].get_yticks()
    ylabel = list(np.around(yticks.copy(), 2))
    ylabel[0] = "Less\nbroadened\n"
    ylabel[-1] = "Broadened\nwings"
    ax[2][0].set_yticks(yticks)
    ax[2][0].set_yticklabels(ylabel)
    ax[2][1].set_yticks(yticks)
    ax[2][1].set_yticklabels([None for i in ylabel])


    ########################################################################################
    # Colorbar
    cax1 = fig.add_subplot(gs[3, 1])
    axcb = logg_colorbar(fig, cax = cax1,  orientation='horizontal')
    axcb.set_label(r'Surface Gravity', fontsize=10)  # empty label

    cax1 = fig.add_subplot(gs[3, 0])
    axcb = fsed_colorbar(fig, cax = cax1,  orientation='horizontal')
    axcb.set_label(r'Cloudiness', fontsize=10)   # empty label


    fig.suptitle( title , fontsize=16)
    


# plotting each doublet separately
def all_parameter_plot_separate(parameter_df, lines_TF = False, title =  'P-Voigt Parameters vs. Gravity and Clouds'):

    """
    Plots Pseudo-Voigt parameters against gravity (logg) and cloud sedimentation efficiency (f_sed).

    This function creates a multi-panel plot with the following subplots:
    1. Absorption depth (A) of each doublet vs. logg and f_sed.
    2. Full Width at Half Maximum (FWHM) of each doublet vs. logg and f_sed.
    3. Mixing parameter (η) of each doublet vs. logg and f_sed.

    Parameters:
    -----------
    parameter_df : pandas.DataFrame
        DataFrame containing the Pseudo-Voigt parameters with columns:
        'A1', 'A2', 'FWHM1', 'FWHM2', 'nu1', 'nu2', 'logg', 'clouds'.
    lines_TF : bool, optional
        If True, lines are added to the plots for each color grouping in each plot providing multiple lines
        (default is False).
    title : str, optional
        Title of the entire plot (default is 'P-Voigt Parameters vs. Gravity and Clouds').

    Returns:
    None
    """

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)

    gs = fig.add_gridspec(4, 4, height_ratios=[
                        1, 1, 1, 0.05], width_ratios=[1, 1, 1, 1])

    ax = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), fig.add_subplot(
        gs[i, 2]), fig.add_subplot(gs[i, 3])] for i in range(3)]
    ########################################################################################
    # Plotting each data points
    # A
    # doublet 1
    plot_parameter_vs_logg_fsed(ax[0][0], ax[0][2], -parameter_df.A1,
                parameter_df.logg, parameter_df.clouds, r"$A$, Max Depth", lines = lines_TF)

    # doublet 2
    plot_parameter_vs_logg_fsed(ax[0][1], ax[0][3], -parameter_df.A2,
                parameter_df.logg, parameter_df.clouds, r"", lines = lines_TF)


    # FWHM
    # doublet 1
    plot_parameter_vs_logg_fsed(ax[1][0], ax[1][2], parameter_df.FWHM1,
                parameter_df.logg, parameter_df.clouds, r"$\sigma$, FWHM", lines = lines_TF)

    # doublet 2
    plot_parameter_vs_logg_fsed(ax[1][1], ax[1][3], parameter_df.FWHM2,
                parameter_df.logg, parameter_df.clouds, r"", lines = lines_TF)


    # mixing parameter
    # doublet 1
    plot_parameter_vs_logg_fsed(ax[2][0], ax[2][2], parameter_df.nu1,
                parameter_df.logg, parameter_df.clouds, r"$η$, Mixing Parameter", lines = lines_TF)

    # doublet 2
    plot_parameter_vs_logg_fsed(ax[2][1], ax[2][3], parameter_df.nu2,
                parameter_df.logg, parameter_df.clouds, r"", lines = lines_TF)


    ########################################################################################
    # labels along the bottom
    ax[2][0].set_xlabel("Gravity")
    ax[2][0].set_xticklabels(logg_ticks)
    ax[2][1].set_xlabel("Gravity")
    ax[2][1].set_xticklabels(logg_ticks)

    ax[2][2].set_xlabel(r"$f_{sed}$")
    ax[2][2].set_xticklabels(fsed_ticks)
    ax[2][3].set_xlabel(r"$f_{sed}$")
    ax[2][3].set_xticklabels(fsed_ticks)

    ########################################################################################
    # labels along y axis # need to fix power
    # Max depth
    yticks = ax[0][0].get_yticks()
    ylabel = list(yticks.copy()/1e11)
    ylabel[-1] = "stronger\nabsorption\n"
    ylabel[0] = "weaker\nabsorption"
    ax[0][0].set_yticks(yticks)
    ax[0][0].set_yticklabels(ylabel)
    ax[0][1].set_yticks(yticks)  # setting ticks to none for the rest of the rows
    ax[0][1].set_yticklabels([None for i in ylabel])
    ax[0][2].set_yticks(yticks)
    ax[0][2].set_yticklabels([None for i in ylabel])
    ax[0][3].set_yticks(yticks)
    ax[0][3].set_yticklabels([None for i in ylabel])

    # FWHM
    yticks = ax[1][1].get_yticks()
    ylabel = list(np.around(yticks.copy()/1e-3, 1))
    ylabel[0] = "broader"
    ylabel[-1] = "narrower"
    ax[1][0].set_yticks(yticks)
    ax[1][0].set_yticklabels(ylabel)
    ax[1][1].set_yticks(yticks)  # setting ticks to none for the rest of the rows
    ax[1][1].set_yticklabels([None for i in ylabel])
    ax[1][2].set_yticks(yticks)
    ax[1][2].set_yticklabels([None for i in ylabel])
    ax[1][3].set_yticks(yticks)
    ax[1][3].set_yticklabels([None for i in ylabel])

    # Mixing parameter
    yticks = ax[2][0].get_yticks()
    ylabel = list(np.around(yticks.copy(), 2))
    ylabel[0] = "less\nbroadened\n"
    ylabel[-1] = "\nbroadened\nwings"
    ax[2][0].set_yticks(yticks)
    ax[2][0].set_yticklabels(ylabel)
    ax[2][1].set_yticks(yticks)  # setting ticks to none for the rest of the rows
    ax[2][1].set_yticklabels([None for i in ylabel])
    ax[2][2].set_yticks(yticks)
    ax[2][2].set_yticklabels([None for i in ylabel])
    ax[2][3].set_yticks(yticks)
    ax[2][3].set_yticklabels([None for i in ylabel])


    ########################################################################################
    # Colorbar
    cax1 = fig.add_subplot(gs[3, 2])
    axcb = logg_colorbar(fig, cax1,  orientation='horizontal')
    axcb.set_label(r'$\log(g)$', fontsize=10)  # empty label


    cax1 = fig.add_subplot(gs[3, 0])
    axcb = fsed_colorbar(fig, cax1,  orientation='horizontal')
    axcb.set_label(r'$f_{sed}$', fontsize=10)   # empty label

    ########################################################################################
    # setting titles
    # set doublet title
    ax[0][0].set_title("Doublet 1")
    ax[0][1].set_title("Doublet 2")
    ax[0][2].set_title("Doublet 1")
    ax[0][3].set_title("Doublet 2")
    # figure title
    fig.suptitle(title, fontsize=16)



def focused_correlation_plot(parameter_df, lines_TF = False, title =  'P-Voigt Parameters vs. Gravity and Clouds'):
    """
    Plots Pseudo-Voigt parameters against gravity (logg) and cloud sedimentation efficiency (f_sed).

    This function creates a multi-panel plot with the following subplots:
    1. Average absorption depth (A) vs. f_sed.
    2. Average mixing parameter (η) vs. logg.

    Parameters:
    -----------
    parameter_df : pandas.DataFrame
        DataFrame containing the Pseudo-Voigt parameters with columns:
        'A1', 'A2', 'FWHM1', 'FWHM2', 'logg', 'clouds'.
    lines_TF : bool, optional
        If True, lines are added to the plots for each color grouping in each plot providing multiple lines
        (default is False).
    title : str, optional
        Title of the entire plot (default is 'P-Voigt Parameters vs. Gravity and Clouds').

    Returns:
    None
    """

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)

    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05])

    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]

    ########################################################################################
    # Plotting each data points
    # A
    plot_parameter_vs_fsed(ax[1], -(parameter_df.A1 + parameter_df.A2)/2,
                parameter_df.logg, parameter_df.clouds, r"$avg(A)$, Max Depth", lines = lines_TF)

    # FWHM
    plot_parameter_vs_logg(ax[0], (parameter_df.FWHM1 + parameter_df.FWHM2)/2,
                parameter_df.logg, parameter_df.clouds, r"$avg(\sigma)$, FWHM", lines = lines_TF)


    ########################################################################################
    # labels along the bottom
    ax[0].set_xlabel("Gravity")
    ax[0].set_xticklabels(logg_ticks)

    ax[1].set_xlabel(r"$f_{sed}$")
    ax[1].set_xticklabels(fsed_ticks)

    ########################################################################################
    # labels along y axis
    # Max depth
    yticks = ax[1].get_yticks()
    ylabel = list(np.around(yticks.copy()/1e11, 1))
    ylabel[-1] = "Stronger\nabsorption\n"
    ylabel[0] = "Weaker\nabsorption"
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(ylabel)
 
    # FWHM
    yticks = ax[0].get_yticks()
    ylabel = list(np.around(yticks.copy()/1e-3, 1))
    ylabel[-1] = "Broader"
    ylabel[0] = "Narrower"
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(ylabel)

    ########################################################################################
    # Colorbar
    cax1, cax2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3]) 
    
    axcb = logg_colorbar(fig, cax = cax2,  orientation='vertical')
    axcb.set_label(r'$\log(g)$', fontsize=10, rotation = -90)  

    axcb = fsed_colorbar(fig, cax = cax1,  orientation='vertical')
    axcb.set_label(r'$f_{sed}$', fontsize=10, rotation = -90)   


    fig.suptitle( title , fontsize=16)
    
