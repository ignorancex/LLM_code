import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns


def corr_2d_heatmap(df, suptitle=None):
    corr = df.corr()

    f, ax = plt.subplots(figsize=(10,6))
    hm = sns.heatmap(
        round(corr,2),
        annot=True,
        ax=ax,
        cmap="coolwarm",
        fmt='.2f',
        linewidths=.05,
    )

    f.subplots_adjust(top=0.93)

    if isinstance(suptitle, str):
        t=f.suptitle(suptitle, fontsize=14)


def many_jointplots(df, title=None, axes_lim=None, diag_kind='kde', color=None):
    plot_kws = {'edgecolor': 'k', 'linewidth': 0.5}

    pp = sns.pairplot(
        df,
        c=color,
        size=1.8,
        aspect=1.8,
        plot_kws=plot_kws,
        diag_kind=diag_kind,
    )

    pp.fig.subplots_adjust(top=0.93, wspace=0.3)

    if axes_lim:
        for i, row in enumerate(pp.axes):
            for j, ax in enumerate(row):
                if i == j: continue
                ax.set_xlim(axes_lim[:2])
                ax.set_ylim(axes_lim[2:])

    if isinstance(title, str):
        t = pp.fig.suptitle(title, fontsize=14)


def scatter3d(df, cols, ax=None, **scatter_kws):
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None

    xs = df[cols[0]]
    ys = df[cols[1]]
    zs = df[cols[2]]
    ax.scatter(xs, ys, zs, s=50, alpha=0.5, edgecolors='w', **scatter_kws)

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    # for use of rotating orientation.
    return fig, ax

# plotting prob simplex 3d
#fig, ax = scatter3d(df_all, [0,1,2], **{'c':df_all['c']})
#for i in range(len(eye)):
#    ax.plot([eye[0][i], eye[1][i]], [eye[1][i], eye[2][i]], [eye[2][i], eye[0][i]], color='black')
#ax.view_init(elev=45., azim=45)
#plt.savefig('../../sjd/sim/fq/small_10samples/exp1/tmp_visualize_5u_10s_bnn.png', dpi=400)
#plt.close()



def aligned_hists(df, bins=10, title=None, xaxis_label=None, yaxis_label=None):
    fig, axes = plt.subplots()

    df.hist(bins=bins, ax=axes, sharex=True)

    if isinstance(title, str):
        plt.suptitle(title)

    if isinstance(xaxis_label, str):
        fig.text(0.5, 0.04, xaxis_label, ha='center')
    if isinstance(yaxis_label, str):
        fig.text(0.04, 0.5, yaxis_label, va='center', rotation='vertical')


def func_on_scatter(
    df1,
    df2,
    df1_label,
    df2_label,
    title=None,
    axes_lim=None,
):
    """Plots two sets of data, one as datapoints and the other as a function"""
    plot_kws = {'edgecolor': 'face', 'linewidth': 0.2, 'alpha': 0.6}

    #append them while adding another column indicating
    df1['plot_id'] = [df1_label] * len(df1)
    df2['plot_id'] = [df2_label] * len(df2)

    df = df1.append(df2, True)

    pp = sns.pairplot(
        df,
        hue='plot_id',
        size=1.8,
        aspect=1.8,
        palette={df1_label: "#6666FF", df2_label: "#FF6666"},
        markers=['o', 's'],
        #markers=['+', 'x'],
        plot_kws=plot_kws,
    )

    pp.fig.subplots_adjust(top=0.93, wspace=0.3)

    if axes_lim:
        for i, row in enumerate(pp.axes):
            for j, ax in enumerate(row):
                if i == j: continue
                ax.set_xlim(axes_lim[:2])
                ax.set_ylim(axes_lim[2:])

    if isinstance(title, str):
        t = pp.fig.suptitle(title, fontsize=14)


def hist_func(
    data,
    #func_range,
    func,
    bins=40,
    title=None,
    xaxis_label=None,
    yaxis_label=None,
    nrows=2,
    ncols=2,
    axes_lim=None,
    density=True,
):
    """Plot a function line on top of a histogram."""
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10, 10),
    )

    plot_count = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]

            if plot_count < len(data.columns):
                # Draw histogram
                ax.hist(
                    data.iloc[:, plot_count],
                    bins=bins,
                    color=(0.0, 0.0, 1.0, 0.5),
                    label=str(data.columns[plot_count]),
                    density=density,
                )

                if axes_lim:
                    # Set axis limits
                    ax.set_xlim(axes_lim[:2])
                    # No setting of the y axis limits due to not necessarily
                    # being bounded by the same anyways.
                    #ax.set_ylim(axes_lim[2:])

                # Draw axis values or not (if aligned only draw on left and bottom)
                if j == 0:
                    # Draw the y axis values
                    pass

                if i == nrows - 1:
                    # Draw the x axis values
                    pass

                # Draw the function over the histogram
                #ax.plot(
                # plot the histogram of samples to compare to.
                # look at densities
                ax.hist(
                    func[:, plot_count],
                    bins=bins,
                    color=(1.0, 0.0, 0.0, 0.5),
                    density=density,
                )
            else:
                # Turn off the axes for the blank plots, if any.
                ax.set_axis_off()

            plot_count += 1

    # TODO draw the legend of entire figure, data and function

    if isinstance(title, str):
        plt.suptitle(title)

    if isinstance(xaxis_label, str):
        fig.text(0.5, 0.04, xaxis_label, ha='center')
    if isinstance(yaxis_label, str):
        fig.text(0.04, 0.5, yaxis_label, va='center', rotation='vertical')


def pair_plot_info(
    df,
    title=None,
    axes_lim=None,
    diag_kind='kde',
    diag_kws=None,
    lower_kws=None,
    upper_kws=None,
    pair_grid_args=None,
):
    """Creates a paired plot with hist/kde down the diagonal, scatter plot on
    the upper right triangle, and density plot on the lower right triangle.
    """
    if pair_grid_args is None:
        pair_grid_args = {}

    sns.set_style("whitegrid")
    pg = sns.PairGrid(df, **pair_grid_args)

    # Set diagonal hist or kde
    if diag_kind == 'kde':
        diag_args = {
            'shade': True,
            'legend': False,
        }
        if diag_kws:
            diag_args.update(diag_kws)

        pg.map_diag(sns.kdeplot, **diag_args)
    elif diag_kind == 'hist':
        diag_args = {
            'bins': 20,
        }
        if diag_kws:
            diag_args.update(diag_kws)

        pg.map_diag(plt.hist, **diag_args)
    else:
        # distance plot: a hist with a overlayed kde
        diag_args = {
            'bins': 20,
            'kde_kws': {'legend': False, 'color':'k', 'alpha': 0.5},
            'hist_kws': {'alpha': 1.0},
            #'norm_hist': True, # Needs its own scale for this to be useful
        }
        if diag_kws:
            diag_args.update(diag_kws)

        pg.map_diag(sns.distplot, **diag_args)

    # Set lower left triangle to density
    lower_args = {
        #'shade': True,
        #'shade_lowest': False,
        'legend': False,
        #cbar=True, # need to provide a c_bar alignment (ie. 1 cbar for all)
    }
    if lower_kws:
        lower_args.update(lower_kws)

    pg.map_lower(sns.kdeplot, **lower_args)

    # Set upper right triangle to scatter plot
    upper_args = {
        'edgecolor': 'face',
        'linewidth': 0.2,
        'alpha': 0.5,
    }
    if upper_kws:
        upper_args.update(upper_kws)

    pg.map_upper(plt.scatter, **upper_args)

    if axes_lim:
        for i, row in enumerate(pg.axes):
            for j, ax in enumerate(row):
                if i == j:
                    # Turn off grid for diagonal, because it is only meant to
                    # indicate shape, nothing quantifiable or about magnitudes;
                    # Needs a separate plot for that (histogram).
                    ax.grid(False)
                    continue
                ax.set_xlim(axes_lim[:2])
                ax.set_ylim(axes_lim[2:])

    if isinstance(title, str):
        t = pg.fig.suptitle(title, fontsize=14)


def overlaid_pairplot(dfs, *args, **kwargs):
    """Creates a pair plot of overlaid data (ie. mutliple data depicted in
    different colors).
    """

    # TODO create / get joint axes for each plot

    # TODO get optional

    for df in dfs:
        pair_plot_info(df, *args, **kwargs)


def add_interval_to_hist():
    """Adds lines to a histogram indicating
    """

    return


def violins(
    vectors,
    paired_vectors=False,
    full_violin=False,
    axis=1,
    ax=None,
    positions=None,
    vert=True,
    widths=0.5,
    showmean=False,
    showextrema=False,
    showmedian=False,
    points=100,
    bw_method=None,
    *args,
    **kwargs,
):
    """Creates violin plots of different univariate data. Optionally plot
    interval information on individual violin plots, such as the credible
    interval.

    Notes
    -----
    - Want to be able to use a histogram or any other KDE than just a gaussian
    KDE
    - Want to be able to use a:
        - half violin: 1 hist/kde on one side and
    nothing on the other: saves space
        - split violin: two different distribs on opposite sides of same
          violin. Good for comparing in sample and out of sample distribs of
          measures.
            - be able to give the two different splits different colors.
        - full violin only for completion of general code.
    - Want to be able to visualize both the mean and median of the data in the
      violin with clearly marked dashed lines that makes it easy to distinguish
      between the two given a legend for them and allows them to be both seen
      when they overlap.
    - Want to be able to clearly mark quantiles
    - Want to be able to clearly mark given values as a credible interval (or
      any interval, and multiple of them)
        - Would like them to (optionally) extend to the entire hieght of distrib and optionally change color of background or use a very light transparent overlay of the interval.
        - Also would like option to have the inverse of the interval
          highlighted a color to indicate where is outside of the credible
          interval (such as some contrasting color or given color, perhaps
          default to a transparent red).

    Parameters
    ----------
    vectors : list(np.ndarray)
        A list of 1d arrays. Each array is a different set of data.
    paired_vectors : bool, optional
        If True, then the number of given vectors must be even and every two
        vectors are then a different half of the same plot.
    full_violin : bool, optional
        If paired_vectors is False and this is False, then only half the
        violins will be created to save space in the plot. This is the default
        behavior. If full violins per vector are desired, where both sides of
        the violin are representing the same data and are thus symmetric, then
        full_violin must be provided as True.
    keep_top : bool
        Only applies if both `full_violin` and `paired_vectors` are both False,
        and if so, then if this is True then the top (or left if `vert` is
        True) part of the violin is kept to create the half violin, otherwise
        if False, then the bottom (or right if `vert` is True) half of the
        violin is kept.
    axis : int, optional
        The axis along which the violin plot is computed. The default is 1.
    density : bool, optional
        Finds the frequency of all values in order to fairly compare

    positions :
        Same as `matplotlib.axes.Axes.violinplot`.
    vert :
        Same as `matplotlib.axes.Axes.violinplot`.
    widths :
        Same as `matplotlib.axes.Axes.violinplot`.
    showmean :
        Same as `matplotlib.axes.Axes.violinplot`.
    showextrema :
        Same as `matplotlib.axes.Axes.violinplot`.
    showmedian :
        Same as `matplotlib.axes.Axes.violinplot`.
    points :
        Same as `matplotlib.axes.Axes.violinplot`.
    bw_method :
        Same as `matplotlib.axes.Axes.violinplot`.

    Returns
    -------
    """
    # TODO Only necessary if seaborn plot cannot easily be given a credible interval
    # to overlay on the violins.
    # ax = sns.violinplot(x=vectors concatenated, y = model id, hue = train/test)
    # If able to add the interval, then sns has everything I need, and this
    # would just be making the dataframe given data and running the
    # sns.violinplot appropriately.

    # TODO return whatever to make it so multiple vertical stacked violin plots
    # can be placed in a figure together, for easily visualizing the distribs
    # on the different datasets.
    if ax is None:
        fig, axx = plt.subplots()
    else:
        axx = ax

    #if density:
    #    # Turn all values into densities so the integral (area) sums to one.
    #   NOTE just want the hists/kdes to be representative

    # Create the individual violins as per normal.
    violins_parts = axx.violinplot(
        vectors,
        positions,
        vert,
        widths,
        showmeans,
        showextrema,
        showmedians,
        points,
        bw_method,
        *args,
    )

    if paired_vectors:
        # Create split violins from pairs of vectors
        if len(vectors) % 2 != 0:
            raise ValueError(
                '`paired_vectors` is True, but number of vectors is odd.',
            )

    elif full_violin:
        # Create standard symmetric violin per vector
        if showmeans and showmedians:
            # TODO change line type
            pass
    else:
        # Create one-sided (half) violin per vector
        pass

        # clip the selected side to be removed based on keep_top

    # TODO set shared axis range

    # Could use Split violins, instead of juts halves and show the In Sample
    # and Out of sample
    return axx
