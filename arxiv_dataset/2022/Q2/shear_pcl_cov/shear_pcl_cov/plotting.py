"""
Functions for producing plots.
"""

import copy
import warnings

import healpy as hp
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.ndimage as ndimage
import scipy.special
import skimage.measure

import gaussian_cl_likelihood.python.posteriors as gcl_post
import plot_utils


def post_3d(post_paths, labels, colours, linestyles, contour_levels_sig, x_label=None, y_label=None, z_label=None,
            x_lims=None, y_lims=None, z_lims=None, smooth_xy=None, smooth_xz=None, smooth_yz=None, smooth_x=None,
            smooth_y=None, smooth_z=None, print_areas=False, save_path=None):
    """
    Produce triangle plot showing multiple 3D posteriors, each as output by plot_utils.get_3d_post.

    Args:
        post_paths (list): List of paths to 3D posterior .npz files, each as output by plot_utils.get_3d_post.
        labels (list): List of legend labels, one for each posterior grid.
        colours (list): List of colours, one for each posterior grid.
        linestyles (list): List of linestyles, one for each posterior grid.
        contour_levels_sig (list): List of confidence regions to plot in ascending order, e.g. [1, 3].
        x_label (str, optional): X-axis label - default None, i.e. no label.
        y_label (str, optional): Y-axis label - default None, i.e. no label.
        z_label (str, optional): Z-axis label - default None, i.e. no label.
        x_lims ((float, float), optional): X-axis limits - default None, limits set automatically.
        y_lims ((float, float), optional): Y-axis limits - default None, limits set automatically.
        z_lims ((float, float), optional): Z-axis limits - default None, limits set automatically.
        smooth_xy (list, optional): List of kernel standard deviations for Gaussian smoothing in the x-y plane, one for
                                    each posterior grid, or None for no smoothing (default None).
        smooth_xz (list, optional): List of kernel standard deviations for Gaussian smoothing in the x-z plane, one for
                                    each posterior grid, or None for no smoothing (default None).
        smooth_yz (list, optional): List of kernel standard deviations for Gaussian smoothing in the y-z plane, one for
                                    each posterior grid, or None for no smoothing (default None).
        smooth_x (list, optional): List of kernel standard deviations for Gaussian smoothing of the 1D x posterior, one
                                   for each posterior grid, or None for no smoothing (default None).
        smooth_y (list, optional): List of kernel standard deviations for Gaussian smoothing of the 1D y posterior, one
                                   for each posterior grid, or None for no smoothing (default None).
        smooth_z (list, optional): List of kernel standard deviations for Gaussian smoothing of the 1D z posterior, one
                                   for each posterior grid, or None for no smoothing (default None).
        print_areas (bool, optional): If True, print relative areas/widths of the different posteriors. Note that
                                      smoothing can affect these results, so for reliable results smoothing should be
                                      switched off to extract relative areas, and then smoothing values should be set to
                                      preserve unsmoothed relative areas. Default False.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Load unnormalised 3D posteriors
    post_grids = []
    for post_idx, post_path in enumerate(post_paths):
        print(f'Loading {post_idx + 1} / {len(post_paths)}')
        with np.load(post_path) as data:
            x_grid_tmp = data['x_grid']
            y_grid_tmp = data['y_grid']
            z_grid_tmp = data['z_grid']
            post_grids.append(data['post_grid'])

        # Check grids consistent
        if post_idx == 0:
            x_grid, y_grid, z_grid = x_grid_tmp, y_grid_tmp, z_grid_tmp
        else:
            assert np.array_equal(x_grid, x_grid_tmp)
            assert np.array_equal(y_grid, y_grid_tmp)
            assert np.array_equal(z_grid, z_grid_tmp)

    # Form 1D & 2D grids
    print('Forming 1D & 2D grids')
    x = x_grid[:, 0, 0]
    y = y_grid[0, :, 0]
    z = z_grid[0, 0, :]
    xy_x, xy_y = np.meshgrid(x, y, indexing='ij')
    xz_x, xz_z = np.meshgrid(x, z, indexing='ij')
    yz_y, yz_z = np.meshgrid(y, z, indexing='ij')

    # Calculate integration elements
    print('Calculating integration elements')
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    assert np.allclose(np.diff(x), dx)
    assert np.allclose(np.diff(y), dy)
    assert np.allclose(np.diff(z), dz)
    dxdy = dx * dy
    dxdz = dx * dz
    dydz = dy * dz
    dxdydz = dx * dy * dz

    # Normalise 3D posteriors
    print('Normalising')
    post_grids = [post_grid / (np.sum(post_grid) * dxdydz) for post_grid in post_grids]
    assert all([np.isclose(np.sum(post_grid) * dxdydz, 1) for post_grid in post_grids])

    # Marginalise to get 2D posteriors
    print('Marginalising 3D -> 2D')
    posts_xy = [np.sum(post_grid, axis=2) * dz for post_grid in post_grids]
    posts_xz = [np.sum(post_grid, axis=1) * dy for post_grid in post_grids]
    posts_yz = [np.sum(post_grid, axis=0) * dx for post_grid in post_grids]
    assert all([np.isclose(np.sum(post_xy) * dxdy, 1) for post_xy in posts_xy])
    assert all([np.isclose(np.sum(post_xz) * dxdz, 1) for post_xz in posts_xz])
    assert all([np.isclose(np.sum(post_yz) * dydz, 1) for post_yz in posts_yz])

    # Marginalise again to get 1D posteriors
    print('Marginalising 2D -> 1D')
    posts_x = [np.sum(post_xy, axis=1) * dy for post_xy in posts_xy]
    posts_y = [np.sum(post_xy, axis=0) * dx for post_xy in posts_xy]
    posts_z = [np.sum(post_xz, axis=0) * dx for post_xz in posts_xz]
    assert all([np.isclose(np.sum(post_x) * dx, 1) for post_x in posts_x])
    assert all([np.isclose(np.sum(post_y) * dy, 1) for post_y in posts_y])
    assert all([np.isclose(np.sum(post_z) * dz, 1) for post_z in posts_z])

    # Additional marginalisation checks
    print('Checking normalisation')
    assert all([np.allclose(post_x, np.sum(post_xz, axis=1) * dz) for post_x, post_xz in zip(posts_x, posts_xz)])
    assert all([np.allclose(post_y, np.sum(post_yz, axis=1) * dz) for post_y, post_yz in zip(posts_y, posts_yz)])
    assert all([np.allclose(post_z, np.sum(post_yz, axis=0) * dy) for post_z, post_yz in zip(posts_z, posts_yz)])
    assert all([np.allclose(post_x, np.sum(p_3d, axis=(1, 2)) * dydz) for post_x, p_3d in zip(posts_x, post_grids)])
    assert all([np.allclose(post_y, np.sum(p_3d, axis=(0, 2)) * dxdz) for post_y, p_3d in zip(posts_y, post_grids)])
    assert all([np.allclose(post_z, np.sum(p_3d, axis=(0, 1)) * dxdy) for post_z, p_3d in zip(posts_z, post_grids)])

    # Apply smoothing
    if smooth_xy is not None:
        posts_xy = [ndimage.gaussian_filter(post_xy, [sig, sig / 2.]) for post_xy, sig in zip(posts_xy, smooth_xy)]
    if smooth_xz is not None:
        posts_xz = [ndimage.gaussian_filter(post_xz, sig) for post_xz, sig in zip(posts_xz, smooth_xz)]
    if smooth_yz is not None:
        posts_yz = [ndimage.gaussian_filter(post_yz, sig) for post_yz, sig in zip(posts_yz, smooth_yz)]
    if smooth_x is not None:
        posts_x = [ndimage.gaussian_filter(post_x, sig) for post_x, sig in zip(posts_x, smooth_x)]
    if smooth_y is not None:
        posts_y = [ndimage.gaussian_filter(post_y, sig) for post_y, sig in zip(posts_y, smooth_y)]
    if smooth_z is not None:
        posts_z = [ndimage.gaussian_filter(post_z, sig) for post_z, sig in zip(posts_z, smooth_z)]

    # Convert 2D & 1D posteriors to confidence levels
    print('Converting to confidence levels')
    confs_xy = [gcl_post.post_to_conf(post_xy, dxdy) for post_xy in posts_xy]
    confs_xz = [gcl_post.post_to_conf(post_xz, dxdz) for post_xz in posts_xz]
    confs_yz = [gcl_post.post_to_conf(post_yz, dydz) for post_yz in posts_yz]
    confs_x = [gcl_post.post_to_conf(post_x, dx) for post_x in posts_x]
    confs_y = [gcl_post.post_to_conf(post_y, dy) for post_y in posts_y]
    confs_z = [gcl_post.post_to_conf(post_z, dz) for post_z in posts_z]

    # Extract out relative widths and areas
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]
    if print_areas:
        print('Note that smoothing should be switched off to extract unbiased relative areas, and smoothing should be '
              'set such that relative areas are preserved')
        def count_points_within_outermost_contour(conf_grid):
            return np.count_nonzero(conf_grid < contour_levels[-1])
        rel_areas_xy = list(map(count_points_within_outermost_contour, confs_xy))
        print('Relative areas x-y:', np.divide(rel_areas_xy, max(rel_areas_xy)))
        rel_areas_xz = list(map(count_points_within_outermost_contour, confs_xz))
        print('Relative areas x-z:', np.divide(rel_areas_xz, max(rel_areas_xz)))
        rel_areas_yz = list(map(count_points_within_outermost_contour, confs_yz))
        print('Relative areas y-z:', np.divide(rel_areas_yz, max(rel_areas_yz)))
        rel_widths_x = list(map(count_points_within_outermost_contour, confs_x))
        print('Relative widths x:', np.divide(rel_widths_x, max(rel_widths_x)))
        rel_widths_y = list(map(count_points_within_outermost_contour, confs_y))
        print('Relative widths y:', np.divide(rel_widths_y, max(rel_widths_y)))
        rel_widths_z = list(map(count_points_within_outermost_contour, confs_z))
        print('Relative widths z:', np.divide(rel_widths_z, max(rel_widths_z)))

    # Plot everything
    print('Plotting')
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['axes.titlesize'] = 17
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex='col', figsize=(12.8, 8.6))
    plt.subplots_adjust(left=.08, right=.97, bottom=.08, top=.97, wspace=0, hspace=0)
    fill_colours = [[np.squeeze(matplotlib.colors.to_rgba_array(c, a)) for a in [0.3, 0.1, 0]] for c in colours]

    # Row 0: x
    for post_x, colour, fill, linestyle, label in zip(posts_x, colours, fill_colours, linestyles, labels):
        axes[0, 0].plot(x, post_x, color=colour, ls=linestyle, lw=2, label=label)
        axes[0, 0].fill_between(x, post_x, color=fill[1])
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

    # Row 1: x vs y, y
    for conf_xy, post_y, colour, fill, linestyle in zip(confs_xy, posts_y, colours, fill_colours, linestyles):
        axes[1, 0].contour(xy_x, xy_y, conf_xy, levels=contour_levels, colors=colour, linestyles=[linestyle],
                           linewidths=2)
        axes[1, 0].contourf(xy_x, xy_y, conf_xy, levels=contour_levels, colors=fill)
        axes[1, 1].plot(y, post_y, color=colour, ls=linestyle, lw=2)
        axes[1, 1].fill_between(y, post_y, color=fill[1])
    axes[1, 2].axis('off')

    # Row 2: x vs z, y vs z, z
    for conf_xz, conf_yz, post_z, colour, fill, linestyle in zip(confs_xz, confs_yz, posts_z, colours, fill_colours,
                                                                 linestyles):
        axes[2, 0].contour(xz_x, xz_z, conf_xz, levels=contour_levels, colors=colour, linestyles=[linestyle],
                           linewidths=2)
        axes[2, 0].contourf(xz_x, xz_z, conf_xz, levels=contour_levels, colors=fill)
        axes[2, 1].contour(yz_y, yz_z, conf_yz, levels=contour_levels, colors=colour, linestyles=[linestyle],
                           linewidths=2)
        axes[2, 1].contourf(yz_y, yz_z, conf_yz, levels=contour_levels, colors=fill)
        axes[2, 2].plot(z, post_z, color=colour, ls=linestyle, lw=2)
        axes[2, 2].fill_between(z, post_z, color=fill[1])

    # Hide y ticks for 1D posteriors
    axes[0, 0].tick_params(axis='y', which='both', left=False, labelleft=False)
    axes[1, 1].tick_params(axis='y', which='both', left=False, labelleft=False)
    axes[2, 2].tick_params(axis='y', which='both', left=False, labelleft=False)

    # Add x ticks at top and bottom of 2D posteriors and at bottom of 1D posteriors
    axes[0, 0].tick_params(axis='x', which='both', bottom=True, direction='in')
    axes[1, 0].tick_params(axis='x', which='both', top=True, bottom=True, direction='in')
    axes[2, 0].tick_params(axis='x', which='both', top=True, bottom=True, direction='inout', length=7.5)
    axes[0, 1].tick_params(axis='x', which='both', bottom=True, direction='in')
    axes[2, 1].tick_params(axis='x', which='both', top=True, bottom=True, direction='inout', length=7.5)
    axes[2, 2].tick_params(axis='x', which='both', bottom=True, direction='inout', length=7.5)

    # Add y ticks at left and right of 2D posteriors
    axes[1, 0].tick_params(axis='y', which='both', left=True, direction='inout', length=7.5)
    axes[1, 0].secondary_yaxis('right').tick_params(axis='y', which='both', right=True, direction='in',
                                                    labelright=False)
    axes[2, 0].tick_params(axis='y', which='both', left=True, right=True, direction='inout', length=7.5)
    axes[2, 1].tick_params(axis='y', which='both', left=True, right=True, labelleft=False, direction='in')

    # Limits
    axes[2, 0].set_xlim(x_lims)
    axes[2, 1].set_xlim(y_lims)
    axes[2, 2].set_xlim(z_lims)
    axes[1, 0].set_ylim(y_lims)
    axes[2, 0].set_ylim(z_lims)
    axes[2, 1].set_ylim(z_lims)

    # Fix overlapping z tick labels by removing every other tick
    axes[2, 2].set_xticks(axes[2, 2].get_xticks()[1::2])

    # Label axes
    axes[2, 0].set_xlabel(x_label)
    axes[2, 1].set_xlabel(y_label)
    axes[2, 2].set_xlabel(z_label)
    axes[1, 0].set_ylabel(y_label)
    axes[2, 0].set_ylabel(z_label)
    fig.align_ylabels()

    # Title
    axes[0, 0].annotate('Full Euclid-like mask', xy=(2.95, .95), xycoords='axes fraction', ha='right',
                        va='top', size=plt.rcParams['axes.titlesize'])

    # Legend
    leg_title = f'{min(contour_levels_sig)}\N{en dash}{max(contour_levels_sig)}$\\sigma$ confidence'
    axes[0, 0].legend(loc='upper right', bbox_to_anchor=(3, .8), handlelength=4, frameon=False, title=leg_title)

    if save_path is not None:
        plt.savefig(save_path)
        print('Saved ' + save_path)
    else:
        plt.show()


def cov_diags(data_path, lmax, lmin, lmax_plot, roll_window_size, save_path=None):
    """
    Plot diagonals of the covariance matrix for each theory contribution and simulations, for each mask.

    Args:
        data_path (str): Path to prepared data as output by plot_utils.get_cov_diags.
        lmax (int): Maximum l in the data.
        lmin (int): Minimum l in the data.
        lmax_plot (int): Maximum l to include in the plot.
        roll_window_size (int): Number of ells to include in the rolling average.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure will be displayed.
    """

    # Load everything
    print('Loading data')
    with np.load(data_path, allow_pickle=True) as data:
        results = data['results']
        lmin = data['lmin']
        diags = data['diags']

    # Prepare plot
    n_diags = len(diags)
    n_masks = len(results)
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['lines.linewidth'] = 2.5
    fig, ax = plt.subplots(nrows=n_diags, ncols=n_masks, sharex=True, sharey='row',
                           figsize=(2/3 * n_masks * plt.figaspect(n_masks / n_diags)))
    plt.subplots_adjust(wspace=0, hspace=0)

    # Loop over masks and diagonals
    print('Plotting')
    for col, mask_results in enumerate(results):
        for row, (diag, diag_results) in enumerate(zip(diags, mask_results['results_per_diag'])):

            # Calculate rolling mean and standard deviation of the sims and ells
            roll = pd.Series(diag_results['sim']).rolling(roll_window_size)
            sim_avg = roll.mean().to_numpy()[(roll_window_size - 1):]
            sim_std = roll.std().to_numpy()[(roll_window_size - 1):]
            ell_full = np.arange(lmin, lmax - diag + 1)
            ell_roll = pd.Series(ell_full).rolling(roll_window_size).mean().to_numpy()[(roll_window_size - 1):]

            # Plot simulations
            ell_roll_mask = ell_roll <= lmax_plot
            ell_roll = ell_roll[ell_roll_mask]
            sim_avg = sim_avg[ell_roll_mask]
            sim_std = sim_std[ell_roll_mask]
            sim_std[1::2] *= -1
            sim_err = sim_avg + 0.5 * sim_std
            ax[row, col].plot(ell_roll, sim_err, c='C0', alpha=.2)
            ax[row, col].plot(ell_roll, sim_avg, c='C0', lw=(5 if diag == 0 else None), label='sim')

            # Theory components
            ell_cut = np.arange(lmin, lmax_plot - diag + 1)
            g = diag_results['g'][:(lmax_plot - diag - lmin + 1)]
            ss = diag_results['ss'][:(lmax_plot - diag - lmin + 1)]
            cng = diag_results['cng'][:(lmax_plot - diag - lmin + 1)]
            ax[row, col].plot(ell_cut, g, c='limegreen', label='g')
            ax[row, col].plot(ell_cut, ss, c='C3', label='ss')
            ax[row, col].plot(ell_cut, cng, c='C4', label='cng')
            ax[row, col].plot(ell_cut, g + ss + cng, c='C1', ls=(0, (5, 4)), label='tot')

            # Log scale for diagonal, otherwise dotted line at 0
            if diag == 0:
                ax[row, col].set_yscale('log')
            else:
                ax[row, col].axhline(0, ls='--', c='k', alpha=.5, lw=1)

    # Axis labels
    _ = [ax[-1, col].set_xlabel(r'$\ell$') for col in range(n_masks)]
    for row, diag in enumerate(diags):
        ax[row, 0].set_ylabel(r'Var($C_\ell$) / ${C_\ell}^2$' if diag == 0 else r'Corr($\ell$, $\ell + \Delta \ell$)')
    fig.align_ylabels()

    # Row and column titles
    _ = [ax[0, col].set_title(mask_results['mask_label']) for col, mask_results in enumerate(results)]
    _ = [row[0].annotate(f'$\\Delta \\ell =$ {diag}', xy=(-0.3, 0.5), xycoords='axes fraction', ha='right',
                         va='center', size=plt.rcParams['axes.titlesize']) for row, diag in zip(ax, diags)]

    # Legend - two legends with dummy handles to space out correctly
    handles, labels = ax[0, 0].get_legend_handles_labels()
    dummy_handle = matplotlib.lines.Line2D([0], [0], alpha=0)
    sim_handle = matplotlib.lines.Line2D([0], [0])
    sim_handle.update_from(handles[labels.index('sim')])      # best way I could find to 'copy' a Line2D object
    sim_handle.set_linewidth(plt.rcParams['lines.linewidth']) # so I can set the width just for the legend
    top_handles = [sim_handle]
    top_labels = ['Simulations (rolling mean + std. dev.)']
    bottom_handles = [dummy_handle, handles[labels.index('g')], dummy_handle, handles[labels.index('ss')], dummy_handle,
                      handles[labels.index('cng')], dummy_handle, handles[labels.index('tot')]]
    bottom_labels = [' ', 'Gaussian', ' ', 'Super-sample', ' ', 'Connected non-Gaussian', ' ', 'Total theory']
    leg_bottom = ax[0, n_masks // 2].legend(bottom_handles, bottom_labels, loc='lower center',
                                            bbox_to_anchor=(0.5, 1.2), ncol=4, handlelength=3)
    ax[0, n_masks // 2].legend(top_handles, top_labels, loc='lower center', bbox_to_anchor=(0.5, 1.34), frameon=False,
                               handlelength=3)
    ax[0, n_masks // 2].add_artist(leg_bottom)

    # Ticks
    _ = [a.tick_params(axis='both', which='both', direction='in', length=3, bottom=True, left=True, top=True,
                       right=True) for row in ax for a in row]
    _ = [row[0].tick_params(axis='y', which='both', direction='inout', length=6, left=True) for row in ax]
    _ = [ax[-1, col].tick_params(axis='x', which='both', direction='inout', length=6) for col in range(n_masks)]

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def cov_mats(data_path, lmax_plot, lmin, row_order, row_labels, downsample_fac=30, save_path=None):
    """
    Plot correlation matrices for each theory contribution and covariance, for each mask.

    Args:
        data_path (str): Path to prepared data as output by plot_utils.get_cov_mats.
        lmax_plot (int): Maximum l to include in the plot.
        lmin (int): Minimum l in the data.
        row_order (list): Order for rows as they are labelled in the input data,
                          e.g. ['corr_g', 'corr_ss', 'corr_cng', 'corr_tot', 'sim_corr'].
        row_labels (list): Label for each row.
        downsample_fac (int, optional): Factor by which to downsample the matrices for plotting, default 30.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure will be displayed.
    """

    # Load everything
    print('Loading data')
    with np.load(data_path, allow_pickle=True) as data:
        results = data['results']

    # Trim to lmax_plot
    n_masks = len(results)
    n_ell_plot = lmax_plot - lmin + 1
    for mask_no, mask_results in enumerate(results, 1):
        print(f'Trimming for mask {mask_no} / {n_masks}')
        for corr in row_order:
            mask_results[corr] = mask_results[corr][:n_ell_plot, :n_ell_plot]

    # Plot all matrices with one column per mask
    print('Plotting')
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.titlesize'] = 17
    _, ax = plt.subplots(ncols=n_masks, nrows=5, sharex=True, sharey=True, figsize=(12.8, 26.5))
    plt.subplots_adjust(wspace=0, hspace=0)
    vmin, vmax = 0, 1
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    imshow_args = {'norm': norm, 'origin': 'lower', 'cmap': 'cividis', 'interpolation': 'none',
                   'extent': (lmin, lmax_plot, lmin, lmax_plot)}
    for col, mask_results in enumerate(results):

        # Due to numerical imprecision full-sky Gaussian is not-quite diagonal, which causes problems for plotting
        if mask_results['mask_label'] == 'Full sky':
            mask_results['corr_g'] = np.where(mask_results['corr_g'] < vmin, vmin, mask_results['corr_g'])

        for row, corr in enumerate(row_order):
            # Downsample
            thresh = 0.6 if corr == 'sim_corr' else 0
            to_plot = skimage.measure.block_reduce(mask_results[corr], (downsample_fac, downsample_fac),
                                                   plot_utils.cov_pool, cval=np.nan, func_kwargs={'threshold': thresh})
            im = ax[row, col].imshow(to_plot, **imshow_args)

    # Axis labels
    _ = [ax[-1, col].set_xlabel(r'$\ell$') for col in range(n_masks)]
    _ = [row[0].set_ylabel(r"$\ell'$") for row in ax]

    # Colour bar
    plt.colorbar(im, ax=ax, location='bottom', shrink=0.8, aspect=30, pad=0.04, label=r"Corr($\ell$, $\ell'$)")

    # Row/col titles
    _ = [ax[0, col].set_title(mask_results['mask_label'], pad=12) for col, mask_results in enumerate(results)]
    for row, label in enumerate(row_labels):
        ax[row, 0].annotate(label, xy=(-0.3, 0.5), xycoords='axes fraction', ha='right', va='center',
                            size=plt.rcParams['axes.titlesize'])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_masks(full_euclid_mask_path, dr1_mask_path, full_euclid_coord='E', dr1_coord='G', save_path=None):
    """
    Plot the full Euclid-like mask and Euclid DR1-like masks together, in galactic coordinates.

    Args:
        full_euclid_mask_path (str): Path to full Euclid-like mask.
        dr1_mask_path (str): Path to Euclid DR1-like mask.
        full_euclid_coord (str, optional): Input coordinates for full Euclid-like mask: 'E' for ecliptic,
                                           'G' for galactic. Default 'E'.
        dr1_coord (str, optional): Input coordinates for Euclid DR1-like mask: 'E' for ecliptic, 'G' for galactic.
                                   Default 'G'.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure will be displayed.
    """

    # Load masks
    euclid_mask = hp.fitsfunc.read_map(full_euclid_mask_path, dtype=float, verbose=False)
    dr1_mask = hp.fitsfunc.read_map(dr1_mask_path, dtype=float, verbose=False)

    # Calculate Mollweide projections into galactic coordinates if not already
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # mollview's plotting code creates warnings
        euclid_proj = hp.visufunc.mollview(euclid_mask, coord=[full_euclid_coord, 'G'], return_projected_map=True)
        plt.close()
        dr1_proj = hp.visufunc.mollview(dr1_mask, coord=[dr1_coord, 'G'], return_projected_map=True)
        plt.close()

    # Plot side-by-side
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.titlesize'] = 17
    _, ax = plt.subplots(ncols=2, figsize=(12.8, 3.6))
    plt.subplots_adjust(left=0.02, right=0.97, bottom=0, top=1, wspace=0.04, hspace=0)
    cmap = copy.copy(matplotlib.cm.get_cmap('cividis'))
    cmap.set_bad(color='white')
    titles = ['Full Euclid-like', 'Euclid DR1-like']
    for a, proj, title in zip(ax, [euclid_proj, dr1_proj], titles):
        im = a.imshow(proj, origin='lower', cmap=cmap, interpolation='none')
        a.axis('off')
        a.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.65, aspect=10, fraction=0.05, pad=0.03)

    # Save or show
    if save_path is not None:
        plt.savefig(save_path)
        print('Saved ' + save_path)
    else:
        plt.show()


def post_2d(panels, labels, colours, linestyles, contour_levels_sig, column_titles, param_labels, print_areas=False,
            save_path=None):
    """
    Plot a grid of 2D posteriors, each panel itself containing multiple posterior distributions.

    Args:
        panels (list): Configuration of the panels, as a list of lists of dicts.
                       The top-level list defines the rows; the middle-level list defines the panels within each row;
                       the dict contains the config for each panel.
                       The dict has the following keys:
                       ``like_paths`` (list) - list of paths to each log-likelihood txt file;
                       ``xlims`` (float, float) - x-axis limits;
                       ``ylims`` (float, float) - y-axis limits;
                       ``smoooth_sigma`` (list) - kernel standard deviation for Gaussian smoothing for each posterior.
        labels (list): List of legend labels for each posterior (same for every panel).
        colours (list): List of colours for each posterior (same for every panel).
        linestyles (list): List of linestyles for each posterior (same for every panel).
        contour_levels_sig (list): List of confidence regions to plot in ascending order, e.g. [1, 3].
        column_titles (list): List of column titles.
        param_labels (list): Axis labels for the x, y, z parameters.
        print_areas (bool, optional): If True, print relative areas of the different posteriors. Note that smoothing can
                                      affect these results, so for reliable results smoothing should be switched off to
                                      extract relative areas, and then smoothing values should be set to preserve
                                      unsmoothed relative areas. Default False.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Obtain dimensions and check for consistent input
    n_rows = len(panels)
    assert n_rows == len(param_labels)
    n_cols = len(column_titles)
    assert all(n_cols == len(row) for row in panels)

    # Calculate contour levels
    if contour_levels_sig is not None:
        contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]
    else:
        contour_levels = None

    # Make plot
    n_panels = n_rows * n_cols
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.titlesize'] = 17
    _, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12.8, 9.2))
    plt.subplots_adjust(left=.065, right=.999, bottom=.08, top=.83, wspace=.20, hspace=.3)

    # Iterate over panels
    for row_idx, (row, row_config) in enumerate(zip(ax, panels)):
        for col_idx, (panel, panel_config) in enumerate(zip(row, row_config)):
            print(f'Preparing panel {row_idx * n_cols + col_idx + 1} / {n_panels}')

            # Load data and form various 'flat' lists (i.e. not nested)
            x_vals = []
            y_vals = []
            log_likes = []
            for like_path in panel_config['like_paths']:

                # Determine how many columns to read by whether like path is a single path or a list
                n_like_cols = len(like_path) if isinstance(like_path, list) else 1

                # Load in the x and y columns and the appropriate number of log-likelihoods
                data = np.loadtxt(like_path)
                x = data[:, 0]
                y = data[:, 1]
                for col in range(n_like_cols):
                    log_like = data[:, 2 + col]
                    x_vals.append(x)
                    y_vals.append(y)
                    log_likes.append(log_like)

            # Beyond this point everything is a flat list, meaning no nested lists
            # and no knowledge of which columns came from which files

            # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
            posts = []
            for log_like in log_likes:
                posts.append(np.exp(log_like - np.amax(log_like) + 100))

            # Form x and y grids and determine grid cell size (requires and checks for regular grid)
            dxdys = []
            x_grids = []
            y_grids = []
            for x_val, y_val in zip(x_vals, y_vals):
                x_val_unique = np.unique(x_val)
                dx = x_val_unique[1] - x_val_unique[0]
                assert np.allclose(np.diff(x_val_unique), dx)
                y_val_unique = np.unique(y_val)
                dy = y_val_unique[1] - y_val_unique[0]
                assert np.allclose(np.diff(y_val_unique), dy)
                dxdys.append(dx * dy)

                x_grid, y_grid = np.meshgrid(x_val_unique, y_val_unique)
                x_grids.append(x_grid)
                y_grids.append(y_grid)

            # Grid, apply smoothing and convert to confidence intervals
            conf_grids = []
            for x_val, y_val, post, x_grid, y_grid, dxdy, sig in zip(x_vals, y_vals, posts, x_grids, y_grids, dxdys,
                                                                     panel_config['smooth_sigma']):
                post_grid = scipy.interpolate.griddata((x_val, y_val), post, (x_grid, y_grid), fill_value=0)
                post_grid = ndimage.gaussian_filter(post_grid, [sig, sig / 2.])
                if contour_levels_sig is not None:
                    conf_grids.append(gcl_post.post_to_conf(post_grid, dxdy))
                else:
                    conf_grids.append(post_grid) # Plot raw posterior

            # Print relative areas of outermost contours for each likelihood, normalised to the largest one
            # For this to be reliable smoothing should be switched off first
            if print_areas:
                print('Note that smoothing should be switched off to extract unbiased relative areas, and smoothing '
                      'should be set such that relative areas are preserved')
                def count_points_within_outermost_contour(conf_grid):
                    return np.count_nonzero(conf_grid < contour_levels[-1])
                points_within_outermost_contour = list(map(count_points_within_outermost_contour, conf_grids))
                print('Relative areas:',
                      np.divide(points_within_outermost_contour, max(points_within_outermost_contour)))

            # Plot contours
            contours = []
            for x_grid, y_grid, conf_grid, colour, linestyle in zip(x_grids, y_grids, conf_grids, colours, linestyles):
                cont = panel.contour(x_grid, y_grid, conf_grid, levels=contour_levels, colors=colour,
                                     linestyles=linestyle, linewidths=2)
                contours.append(cont)

                # Custom fill
                alphas = [0.3, 0.1, 0]
                colours = [np.squeeze(matplotlib.colors.to_rgba_array(colour, alpha)) for alpha in alphas]
                panel.contourf(x_grid, y_grid, conf_grid, levels=contour_levels, colors=colours)

            # Limits
            panel.set_xlim(panel_config['xlims'])
            panel.set_ylim(panel_config['ylims'])

    # Axis labels
    for row, param_labels in zip(ax, param_labels):
        _ = [panel.set_xlabel(param_labels[0]) for panel in row]
        row[0].set_ylabel(param_labels[1])

    # Column titles
    for col_idx, title in enumerate(column_titles):
        ax[0, col_idx].set_title(title, pad=20)

    # Legend
    if contour_levels_sig is None:
        leg_title = '(arbitrary contours)'
    elif len(contour_levels_sig) > 1:
        leg_title = f'{min(contour_levels_sig)}\N{en dash}{max(contour_levels_sig)}$\\sigma$ confidence'
    else:
        leg_title = f'{contour_levels_sig[0]}$\\sigma$ confidence'
    handles = [cont.legend_elements()[0][0] for cont in contours] # Just want one handle per set of contours
    ax[0, n_cols // 2].legend(handles, labels, title=leg_title, handlelength=3, loc='lower center',
                              bbox_to_anchor=(0.5, 1.23), ncol=4)

    # Save or show
    if save_path is not None:
        plt.savefig(save_path)
        print('Saved ' + save_path)
    else:
        plt.show()


def cng_approx(bin_weight_ratios_path, mix_weight_ratios_path, save_path=None):
    """
    Plot histograms of test ratios for the connected non-Gaussian approximation as obtained using
    cng_approx.test_bin_weights and cng_approx.test_mix_weights.

    Args:
        bin_weight_ratios_path: Path to binning weight test ratios output by cng_approx.test_bin_weights.
        mix_weight_ratios_path: Path to mixing weight test ratios output by cng_approx.test_mix_weights.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Load data and flatten
    with np.load(bin_weight_ratios_path) as data:
        ratios_part1 = np.ravel(data['ratios'])
    with np.load(mix_weight_ratios_path) as data:
        ratios_part2 = np.ravel(data['ratios'])

    # Plot histograms
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['lines.linewidth'] = 2.5
    _, ax = plt.subplots(ncols=2, figsize=(12.8, 4))
    plt.subplots_adjust(bottom=0.18, top=0.85, wspace=0.2, left=0.15, right=0.9)

    for a, ratios in zip(ax, (ratios_part1, ratios_part2)):
        a.hist(ratios, bins=50, histtype='stepfilled', color='cadetblue')
        a.axvline(np.mean(ratios[np.isfinite(ratios)]), ls='--', c='k', label='Mean')

    # Titles
    ax[0].set_title('Binning  ', loc='right', y=0.85)
    ax[1].set_title('  Mixing', loc='left', y=0.85)

    # Axis labels
    _ = [a.set_xlabel(r'$\mathrm{Cov}_\mathrm{approx}$ / $\mathrm{Cov}_\mathrm{exact}$') for a in ax]
    ax[0].set_ylabel('Number of elements')

    # Legend
    ax[0].legend(loc='lower center', bbox_to_anchor=(1.1, 1), handlelength=3)

    # Save or show
    if save_path is not None:
        plt.savefig(save_path)
        print('Saved ' + save_path)
    else:
        plt.show()


def cov_withnoise(data_path, lmax_plot, save_path=None):
    """
    Plot diagonals of the covariance matrix with noise included for each theory contribution (no simulations),
    for each mask.

    Args:
        data_path (str): Path to prepared data as output by plot_utils.get_cov_diags_withnoise.
        lmax_plot (int): Maximum l to include in the plot.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure will be displayed.
    """

    # Load everything
    print('Loading data')
    with np.load(data_path, allow_pickle=True) as data:
        results = data['results']
        lmin = data['lmin']
        diags = data['diags']

    # Prepare plot
    n_diags = len(diags)
    n_masks = len(results)
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['lines.linewidth'] = 2.5
    fig, ax = plt.subplots(nrows=n_diags, ncols=n_masks, sharex=True, sharey='row',
                           figsize=(2/3 * n_masks * plt.figaspect(n_masks / n_diags)))
    plt.subplots_adjust(wspace=0, hspace=0)

    # Loop over masks and diagonals
    print('Plotting')
    for col, mask_results in enumerate(results):
        for row, (diag, diag_results) in enumerate(zip(diags, mask_results['results_per_diag'])):

            # Theory components
            ell_cut = np.arange(lmin, lmax_plot - diag + 1)
            g = diag_results['g'][:(lmax_plot - diag - lmin + 1)]
            ss = diag_results['ss'][:(lmax_plot - diag - lmin + 1)]
            cng = diag_results['cng'][:(lmax_plot - diag - lmin + 1)]
            ax[row, col].plot(ell_cut, g, c='limegreen', label='Gaussian')
            ax[row, col].plot(ell_cut, ss, c='C3', label='Super-sample')
            ax[row, col].plot(ell_cut, cng, c='C4', label='Connected non-Gaussian')
            ax[row, col].plot(ell_cut, g + ss + cng, c='C1', ls=(0, (5, 4)), label='Total theory')

            # Log scale for diagonal, otherwise dotted line at 0
            if diag == 0:
                ax[row, col].set_yscale('log')
            else:
                ax[row, col].axhline(0, ls='--', c='k', alpha=.5, lw=1)

    # Axis labels
    _ = [ax[-1, col].set_xlabel(r'$\ell$') for col in range(n_masks)]
    for row, diag in enumerate(diags):
        ax[row, 0].set_ylabel(r'Var($C_\ell$)' if diag == 0 else r'Corr($\ell$, $\ell + \Delta \ell$)')
    fig.align_ylabels()

    # Row and column titles
    _ = [ax[0, col].set_title(mask_results['mask_label']) for col, mask_results in enumerate(results)]
    _ = [row[0].annotate(f'$\\Delta \\ell =$ {diag}', xy=(-0.3, 0.5), xycoords='axes fraction', ha='right',
                         va='center', size=plt.rcParams['axes.titlesize']) for row, diag in zip(ax, diags)]

    # Legend - two legends with dummy handles to space out correctly
    ax[0, n_masks // 2].legend(loc='lower center', bbox_to_anchor=(0.5, 1.2), ncol=4, handlelength=3)

    # Ticks
    _ = [a.tick_params(axis='both', which='both', direction='in', length=3, bottom=True, left=True, top=True,
                       right=True) for row in ax for a in row]
    _ = [row[0].tick_params(axis='y', which='both', direction='inout', length=6, left=True) for row in ax]
    _ = [ax[-1, col].tick_params(axis='x', which='both', direction='inout', length=6) for col in range(n_masks)]

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def cov_gaussian(data_path, lmax, lmin, roll_window_size, save_path=None):
    """
    Plot diagonals of the Gaussian covariance matrix compared to simulations, for each mask.

    Args:
        data_path (str): Path to prepared data as output by plot_utils.get_cov_diags_gaussian.
        lmax (int): Maximum l.
        lmin (int): Minimum l.
        roll_window_size (int): Number of ells to include in the rolling average.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure will be displayed.
    """

    # Load everything
    print('Loading data')
    with np.load(data_path, allow_pickle=True) as data:
        results = data['results']
        diags = data['diags']

    # Prepare plot
    n_diags = len(diags)
    n_masks = len(results)
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['lines.linewidth'] = 2.5
    fig, ax = plt.subplots(nrows=n_diags, ncols=n_masks, sharex=True, sharey='row',
                           figsize=(2/3 * n_masks * plt.figaspect(n_masks / n_diags)))
    plt.subplots_adjust(wspace=0, hspace=0)

    # Loop over masks and diagonals
    print('Plotting')
    for col, mask_results in enumerate(results):
        for row, (diag, diag_results) in enumerate(zip(diags, mask_results['results_per_diag'])):

            # Calculate rolling mean and standard deviation of the sims and ells
            roll = pd.Series(diag_results['sim']).rolling(roll_window_size)
            sim_avg = roll.mean().to_numpy()[(roll_window_size - 1):]
            sim_std = roll.std().to_numpy()[(roll_window_size - 1):]
            ell_full = np.arange(lmin, lmax - diag + 1)
            ell_roll = pd.Series(ell_full).rolling(roll_window_size).mean().to_numpy()[(roll_window_size - 1):]

            # Plot simulations
            ell_roll_mask = ell_roll <= lmax
            ell_roll = ell_roll[ell_roll_mask]
            sim_avg = sim_avg[ell_roll_mask]
            sim_std = sim_std[ell_roll_mask]
            sim_std[1::2] *= -1
            sim_err = sim_avg + 0.5 * sim_std
            ax[row, col].plot(ell_roll, sim_err, c='C0', alpha=.2)
            ax[row, col].plot(ell_roll, sim_avg, c='C0', lw=(5 if diag == 0 else None), label='sim')

            # Theory components
            ell_cut = np.arange(lmin, lmax - diag + 1)
            g = diag_results['g'][:(lmax - diag - lmin + 1)]
            ax[row, col].plot(ell_cut, g, c='limegreen', label='g')

            # Log scale for diagonal, otherwise dotted line at 0
            if diag == 0:
                ax[row, col].set_yscale('log')
            else:
                ax[row, col].axhline(0, ls='--', c='k', alpha=.5, lw=1)

    # Axis labels
    _ = [ax[-1, col].set_xlabel(r'$\ell$') for col in range(n_masks)]
    for row, diag in enumerate(diags):
        ax[row, 0].set_ylabel(r'Var($C_\ell$)' if diag == 0 else r'Corr($\ell$, $\ell + \Delta \ell$)')
    fig.align_ylabels()

    # Row and column titles
    _ = [ax[0, col].set_title(mask_results['mask_label']) for col, mask_results in enumerate(results)]
    _ = [row[0].annotate(f'$\\Delta \\ell =$ {diag}', xy=(-0.3, 0.5), xycoords='axes fraction', ha='right',
                         va='center', size=plt.rcParams['axes.titlesize']) for row, diag in zip(ax, diags)]

    # Legend
    orig_handles, orig_labels = ax[0, 0].get_legend_handles_labels()
    sim_handle = matplotlib.lines.Line2D([0], [0])
    sim_handle.update_from(orig_handles[orig_labels.index('sim')]) # best way I could find to 'copy' a Line2D object
    sim_handle.set_linewidth(plt.rcParams['lines.linewidth'])      # so I can set the width just for the legend
    handles = [sim_handle, orig_handles[orig_labels.index('g')]]
    labels = ['Gaussian simulations (rolling mean + std. dev.)', 'Improved NKA']
    ax[0, n_masks // 2].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.2), handlelength=3, ncol=2)

    # Ticks
    _ = [a.tick_params(axis='both', which='both', direction='in', length=3, bottom=True, left=True, top=True,
                       right=True) for row in ax for a in row]
    _ = [row[0].tick_params(axis='y', which='both', direction='inout', length=6, left=True) for row in ax]
    _ = [ax[-1, col].tick_params(axis='x', which='both', direction='inout', length=6) for col in range(n_masks)]

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()
