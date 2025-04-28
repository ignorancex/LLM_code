#!/usr/bin/env python3

# Place import files below
import h5py
import matplotlib.pyplot as plt
import numpy as np

from common_functions import save_figures
from universal_settings import (MV_bins, generated_data_file_template, sim_ids,
                                sim_n_part, sim_styles)


def main():
    # Plot settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    mv_xlims = [-16.5, -11.]
    min_mv = -7
    percentile = [16., 84.]

    # File locations
    mock_output_file = 'fig5_mock_sdss_LFs.pdf'

    ####################################################################
    # Plot mock SDSS observations
    ####################################################################

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    for sim_id in sim_ids:
        with h5py.File(generated_data_file_template.format(sim_n_part, sim_id),
                       'r') as f:
            # Read in mock SDSS observations from generated file
            sdss_mock_lfs_mw = f['MW/SDSS UDG LF/Selection 4'][()]
            sdss_mock_lfs_m31 = f['M31/SDSS UDG LF/Selection 4'][()]

        # Median observations by halo analogue
        med_mock_mw = np.nanmedian(sdss_mock_lfs_mw, axis=0)
        med_mock_m31 = np.nanmedian(sdss_mock_lfs_m31, axis=0)

        # Scatter in MW mock observations
        spread_mock_mw = np.nanpercentile(sdss_mock_lfs_mw, percentile, axis=0)

        # Data to plot
        plot_MV_bins = np.concatenate((MV_bins, [min_mv]))
        plot_med_mock_mw = np.concatenate((med_mock_mw, [med_mock_mw[-1]]))
        plot_med_mock_m31 = np.concatenate((med_mock_m31, [med_mock_m31[-1]]))
        plot_spread_mock_mw = np.column_stack(
            (spread_mock_mw, spread_mock_mw[:, -1]))

        # Plot median line for MW and M31 mock data
        line, = ax.plot(plot_MV_bins,
                        plot_med_mock_mw,
                        color=sim_styles[sim_id]['color'],
                        label=r'${}$'.format(str(sim_id.replace('_', '\_'))))
        ax.plot(plot_MV_bins,
                plot_med_mock_m31,
                linestyle='--',
                color=line.get_color())

        # Plot scatter for MW data only
        ax.fill_between(plot_MV_bins,
                        *plot_spread_mock_mw,
                        facecolor=line.get_color(),
                        edgecolor='None',
                        alpha=0.2,
                        zorder=0)

    ####################################################################
    # Axis settings
    ####################################################################
    ax.set(xlabel=r'$M_{\rm V}$',
           ylabel=r'$N_{\rm UDG}\!\left(<\, M_{\rm V}\right)$',
           xlim=mv_xlims,
           ylim=[0, None])
    ax.minorticks_on()
    ax.tick_params(axis='x', which='major', pad=7)
    ax.invert_xaxis()

    ####################################################################
    # Add legends
    ####################################################################
    # Colour-coded simulation legend
    orig_leg = ax.legend(loc='upper right', handlelength=0, handletextpad=0)
    for t_item, h_item, sim_id in zip(orig_leg.get_texts(),
                                      orig_leg.legend_handles, sim_ids):
        t_item.set_color(h_item.get_color())
        h_item.set_visible(False)
    # Analogue halo legend in lower panel
    ax.legend([
        plt.Line2D((0, 1), (0, 0), color='k', linestyle='-'),
        plt.Line2D((0, 1), (0, 0), color='k', linestyle='--'),
    ], ['MW', 'M31'],
              loc='upper center')
    ax.add_artist(orig_leg)

    save_figures(fig, mock_output_file)

    return None


if __name__ == '__main__':
    main()
