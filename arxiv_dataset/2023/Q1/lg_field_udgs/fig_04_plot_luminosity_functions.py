#!/usr/bin/env python3

# Place import files below
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from common_functions import save_figures
from universal_settings import (des_app_mv_lim, generated_data_file_template,
                                hsc_app_mv_lim, lsst_app_mv_lim,
                                sdss_app_mv_lim, sim_ids, sim_n_part,
                                sim_styles)


def main():
    # Plot settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    apparent_lf_ylims = [0., 27.]
    abs_mv_xlims = [-17.5, -8.]
    abs_lf_ylims = [0, 1.05]

    # File locations
    lf_abs_app_file = 'fig4_abs_app_LFs.pdf'

    ################################################################
    # Combined absolute and apparent LFs (by sim)
    ################################################################
    fig, (ax_abs, ax_app) = plt.subplots(1,
                                         2,
                                         figsize=(16, 8),
                                         gridspec_kw={'hspace': 0})

    for sim_id in sim_ids:
        print(sim_id)
        with h5py.File(generated_data_file_template.format(sim_n_part, sim_id),
                       'r') as f:
            # Read in absolute V-band luminosity functions
            # Total, SDSS, DES, LSST
            (abs_mv, n_udg_tot, n_udg_sdss, n_udg_des, n_udg_lsst) = list(
                zip(*f['LG/UDG absolute V-band LFs/Selection 4'][()]))

            # Read in apparent V-band luminosity functions
            # MW
            (app_mv_mw,
             n_udg_mw) = list(zip(*f['MW/Apparent UDG LF/Selection 4'][()]))
            # M31
            (app_mv_m31,
             n_udg_m31) = list(zip(*f['M31/Apparent UDG LF/Selection 4'][()]))

        ################################################################
        # Print relevant information
        ################################################################
        max_udgs = np.nanmax(n_udg_tot)
        print("f_UDGs in SDSS: {}".format(np.nanmax(n_udg_sdss) / max_udgs))
        print("f_UDGs in DES: {}".format(np.nanmax(n_udg_des) / max_udgs))
        print("f_UDGs in LSST: {}".format(np.nanmax(n_udg_lsst) / max_udgs))
        print("N_UDGs,tot: {}".format(max_udgs))

        # Plot line for the total UDG population (idealized, face-on)
        total_line, = ax_abs.plot(abs_mv,
                                  n_udg_tot / max_udgs,
                                  drawstyle='steps-post',
                                  color=sim_styles[sim_id]['color'],
                                  label=r'{}'.format(sim_id.replace('_',
                                                                    '\_')))
        # Plot lines for the UDG population detectable in surveys
        ax_abs.plot(abs_mv,
                    n_udg_sdss / max_udgs,
                    linestyle=':',
                    color=total_line.get_color(),
                    drawstyle='steps-post')
        ax_abs.plot(abs_mv,
                    n_udg_des / max_udgs,
                    linestyle='--',
                    color=total_line.get_color(),
                    drawstyle='steps-post')
        ax_abs.plot(abs_mv,
                    n_udg_lsst / max_udgs,
                    linestyle='-.',
                    color=total_line.get_color(),
                    drawstyle='steps-post')

        # Plot apparent V-band mag LF of UDG population wrt. MW and M31
        mw_line, = ax_app.plot(app_mv_mw,
                               n_udg_mw,
                               drawstyle='steps-post',
                               color=sim_styles[sim_id]['color'],
                               label=r'{}'.format(sim_id.replace('_', '\_')))
        ax_app.plot(app_mv_m31,
                    n_udg_m31,
                    linestyle='--',
                    color=mw_line.get_color(),
                    drawstyle='steps-post')

    # Survey apparent V-band magnitude limits
    survey_app_lim_labels = ['SDSS', 'DES', 'HSC', 'LSST']
    survey_app_lims = [
        sdss_app_mv_lim, des_app_mv_lim, hsc_app_mv_lim, lsst_app_mv_lim
    ]
    survey_text_loc = [val - 0.1 for val in survey_app_lims]
    survey_text_horiz_align = ['left', 'left', 'left', 'left']

    # Plot arrows marking survey magnitude limits
    arrow_length = 2.
    for i in range(len(survey_app_lims)):
        ax_app.arrow(survey_app_lims[i],
                     apparent_lf_ylims[1],
                     0,
                     -arrow_length,
                     head_width=arrow_length * 0.15,
                     head_length=arrow_length * 0.31,
                     fc='k',
                     shape='full',
                     length_includes_head=True)
        ax_app.text(survey_text_loc[i],
                    apparent_lf_ylims[1] - arrow_length * 0.25,
                    survey_app_lim_labels[i],
                    fontsize=16,
                    horizontalalignment=survey_text_horiz_align[i],
                    verticalalignment='top')

    ####################################################################
    # Axis settings
    ####################################################################
    # Absolute LF axis
    ax_abs.set(xlabel=r'$M_{\rm V}$',
               ylabel=r'$N_{\rm UDG}\!\left(<\, M_{\rm V}\right)\, /\, ' +
               r'N_{\rm UDG,\, tot}$',
               xlim=abs_mv_xlims,
               ylim=abs_lf_ylims)
    ax_abs.minorticks_on()
    ax_abs.tick_params(axis='x', which='major', pad=7)

    # Add legends
    ax_abs.legend([
        plt.Line2D((0, 1), (0, 0), color='k', linestyle='-'),
        plt.Line2D((0, 1), (0, 0), color='k', linestyle=':'),
        plt.Line2D((0, 1), (0, 0), color='k', linestyle='--'),
        plt.Line2D((0, 1), (0, 0), color='k', linestyle='-.'),
    ], ['Total', 'SDSS', 'DES', 'LSST'],
                  loc='lower left')
    ax_abs.invert_xaxis()

    ####################################################################
    # Apparent LF axis
    ax_app.set(xlabel=r'$m_{\rm V}$',
               ylabel=r'$N_{\rm UDG}\!\left(<\, m_{\rm V}\right)$',
               ylim=apparent_lf_ylims)
    ax_app.xaxis.set_major_locator(MultipleLocator(2.))
    ax_app.minorticks_on()
    ax_app.tick_params(axis='x', which='major', pad=7)
    ax_app.invert_xaxis()

    # Add legends
    orig_leg = ax_app.legend(loc='upper right',
                             handlelength=0,
                             handletextpad=0)
    for t_item, h_item, sim_id in zip(orig_leg.get_texts(),
                                      orig_leg.legend_handles, sim_ids):
        t_item.set_color(h_item.get_color())
        h_item.set_visible(False)
    ax_app.legend([
        plt.Line2D((0, 1), (0, 0), color='k', linestyle='-'),
        plt.Line2D((0, 1), (0, 0), color='k', linestyle='--'),
    ], ['MW', 'M31'],
                  loc='lower left')
    ax_app.add_artist(orig_leg)

    save_figures(fig, lf_abs_app_file)

    return None


if __name__ == "__main__":
    main()
