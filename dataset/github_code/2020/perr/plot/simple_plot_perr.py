from __future__ import print_function, division

from collections import OrderedDict
import cPickle
import glob
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import spines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np
import os
import random
import re
from scipy import interpolate as interp
from scipy.io import savemat
import sys
import ujson

import constants
import logbook_main_calc_Perr_pickles
from lsstools.results_db import retrieve_pickles
from lsstools.results_db.io import Pickler
from perr.path_utils import get_in_path


def main():
    """
    Read results from pickle file and plot P_error / Poisson prediction.
    """

    # ##########################################################################
    # OPTIONS
    # ##########################################################################

    # path of pickles
    pickle_path = os.path.expandvars('$SCRATCH/perr/pickle/')

    ## Which pickle to load. Edit logbook....py to select a pickle.
    loginfo = logbook_main_calc_Perr_pickles.get_pickle_fname()
    fname = loginfo['fname']
    plot_opts = loginfo['plot_opts']

    # realizations to average
    sim_seeds = [400, 403]

    # add title, legend
    show_title = True
    show_legend = False

    # save plots as pdf or show them
    save_plots = True


    # ##########################################################################
    # START PROGRAM
    # ##########################################################################

    # Default args for varying base params and stacking.
    # Will vary params (halo mass) and stack realizations in plot below.
    default_stack_and_vary_kwargs = dict(
        pickle_path=pickle_path,
        base_fname=fname,
        comp_key='opts',
        fname_pattern=r'^main_calc_Perr.*.dill$',
        ignore_pickle_keys=['pickle_fname', 'out_rho_path', 'in_path', 'cats'],
        skip_keys_when_stacking=['opts', 'flat_opts', 'cat_infos'],
        return_base_vp_pickles=True,
        return_base_vp_pickle_opts=True,
        get_in_path=get_in_path)

    # ######################################################################
    # Plot Perr/P_poisson for different halo mass bins
    # ######################################################################

    ## params to vary for different curves: difft mass bins
    varyparams = OrderedDict()
    #halo_masses = [(10.8, 11.8), (11.8, 12.8), (12.8, 13.8), (13.8, 15.1)]
    halo_masses = [(12.8,13.8)]
    halo_mass_strings = ['%.1f_%.1f' % (hm[0], hm[1]) for hm in halo_masses]
    for halo_mass_string in halo_mass_strings:
        varyparams[((('sim_opts', 'halo_mass_string',),
                     halo_mass_string),)] = dict()
    del halo_mass_string

    # get stacked pickles for each entry of varyparams
    stacked_vp_pickles, base_vp_pickles, base_vp_pickles_flat_opts = (
        retrieve_pickles.get_stacked_pickles_for_varying_base_opts(
            stack_key=('sim_opts', 'sim_seed'),
            stack_values=sim_seeds,
            base_param_names_and_values=varyparams.keys(),
            **default_stack_and_vary_kwargs))

    # Actually plot
    fig, axarr = plt.subplots(2,
                              2,
                              figsize=(12, 10),
                              sharex=False,
                              sharey=False)
    axorder = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # loop over halo mass bins
    for iM in range(len(halo_masses)):
        log10M_min, log10M_max = halo_masses[iM]
        halo_mass_string = halo_mass_strings[iM]
        # vary params key
        vpkey = ((('sim_opts', 'halo_mass_string',), halo_mass_string),)

        stacked_pickles = stacked_vp_pickles[vpkey]
        base_pickle_opts = base_vp_pickles[vpkey]['opts']

        # get some basic info
        boxsize = int(base_pickle_opts['sim_opts'].boxsize)
        redshift = 1.0 / base_pickle_opts['sim_opts'].sim_scale_factor - 1.0
        kmin_plot = 1e-2
        kmax_plot = np.min(np.array([
            1.0,
            base_pickle_opts['grid_opts'].kmax
            ]))

        # set which panels have legend
        show_legend_arr = [True, False, False, False]
        show_legend = show_legend_arr[iM]
        # ymax for each panel
        ymax_plot_arr = [9, 5, 1.8, 1.8]
        ymax_plot = ymax_plot_arr[iM]
        ytitle_arr = [0.825, 0.825, 0.825, 0.825]

        # set up axis
        ax = axarr[axorder[iM]]
        plt.sca(ax)

        # changed dicts so don't have the (R,) key any more, so this is just
        # base pickle.
        base_pickle_first_R = base_vp_pickles[vpkey]
        print("tmp:", base_pickle_first_R['opts'].keys())
        trf_specs = base_pickle_first_R['opts']['trf_specs']
        trf_specs_save_bestfit_fields = [
            t.save_bestfit_field for t in trf_specs
        ]
        trf_specs_targets = [t.target_field for t in trf_specs]
        print("Trf specs:", trf_specs)
        print("\ntrf_specs_save_bestfit_fields:\n",
              "\n".join(trf_specs_save_bestfit_fields))

        # attempt to extract number density:
        nbar_of_cat = OrderedDict()
        cat_infos = base_pickle_first_R['cat_infos']
        for catid, cat_info in cat_infos.items():
            if cat_info.has_key('simple'):
                if cat_info['simple'].has_key('nbar_nonuni_cat'):
                    nbar_of_cat[catid] = cat_info['simple'][
                        'nbar_nonuni_cat']
                elif cat_info['simple'].has_key('nbar'):
                    nbar_of_cat[catid] = cat_info['simple']['nbar']
        print("nbar_of_cat:", nbar_of_cat)

        print('Pkmeas keys:\n', stacked_pickles['Pkmeas'].keys())
        tmp_key = stacked_pickles['Pkmeas'].keys()[0]
        Nsims = stacked_pickles['Pkmeas'][tmp_key]['k'].shape[0]
        print("Nsims:", Nsims)
        assert Nsims == len(sim_seeds)

        N_ortho_iter = base_pickle_opts['trf_fcn_opts'].N_ortho_iter

        for tfcounter, tf in enumerate(trf_specs[:1]):
            tmp_key = stacked_pickles['Pkmeas'].keys()[0]
            kvec = np.mean(
                stacked_pickles['Pkmeas'][tmp_key]['k'],
                axis=0)

        # CIC window (eq. 21 from jing et al 
        # https://arxiv.org/pdf/astro-ph/0409240.pdf)
        Delta_x = base_pickle_opts['sim_opts'].boxsize / float(
            base_pickle_opts['grid_opts'].Ngrid)
        k_nyq = np.pi / Delta_x
        nbar_kvec = np.zeros(kvec.shape[0] + 1)
        nbar_kvec[0] = kmin_plot
        nbar_kvec[1:] = kvec
        Pk_CIC = 1.0 - 2. / 3. * (np.sin(np.pi * nbar_kvec /
                                         (2.0 * k_nyq)))**2
        eval_Pk_CIC_at_kvec = lambda mykvec: 1.0 - 2. / 3. * (np.sin(
            np.pi * mykvec / (2.0 * k_nyq)))**2


        # try to plot 1/nbar_h
        plotted_shotnoise_ids = []
        # plot nbar of delta_h
        if nbar_of_cat.has_key('delta_h'):
            my_one_over_nbar = 1.0 / nbar_of_cat['delta_h']
            label = 'Poisson prediction'
            ax.plot(nbar_kvec,
                    my_one_over_nbar * Pk_CIC / my_one_over_nbar,
                    color='k',
                    ls='-',
                    lw=3,
                    zorder=0,
                    alpha=0.9,
                    label=label)
            plotted_shotnoise_ids.append('delta_h')
        else:
            raise Exception("Could not figure out nbar")

        ymax = None

        if ymax_plot is not None:
            ax.set_ylim((0, ymax_plot))

        color_counter = 0
        for tfcounter, tf in enumerate(trf_specs):
            # skip cubic bias for first few mass bins b/c makes no difference
            if tf.save_bestfit_field in [
                    'hat_delta_h_from_1_Tdeltalin2G23_SHIFTEDBY_PsiZ',
                    'hat_delta_h_from_Tdeltalin2G23_SHIFTEDBY_PsiZ'
            ]:
                if iM < 2:
                    continue
            # plot
            if True:
                tmp_key = stacked_pickles['Pkmeas'].keys()[0]
                kvec = np.mean(
                    stacked_pickles['Pkmeas'][tmp_key]['k'],
                    axis=0)
                Pks = stacked_pickles['Pkmeas']
                id1 = tf.save_bestfit_field
                id2 = tf.target_field
                label = tf.save_bestfit_field

                #col = constants.oranges4[-1-color_counter]
                col = constants.oranges[-2 - color_counter]
                # ls = linestyles[tfcounter]
                ls = ''
                plt_kwargs = {'color': col, 'ls': ls, 'label': label}
                color_counter += 1

                # enforce ls
                linestyles = [
                    '-',
                    (0, [12, 6]),
                    (10, [14, 4, 2, 4]),
                    (0, [3, 3]),
                    (0, [1, 1]),
                    (0, [8, 4]),
                ]
                ls = linestyles[0]
                plt_kwargs['ls'] = ls

                if True:
                    # plot power of (target-best_fit_field) computed on grid
                    residual_key = '[%s]_MINUS_[%s]' % (id1, id2)
                    if Pks.has_key((residual_key, residual_key)):
                        ymat = Pks[(residual_key,
                                    residual_key)]['P'] / my_one_over_nbar
                        if True:
                            # plot line
                            if tfcounter == 0:
                                label = 'redidual'
                            else:
                                label = '_nolabel_'
                            ax.semilogx(kvec,
                                        np.mean(ymat, axis=0),
                                        ls=plt_kwargs['ls'],
                                        color='k',
                                        lw=1,
                                        alpha=0.8,
                                        label=label)
                        if False:
                            # plot errorbar
                            plt_kwargs['marker'] = '.'
                            ax.errorbar(kvec,
                                        np.mean(ymat, axis=0),
                                        yerr=np.std(ymat, axis=0) /
                                        np.sqrt(float(Nsims - 1)),
                                        **plt_kwargs)
                        if True:
                            # plot errors using fill_between
                            yerr = np.std(ymat, axis=0) / np.sqrt(
                                float(Nsims - 1))
                            if show_legend:
                                mylabel = label
                            else:
                                # can annotate lines later
                                mylabel = '_nolabel_'
                            # enforce minimal line width
                            tmp_height = get_ax_size(fig, ax)[1]
                            if ymax_plot is not None:
                                myones = np.ones(
                                    yerr.shape
                                ) * ymax_plot / tmp_height * 2
                            else:
                                myones = np.ones(yerr.shape) * 0.02
                            myerr = np.maximum(yerr, myones)
                            ax.fill_between(
                                kvec,
                                (np.mean(ymat, axis=0) - myerr),
                                (np.mean(ymat, axis=0) + myerr),
                                facecolor=col,
                                lw=0,
                                alpha=0.9,  #alpha=0.6,
                                label=mylabel)

                if ymax is None:
                    tmpmean = np.mean(ymat, axis=0)
                    ymax2 = np.max(tmpmean[~np.isnan(tmpmean)])
                    if ymax2 > ymax:
                        ymax = ymax2

        if ymax_plot is not None:
            ymax = ymax_plot

        # plot cosmetics
        ax.set_xscale('log')
        ax.set_yscale('linear')
        if axorder[iM][0] == 1:
            ax.set_xlabel('$k\;[h\\mathrm{Mpc}^{-1}]$',
                          fontsize=constants.xylabelfs)
        if iM in [0, 1]:
            plt.xticks([0.01, 0.1, 1], [])
        elif iM == 2:
            plt.xticks([0.01, 0.1], ['0.01', '0.1'])
        elif iM == 3:
            plt.xticks([0.01, 0.1, 1], ['0.01', '0.1', '1'])
        if False:
            ax.set_ylabel('$($Model error$)^2$/Poisson prediction',
                          fontsize=constants.xylabelfs)
        # make space for ylabel
        if iM == 2:
            ax.set_ylabel('tmp',
                          color='w',
                          fontsize=constants.xylabelfs,
                          labelpad=20)
        ax.set_xlim((kmin_plot, kmax_plot))
        ymin, tmp_ymax = ax.get_ylim()

        if True:
            # also show curve with P_target,target of last trf_spec (e.g. to plot P_hh).
            # do this such that ylim is not changed
            if True:
                tmp_key = stacked_pickles['Pkmeas'].keys()[0]
                kvec = np.mean(
                    stacked_pickles['Pkmeas'][tmp_key]['k'],
                    axis=0)
                Pks = stacked_pickles['Pkmeas']
                ymat = Pks[(tf.target_field, tf.target_field)]['P']
                if tf.target_field == 'delta_h':
                    label = '_nolabel_'  # '$P_{hh}$'
                elif tf.target_field == 'delta_h_WEIGHT_M1':
                    label = '$P_{\\delta_{M_h}}$'
                else:
                    label = 'P_{%s}' % tf.target_field
                ax.plot(kvec,
                        np.mean(ymat, axis=0) / my_one_over_nbar,
                        lw=2,
                        color='grey',
                        ls=':',
                        label=label)

        ax.set_ylim((0, ymax))

        # yticks
        tmp_ymin, tmp_ymax = ax.get_ylim()
        if tmp_ymax is None:
            pass
        elif tmp_ymax < 2.:
            minorLocator = MultipleLocator(0.1)
            ax.yaxis.set_minor_locator(minorLocator)
            plt.yticks([0, 0.5, 1, 1.5], ['0', '0.5', '1', '1.5'])
            #majorLocator = MultipleLocator(1)
            #ax.yaxis.set_major_locator(majorLocator)
        elif tmp_ymax < 2.:
            minorLocator = MultipleLocator(0.25)
            ax.yaxis.set_minor_locator(minorLocator)
            majorLocator = MultipleLocator(1.0)
            ax.yaxis.set_major_locator(majorLocator)
        elif tmp_ymax < 10.:
            minorLocator = MultipleLocator(0.5)
            ax.yaxis.set_minor_locator(minorLocator)
            majorLocator = MultipleLocator(2.0)
            ax.yaxis.set_major_locator(majorLocator)

        if axorder[iM][1] == 1:
            ax.yaxis.tick_right()
        else:
            ax.yaxis.tick_left()

        # title
        if show_title:
            make_title(show_title,
                       log10M_min,
                       log10M_max,
                       my_one_over_nbar,
                       redshift,
                       ytitle=ytitle_arr[iM])
        if iM == 0:
            plt.title('tmp', color='w',
                      fontsize=constants.xylabelfs - 4)  # just to pad space

        # legend
        if show_legend:
            ax.legend(loc='best',
                      ncol=2,
                      fontsize=constants.xylabelfs - 6,
                      frameon=False,
                      handlelength=1.5)

    fig.text(0.06,
             0.5,
             '$P_\\mathrm{err}\\;/$ Poisson prediction',
             fontsize=constants.xylabelfs,
             ha='center',
             va='center',
             rotation='vertical')
    fig.text(0.5,
             0.96,
             'FOF halos at $\\,z=$%.1f, no mass weighting' % redshift,
             fontsize=constants.xylabelfs - 4,
             ha='center',
             va='center')

    plt.tight_layout(w_pad=-0.5, h_pad=0)

    if save_plots:
        if not os.path.exists('pdf/'):
            os.makedirs('pdf/')
        plot_fname = 'pdf/Perr_over_poisson.pdf'
        plt.savefig(plot_fname)
        print("Made %s" % plot_fname)
    else:
        plt.show()


def make_title(show_title,
               log10M_min,
               log10M_max,
               my_one_over_nbar,
               redshift,
               ytitle=0.85,
               ax=None):
    title_str = None
    if not (log10M_min is None or log10M_max is None):
        if show_title:
            if my_one_over_nbar is not None:
                if np.isclose(float(log10M_max), 16.0):
                    title_str = '$%s\\geq $%s$\\,%s$, $\\,\\bar n=$%.1e' % (
                        constants.log10M_string2, log10M_min,
                        constants.hMsun_string, 1.0 / my_one_over_nbar)
                else:
                    title_str = '$%s=$%s$-$%s\n$\\bar n=$%.1e' % (
                        constants.log10M_string2, log10M_min, log10M_max,
                        1.0 / my_one_over_nbar)
    # write title as text field
    if ax is None:
        ax = plt.gca()
    bbox_props = dict(boxstyle='round',
                      facecolor='lightgrey',
                      alpha=0.5,
                      linewidth=0)
    mypatch = ax.text(0.95,
                      ytitle,
                      title_str,
                      fontsize=constants.xylabelfs - 4,
                      horizontalalignment='right',
                      transform=ax.transAxes,
                      color='black',
                      bbox=bbox_props)
    #mypatch.get_frame().set_linewidth(1)


def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


if __name__ == '__main__':
    sys.exit(main())
