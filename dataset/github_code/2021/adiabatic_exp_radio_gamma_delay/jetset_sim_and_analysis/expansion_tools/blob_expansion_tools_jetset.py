from itertools import cycle
import numpy as np
import pylab as plt
from jetset.jet_timedep import JetTimeEvol
import pickle
import warnings
import jetset
warnings.filterwarnings('ignore')


def plot_2lcs(lc_1, lc_2):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    tab_list = [lc_1, lc_2]

    y_l = [None]*2
    y_l_err = [None]*2
    t_l = [None]*2
    y_label = [None]*2
    labels = [None]*2

    fig = plt.figure()
    fig_h = 6
    fig_w = 15
    gs_w = 7
    d_w = 0

    fig.set_size_inches(fig_w, fig_h)

    gs = fig.add_gridspec(2, gs_w)
    ax1 = fig.add_subplot(gs[0, 0:gs_w-d_w])
    ax2 = fig.add_subplot(gs[1, 0:gs_w-d_w])

    plt.tight_layout()

    for ID, f in enumerate(tab_list):

        labels[ID] = tab_list[ID].meta['name']

        t_l[ID] = tab_list[ID]['time'].to('d').value
        y_l[ID] = tab_list[ID]['flux']
        if 'e_flux' in tab_list[ID].colnames:
            y_l_err[ID] = tab_list[ID]['e_flux']
        else:
            y_l_err[ID] = np.zeros(len(tab_list[ID]))

        y_label[ID] = tab_list[ID]['flux'].unit

    ax1.errorbar(x=t_l[0], y=y_l[0], yerr=0, label=labels[0],
                 marker='o', ms=1, lw=0.3, linestyle='dashed', c='red')
    ax2.errorbar(x=t_l[1], y=y_l[1], yerr=0, label=labels[1],
                 marker='o', ms=1, lw=0.3, linestyle='dashed', c='blue')

    ax1.legend()
    ax1.set_ylabel(y_label[0])

    ax2.set_ylabel(y_label[1])
    ax2.set_xlabel('T days')
    ax2.legend()


def build_Temp_EV(duration,
                  T_SIZE, 
                  flare_duration, 
                  delta_T, 
                  Delta_R_acc_ratio=None, 
                  B_acc_ratio=4.0, 
                  q_inj=None, 
                  jet=None, 
                  show=False, 
                  NUM_SET=200, 
                  L_inj=1E39, 
                  T_esc_rad=None, 
                  T_esc_acc=None, 
                  t_D0=1E5, 
                  t_A0=1E4, 
                  E_acc_max=None, 
                  Acc_Index=1, 
                  Diff_Index=2,    
                  no_cooling=False,
                  synch_cooling_off=False,
                  m_B=1,):

    temp_ev_acc = JetTimeEvol(jet_rad=jet, Q_inj=q_inj, inplace=True)
    temp_ev_acc.rad_region.jet.nu_min = 1E8
    temp_ev_acc.acc_region.jet.nu_min = 1E8
    T_SIZE = np.int(T_SIZE)

    if Delta_R_acc_ratio is not None:
        temp_ev_acc.parameters.Delta_R_acc.val = temp_ev_acc.parameters.R_rad_start.val*Delta_R_acc_ratio

    if T_esc_acc is None:
        T_esc_acc = (t_A0*0.5)/(temp_ev_acc.parameters.Delta_R_acc.val/3E10)

    if T_esc_rad is None:
        T_esc_rad = 2.0

    temp_ev_acc.parameters.duration.val = duration
    temp_ev_acc.parameters.TStart_Acc.val = 0+delta_T
    temp_ev_acc.parameters.TStop_Acc.val = flare_duration+delta_T
    temp_ev_acc.parameters.TStart_Inj.val = 0+delta_T
    temp_ev_acc.parameters.TStop_Inj.val = flare_duration+delta_T
    temp_ev_acc.parameters.T_esc_acc.val = T_esc_acc
    temp_ev_acc.parameters.T_esc_rad.val = T_esc_rad
    temp_ev_acc.parameters.t_D0.val = t_D0
    temp_ev_acc.parameters.t_A0.val = t_A0
    temp_ev_acc.parameters.Esc_Index_acc.val = Diff_Index-2
    temp_ev_acc.parameters.Esc_Index_rad.val = 0
    temp_ev_acc.parameters.Acc_Index.val = Acc_Index
    temp_ev_acc.parameters.Diff_Index.val = Diff_Index
    temp_ev_acc.parameters.t_size.val = T_SIZE
    temp_ev_acc.parameters.num_samples.val = NUM_SET
    temp_ev_acc.parameters.m_B.val=m_B

    if E_acc_max is not None:
        temp_ev_acc.parameters.E_acc_max.val = E_acc_max

        temp_ev_acc.parameters.L_inj.val = L_inj

    temp_ev_acc.parameters.gmin_grid.val = 1.0
    temp_ev_acc.parameters.gmax_grid.val = 1E8
    temp_ev_acc.parameters.gamma_grid_size.val = 1500

    temp_ev_acc.parameters.B_acc.val = temp_ev_acc.rad_region.jet.parameters.B.val*B_acc_ratio

    if no_cooling is True:
        temp_ev_acc.Sync_cooling = 'off'
        temp_ev_acc.IC_cooling = 'off'
        temp_ev_acc.Adiabatic_cooling = 'off'

    if synch_cooling_off is True:
        temp_ev_acc.Sync_cooling = 'off'
        temp_ev_acc.IC_cooling = 'off'


    temp_ev_acc.init_TempEv()
    if show is True:
        temp_ev_acc.show_model()
    return temp_ev_acc


def do_analysistemp_ev(temp_ev,
                       sed_data,
                       only_injection,
                       do_injection,
                       fit_model=None,
                       run=True,
                       cache_SEDs_acc=True,
                       cache_SEDs_rad=True,
                       eval_cross_time=False,
                       delta_t_out=1E4,
                       rest_frame='obs',
                       plot_emitters=False,
                       plot_seds=False,
                       plot_lcs=False,
                       plot_fit_model=False,
                       plot_fit_distr=False):

    temp_ev.init_TempEv()

    if run is True:
        temp_ev.run(only_injection=only_injection, do_injection=do_injection,
                    cache_SEDs_acc=cache_SEDs_acc, cache_SEDs_rad=cache_SEDs_rad)

    if plot_emitters is True:
        p = temp_ev.plot_tempev_emitters(
            region='rad', loglog=False, energy_unit='gamma', pow=0)
        p.ax.axvline(temp_ev.temp_ev.gamma_eq_t_A, ls='--')
        p.ax.axvline(temp_ev.temp_ev.gamma_eq_t_DA, ls='--')
        if plot_fit_distr is True and fit_model is not None:
            x = fit_model.jet_leptonic.emitters_distribution.gamma_e
            y = fit_model.jet_leptonic.emitters_distribution.n_gamma_e
            p.ax.plot(x, y, c='g', label='best fit model')
        #p.ax.plot(temp_ev_acc_flare.time_sampled_emitters.gamma,temp_ev_acc_flare.time_sampled_emitters.n_gamma_rad[55],c='orange',label='best fit model',ms=2)
        if jetset.__version__<'1.2.1':
            p.rescale(x_max=1E7, x_min=1, y_min=1E-18, y_max=100)
        else:
            p.setlim(x_max=1E7, x_min=1, y_min=1E-18, y_max=100)

        if cache_SEDs_acc is True:
            p = temp_ev.plot_tempev_emitters(
                region='acc', loglog=False, energy_unit='gamma', pow=0)
            p.ax.axvline(temp_ev.temp_ev.gamma_eq_t_A, ls='--')
            p.ax.axvline(temp_ev.temp_ev.gamma_eq_t_DA, ls='--')
            if plot_fit_distr is True and fit_model is not None:
                x = fit_model.jet_leptonic.emitters_distribution.gamma_e
                y = fit_model.jet_leptonic.emitters_distribution.n_gamma_e*50
                p.ax.plot(x, y, c='g', label='best fit model')
            if jetset.__version__<'1.2.1':
                p.rescale(x_max=1E7, x_min=1, y_min=1E-18, y_max=100)
            else:
                p.setlim(x_max=1E7, x_min=1, y_min=1E-18, y_max=100)

    if plot_seds is True:
        if cache_SEDs_rad is True:
            p = temp_ev.plot_tempev_model(
                region='rad', sed_data=sed_data, use_cached=True)
            if plot_fit_model is True and fit_model is not None:
                fit_model.eval()
                fit_model.jet_leptonic.plot_model(
                    plot_obj=p, sed_data=sed_data)
            if jetset.__version__<'1.2.1':
                p.rescale(y_min=-18, x_min=7)
            else:
                p.setlim(y_min=1E-18, x_min=1E7)

        if cache_SEDs_acc is True:
            p = temp_ev.plot_tempev_model(
                region='acc', sed_data=sed_data, use_cached=True)
            if plot_fit_model is True and fit_model is not None:
                fit_model.eval()
                fit_model.jet_leptonic.plot_model(
                    plot_obj=p, sed_data=sed_data)
            if jetset.__version__<'1.2.1':
                p.rescale(y_min=-18, x_min=7)
            else:
                p.setlim(y_min=1E-18, x_min=1E7)

    lcs_dict = {}
    if cache_SEDs_rad is True:
        lg = temp_ev.rad_region.make_lc(nu1=2.4E22, nu2=7.2E25, name='gamma', eval_cross_time=eval_cross_time,
                                        delta_t_out=delta_t_out, use_cached=True, frame=rest_frame)
        lcs_dict['lg_%s' % rest_frame] = lg
        for nu in np.linspace(5, 120, 24):
            lr = temp_ev.rad_region.make_lc(nu1=nu*1E9, name='radio_%dGHz' % nu, eval_cross_time=eval_cross_time,
                                            delta_t_out=delta_t_out, use_cached=True, frame=rest_frame,density=True)
            lcs_dict['lr_%s_%dGHz' % (rest_frame, nu)] = lr
        lopt = temp_ev.rad_region.make_lc(
            nu1=5E14, name='opt', eval_cross_time=eval_cross_time, delta_t_out=delta_t_out, use_cached=True, frame=rest_frame)
        lcs_dict['lopt_%s' % rest_frame] = lopt
        lmm = temp_ev.rad_region.make_lc(nu1=1E12, nu2=1E13, name='mm', eval_cross_time=eval_cross_time,
                                         delta_t_out=delta_t_out, use_cached=True, frame=rest_frame)
        lcs_dict['lmm_%s' % rest_frame] = lmm
        lx = temp_ev.rad_region.make_lc(nu1=1E17, nu2=1E18, name='X', eval_cross_time=eval_cross_time,
                                        delta_t_out=delta_t_out, use_cached=True, frame=rest_frame)
        lcs_dict['lx_%s' % rest_frame] = lx

    if plot_lcs is True and cache_SEDs_rad is True:
        plot_2lcs(lg, lcs_dict['lr_%s_%dGHz' % (rest_frame, 10)])
        plot_2lcs(lg, lcs_dict['lr_%s_%dGHz' % (rest_frame, 30)])
        plot_2lcs(lx, lg,)
        plot_2lcs(lg, lmm)

    return lcs_dict


def check_N(temp_ev_expansion, flare_duration, duration, t_ref, N, N_out):
    t1 = np.int(t_ref/duration*N)
    t1_R = temp_ev_expansion.rad_region.time_sampled_emitters.time_blob[t1]
    R1 = temp_ev_expansion._get_R_rad_sphere(t1_R)
    N1 = np.trapz(
        temp_ev_expansion.rad_region.time_sampled_emitters.n_gamma[t1], temp_ev_expansion.rad_region.time_sampled_emitters.gamma)*R1**3

    if t_ref <= 0:
        t_ref = 1.

    t_exp = temp_ev_expansion.parameters.t_jet_exp.val
    if t_exp <= 0:
        t_exp = 1.
    print(t_exp, t_ref, duration)
    for t in np.linspace(flare_duration, duration-1, N_out):
        t2 = np.int(t/duration*N)
        t2_R = temp_ev_expansion.rad_region.time_sampled_emitters.time_blob[t2]
        R2 = temp_ev_expansion._get_R_rad_sphere(t2_R)
        N2 = np.trapz(
            temp_ev_expansion.rad_region.time_sampled_emitters.n_gamma[t2], temp_ev_expansion.rad_region.time_sampled_emitters.gamma)*R2**3
        print('t1=%3.3d t2=%3.3d, t2_R=%3.3e, R2=%5.5e t/t_ref=%3.3f t/t_exp=%3.3f t/duration=%3.3f %5.5e' % (t1,
                                                                                                              t2,
                                                                                                              t2_R,
                                                                                                              R2,
                                                                                                              t/t_ref,
                                                                                                              t/t_exp,
                                                                                                              t2_R/duration,
                                                                                                              N2/N1))


def run_adiabatic_exp(sed_data,
                      fit_model,
                      root_dir,
                      temp_ev_acc_flare,
                      flare_duration,
                      duration,
                      T_esc_rad,
                      T_SIZE,
                      NUM_SET,
                      beta_exp,
                      t_exp,
                      rest_frame='blob',
                      expansion='on',
                      run=True,
                      average=0.2,
                      analysis_dir=None,
                      delta_t_out=1E4,
                      only_injection=False,
                      do_injection=False,
                      plot_fit_model=False,
                      plot_fit_distr=False,
                      eval_cross_time=False,
                      no_cooling=False,
                      synch_cooling_off=False,
                      cache_SEDs_acc=False,
                      cache_SEDs_rad=True,
                      m_B=1):

    temp_ev_acc_flare.set_time(time_slice=-1)

    temp_ev_expansion = build_Temp_EV(duration=duration,
                                      T_SIZE=T_SIZE,
                                      flare_duration=flare_duration,
                                      delta_T=0,
                                      q_inj=None,
                                      jet=temp_ev_acc_flare.rad_region.jet,
                                      show=False,
                                      NUM_SET=NUM_SET,
                                      T_esc_rad=T_esc_rad,
                                      L_inj=0,
                                      no_cooling=no_cooling,
                                      synch_cooling_off=synch_cooling_off,
                                      m_B=m_B)

    temp_ev_expansion.parameters.beta_exp_R.val = beta_exp
    temp_ev_expansion.parameters.t_jet_exp.val = t_exp

    temp_ev_expansion.parameters.R_rad_start.val = temp_ev_acc_flare.rad_region.jet.parameters.R.val

    temp_ev_expansion.region_expansion = expansion

    temp_ev_expansion.init_TempEv()
    if run is False:
        temp_ev_expansion = JetTimeEvol.load_model(
            '%s/temp_ev_expansion_beta_exp=%3.3f.pkl' % (root_dir, beta_exp))

    lcs_v_exp = do_analysistemp_ev(temp_ev_expansion,
                                   sed_data,
                                   run=run,
                                   fit_model=fit_model,
                                   only_injection=only_injection,
                                   do_injection=do_injection,
                                   plot_fit_model=plot_fit_model,
                                   plot_fit_distr=plot_fit_distr,
                                   eval_cross_time=eval_cross_time,
                                   delta_t_out=delta_t_out,
                                   rest_frame=rest_frame,
                                   cache_SEDs_acc=cache_SEDs_acc,
                                   cache_SEDs_rad=cache_SEDs_rad)

    lcs_v_exp['beta_exp'] = beta_exp
    if analysis_dir is None:
        analysis_dir = root_dir
    with open('%s/lc_%s_beta_exp=%3.3f.pkl' % (analysis_dir, rest_frame, beta_exp), 'wb') as f:
        pickle.dump(lcs_v_exp, f, pickle.HIGHEST_PROTOCOL)

    if run is True:
        if expansion == 'on':

            check_N(temp_ev_expansion, 0, duration, temp_ev_expansion.parameters.t_jet_exp.val,
                    temp_ev_expansion.parameters.num_samples.val, 10)
        temp_ev_expansion.save_model(
            '%s/temp_ev_expansion_beta_exp=%3.3f.pkl' % (root_dir, beta_exp))
