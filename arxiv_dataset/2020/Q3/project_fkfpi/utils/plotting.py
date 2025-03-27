#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import os, copy
import gvar as gv
# import chipt lib for fit functions
import chipt


class ExtrapolationPlots:

    def __init__(self, model, model_list, fitEnv, fit_result, switches):
        if not os.path.exists('figures'):
            os.makedirs('figures')
        # Figure formatting
        self.fig_width = 6.75 # in inches, 2x as wide as APS column
        self.gr        = 1.618034333 # golden ratio
        self.fig_size  = (self.fig_width, self.fig_width / self.gr)
        self.fig_size2 = (self.fig_width, self.fig_width * 1.49)
        self.plt_axes  = [0.14,0.14,0.858,0.858]
        self.fs_text   = 20 # font size of text
        self.fs_leg    = 16 # legend font size
        self.mrk_size  = '5' # marker size
        self.tick_size = 20 # tick size
        self.lw        = 1 # line width

        self.colors = {'a15':'#ec5d57', 'a12':'#70bf41', 'a09':'#51a7f9', 'a06':'#00FFFF'}
        self.shapes = {'m400':'h', 'm350':'p', 'm310':'s', 'm220':'^', 'm130':'o', 'm135':'*'}
        self.labels = {
            'a15m400':'', 'a15m350':'', 'a15m310':'a15', 'a15m220':'','a15m135XL':'',
            'a12m400':'', 'a12m350':'', 'a12m310':'a12', 'a12m220':'', 'a12m130':'',
            'a12m220L':'', 'a12m220S':'',
            'a09m400':'', 'a09m350':'', 'a09m310':'a09', 'a09m220':'', 'a09m135':'',
            'a06m310L':'a06',
            }
        self.dx_cont = {
            'a15m400'  :0.0050, 'a12m400' :0.0050, 'a09m400':0.0050,
            'a15m350'  :0.0025, 'a12m350' :0.0025, 'a09m350':0.0025,
            'a15m310'  :0.,     'a12m310' :0.,     'a09m310':0.,     'a06m310L':0.,
            'a15m220'  :-0.0025,'a12m220' :-0.0025,'a09m220':-0.0025,
            'a15m135XL':-0.0050,'a12m130' :-0.0050,'a09m135':-0.0050,
            'a12m220L' :-0.0037,'a12m220S':-0.0012,
        }

        self.model      = model
        self.FF         = model.split('_')[-1]
        self.fv         = 'FV' in model
        self.model_list = model_list
        self.fitEnv     = fitEnv
        self.fit_result = fit_result
        self.switches   = switches

        # create fit functions for original and shifted points
        self.og_fit     = chipt.FitModel(self.model_list, _fv=self.fv, _FF=self.FF)
        self.shift_list = list(model_list)
        if 'ma' in model_list[0]:
            self.shift_list[0] = self.shift_list[0].replace('ma','xpt')
        self.shift_fit  = chipt.FitModel(self.shift_list, _fv=False, _FF=self.FF)


    def plot_vs_eps_asq(self,shift_points):
        self.shift_xp = copy.deepcopy(shift_points)
        for k in self.fit_result.p:
            if isinstance(k,str):
                self.shift_xp['p'][k] = self.fit_result.p[k]
        y_plot = []
        x_plot = []
        a_range = np.sqrt(np.arange(0, .16**2, .16**2 / 50))
        for a_fm in a_range:
            self.shift_xp['p']['aw0'] = a_fm / self.shift_xp['p']['w0']
            x_plot.append((self.shift_xp['p']['aw0'] / 2)**2)
            y_a = self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p'])
            if self.switches['milc_compare']:
                y_a += self.fit_result.phys['dF_iso_xpt2'].mean
            y_plot.append(y_a)
        x  = np.array([k.mean for k in x_plot])
        y  = np.array([k.mean for k in y_plot])
        dy = np.array([k.sdev for k in y_plot])

        if self.switches['milc_compare']:
            figsize = self.fig_size2
        else:
            figsize = self.fig_size

        self.fig_cont = plt.figure('FKFpi_vs_ea_'+self.model,figsize=figsize)
        if self.switches['milc_compare']:
            self.ax_cont  = plt.axes([0.14,0.065,0.858,0.933])
        else:
            self.ax_cont  = plt.axes(self.plt_axes)
        self.ax_cont.fill_between(x, y-dy, y+dy, color='#b36ae2', alpha=0.4)

        self.plot_data(p_type='ea')
        handles, labels = self.ax_cont.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        self.ax_cont.legend(handles, labels, ncol=4, fontsize=self.fs_leg)

        self.ax_cont.set_xlabel(r'$\epsilon_a^2 = a^2 / (2 w_0)^2$',fontsize=self.fs_text)
        self.ax_cont.set_ylabel(r'$F_K / F_\pi$',fontsize=self.fs_text)
        if self.switches['milc_compare']:
            self.ax_cont.axhspan(1.1932-0.0019,1.1932+0.0019, color='k', alpha=.3)
            self.ax_cont.text(0.11, 1.1960, r'FLAG[2019]: $N_F=2+1+1$', \
                horizontalalignment='center', fontsize=self.fs_text)
            self.ax_cont.set_ylim(1.149, 1.228)
            self.ax_cont.set_ylabel(r'$F_{K^+} / F_{\pi^+}$',fontsize=self.fs_text)
            self.ax_cont.set_xlabel(r'$\epsilon_a^2 = a^2 / (2 w_0)^2$',fontsize=24)
            self.ax_cont.tick_params(labelsize=self.tick_size, direction='in')
        else:
            self.ax_cont.set_ylim(1.135, 1.218)
            self.ax_cont.text(0.0175, 1.145, r'%s' %(self.model.replace('_','\_')),\
                horizontalalignment='left', verticalalignment='center', \
                fontsize=self.fs_text, bbox={'facecolor':'None','boxstyle':'round'})
        self.ax_cont.set_xlim(0,.21)


        if self.switches['save_figs']:
            plt.savefig('figures/'+'FKFpi_vs_ea_'+self.model+'.pdf',transparent=True)

    def plot_vs_eps_pi(self,shift_points):
        self.shift_xp = copy.deepcopy(shift_points)
        eps_pisq_phys = (gv.gvar(self.shift_xp['p']['mpi'] / self.shift_xp['p']['Lchi_'+self.FF]))**2
        for k in self.fit_result.p:
            if isinstance(k,str):
                self.shift_xp['p'][k] = self.fit_result.p[k]
        y_plot = dict()
        # fit for each a and continuum
        y_plot['a15'] = []
        y_plot['a12'] = []
        y_plot['a09'] = []
        y_plot['a06'] = []
        y_plot['a00'] = []
        # xpt convergence - these are summed to a given order
        y_conv = dict()
        y_conv['NLO']   = []
        y_conv['NNLO']  = []
        y_conv['NNNLO'] = []
        nlo_lst   = [t for t in self.model_list if 'nlo' in t and not any(n in t for n in ['nnlo','nnnlo'])]
        nnlo_lst  = [t for t in self.model_list if 'nnlo' in t and 'nnnlo' not in t]
        nnnlo_lst = [t for t in self.model_list if 'nnnlo' in t]
        nlo_fit   = chipt.FitModel(nlo_lst, _fv=False, _FF=self.FF)
        nnlo_fit  = chipt.FitModel(nlo_lst+nnlo_lst, _fv=False, _FF=self.FF)
        nnnlo_fit = chipt.FitModel(nlo_lst+nnlo_lst+nnnlo_lst, _fv=False, _FF=self.FF)
        x_plot = []
        mpi_range = np.sqrt(np.arange(100, 411**2, 411**2/200))
        for a_mpi in mpi_range:
            x_plot.append(a_mpi**2 / (self.shift_xp['p']['Lchi_'+self.FF])**2)
            self.shift_xp['p']['mpi'] = a_mpi
            self.shift_xp['p']['aw0'] = 0
            y_plot['a00'].append(self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['NLO'].append(self.fitEnv._fit_function(nlo_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['NNLO'].append(self.fitEnv._fit_function(nnlo_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['NNNLO'].append(self.fitEnv._fit_function(nnnlo_fit, self.shift_xp['x'], self.shift_xp['p']))
            for aa in ['a15','a12','a09','a06']:
                if aa == 'a06':
                    self.shift_xp['p']['aw0'] = self.fitEnv.p[(aa+'m310L','aw0')]
                else:
                    self.shift_xp['p']['aw0'] = self.fitEnv.p[(aa+'m310','aw0')]
                y_plot[aa].append(self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p']))
        x = np.array([k.mean for k in x_plot])
        y  = dict()
        dy = dict()
        y['a00']  = np.array([k.mean for k in y_plot['a00']])
        dy['a00'] = np.array([k.sdev for k in y_plot['a00']])
        for aa in ['a15','a12','a09','a06']:
            y[aa]  = np.array([k.mean for k in y_plot[aa]])
            dy[aa] = np.array([k.sdev for k in y_plot[aa]])

        self.fig_x = plt.figure('FKFpi_vs_epi_'+self.model, figsize=self.fig_size)
        self.ax_x  = plt.axes(self.plt_axes)
        self.ax_x.fill_between(x, y['a00']-dy['a00'], y['a00']+dy['a00'], color='#b36ae2',alpha=0.4)
        for aa in ['a15','a12','a09','a06']:
            self.ax_x.plot(x, y[aa], color=self.colors[aa])

        # plot physical eps_pi**2
        self.ax_x.axvline(eps_pisq_phys.mean,linestyle='--',color='#a6aaa9')
        self.ax_x.axvspan(eps_pisq_phys.mean -eps_pisq_phys.sdev, eps_pisq_phys.mean +eps_pisq_phys.sdev,
            alpha=0.4, color='#a6aaa9')
        # plot data
        self.plot_data(p_type='epi')
        handles, labels = self.ax_x.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        self.ax_x.legend(handles, labels, ncol=1, fontsize=self.fs_leg)

        # labels
        eps_FF = {
            'PP':r'$\epsilon_\pi^2 = (m_\pi / 4\pi F_\pi)^2$',
            'PK':r'$\epsilon_\pi^2 = (m_\pi / 4\pi)^2 /(F_\pi F_K)$',
            'KK':r'$\epsilon_\pi^2 = (m_\pi / 4\pi F_K)^2$'
        }
        xlim_FF = {'PP':.094, 'PK':.088, 'KK':.084}
        self.ax_x.set_xlabel(eps_FF[self.FF],fontsize=self.fs_text)
        self.ax_x.set_xlim(0,xlim_FF[self.FF])
        self.ax_x.set_ylabel(r'$F_K / F_\pi$',fontsize=self.fs_text)
        self.ax_x.set_ylim(1.06, 1.218)
        self.ax_x.text(0.0175, 1.075, r'%s' %(self.model.replace('_','\_')),\
            horizontalalignment='left', verticalalignment='center', \
            fontsize=self.fs_text, bbox={'facecolor':'None','boxstyle':'round'})

        if self.switches['save_figs']:
            plt.savefig('figures/'+'FKFpi_vs_epi_'+self.model+'.pdf',transparent=True)

        # Convergence plot
        order_list = ['NLO']
        if 'nnnlo' in self.model:
            order_list = order_list + ['NNLO','NNNLO']
        elif 'nnlo' in self.model:
            order_list = order_list + ['NNLO']
        self.fig_conv = plt.figure('FKFpi_vs_epi_convergence_'+self.model, figsize=self.fig_size)
        self.ax_conv  = plt.axes(self.plt_axes)
        labels = {'NLO':'NLO', 'NNLO':r'N$^2$LO','NNNLO':r'N$^3$LO'}
        for order in order_list:
            mean = np.array([k.mean for k in y_conv[order]])
            sdev = np.array([k.sdev for k in y_conv[order]])
            self.ax_conv.fill_between(x, mean-sdev, mean+sdev, alpha=.4, label=labels[order])
        self.ax_conv.set_xlabel(eps_FF[self.FF],fontsize=self.fs_text)
        self.ax_conv.set_xlim(0,xlim_FF[self.FF])
        self.ax_conv.set_ylabel(r'$F_K / F_\pi$',fontsize=self.fs_text)
        self.ax_conv.set_ylim(1.06, 1.218)
        self.ax_conv.text(0.0175, 1.075, r'%s' %(self.model.replace('_','\_')),\
            horizontalalignment='left', verticalalignment='center', \
            fontsize=self.fs_text, bbox={'facecolor':'None','boxstyle':'round'})
        self.ax_conv.axvline(eps_pisq_phys.mean,linestyle='--',color='#a6aaa9')
        self.ax_conv.axvspan(eps_pisq_phys.mean -eps_pisq_phys.sdev, eps_pisq_phys.mean +eps_pisq_phys.sdev,
            alpha=0.4, color='#a6aaa9')
        self.ax_conv.legend(ncol=3, fontsize=self.fs_leg)
        if self.switches['save_figs']:
            plt.savefig('figures/'+'FKFpi_vs_epi_convergence_'+self.model+'.pdf',transparent=True)


    def plot_data(self, p_type, offset=False, raw=False):
        y_shift = self.shift_data(p_type=p_type)
        for a_ens in self.switches['ensembles']:
            if a_ens in self.switches['ensembles_fit']:
                c = self.colors[a_ens.split('m')[0]]
                alpha = 1
            else:
                c = 'k'
                alpha = 0.4
            s = self.shapes['m'+a_ens.split('m')[1][0:3]]
            if p_type == 'ea':
                x  = (self.fitEnv.p[(a_ens, 'aw0')] / 2)**2
                dx = self.dx_cont[a_ens]
            elif p_type == 'epi':
                x  = (self.fitEnv.p[(a_ens, 'mpi')] / self.fitEnv.p[(a_ens, 'Lchi_'+self.FF)])**2
                dx = 0
            label = self.labels[a_ens]
            if p_type == 'ea':
                if a_ens in self.switches['ensembles_fit']:
                    mfc = c
                else:
                    mfc = 'None'
            elif p_type == 'epi':
                mfc = 'None'
            y = self.fitEnv.y[a_ens] + y_shift[a_ens]
            if self.switches['milc_compare']:
                y += self.fit_result.phys['dF_iso_xpt2'].mean
            if p_type == 'ea':
                self.ax_cont.errorbar(x=x.mean+dx, y=y.mean,xerr=x.sdev, yerr=y.sdev,
                    marker=s, color=c, mfc=mfc, alpha=alpha, linestyle='None', label=label)
            elif p_type == 'epi':
                self.ax_x.errorbar(x=x.mean+dx, y=y.mean,xerr=x.sdev, yerr=y.sdev,
                    marker=s, color=c, mfc=mfc, alpha=alpha, linestyle='None', label=label)

    def shift_data(self, p_type):
        y_shift = dict()
        if self.switches['debug_shift']:
            print('%9s   %11s   y_shift[ens]' %('ensemble', 'y[ens]'))
            print('---------------------------------------------------------------')
        for a_ens in self.switches['ensembles']:
            og_priors = dict()
            for k, v in self.fitEnv.p.items():
                if type(k) == tuple and k[0] == a_ens:
                    #print(a_ens,k,v)
                    if k in self.fit_result.p:
                    #if a_ens in self.switches['ensembles_fit']:
                        og_priors[k[1]] = self.fit_result.p[k] # the x-params which are priors
                    else:
                        # if it is from an excluded ensemble - get from original data
                        og_priors[k[1]] = v
            for k in self.fit_result.p:
                if isinstance(k,str):# grab the LECs from the fit results
                    og_priors[k] = self.fit_result.p[k]    # the LECs of the fit
            self.shift_xp['p']['aw0'] = self.fitEnv.p[(a_ens,'aw0')]
            if p_type == 'epi':
                self.shift_xp['p']['aw0'] = self.fitEnv.p[(a_ens,'aw0')]
                self.shift_xp['p']['mpi'] = self.fitEnv.p[(a_ens,'mpi')] / self.fitEnv.p[(a_ens, 'Lchi_'+self.FF)]
                self.shift_xp['p']['mk']  = self.shift_xp['p']['mk'] / self.shift_xp['p']['Lchi_'+self.FF]
                self.shift_xp['p']['Lchi_'+self.FF] = 1
                self.shift_xp['x']['alphaS'] = self.fitEnv.x[a_ens]['alphaS']
            og_y    = self.fitEnv._fit_function(self.og_fit,    self.fitEnv.x[a_ens], og_priors)
            shift_y = self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'],   self.shift_xp['p'])
            y_shift[a_ens] = shift_y - og_y
            if self.switches['debug_shift']:
                print('%9s   %11s   %s' %(a_ens, og_y, shift_y))
        return y_shift

    def plot_vs_ml(self):
        fv_dict = dict()
        fv_dict['p'] = dict()
        fv_dict['p']['mpi']     = self.fit_result.p['a12m220L', 'mpi']
        fv_dict['p']['mk']      = self.fit_result.p['a12m220L', 'mk']
        fv_dict['p']['Lchi_PP'] = self.fit_result.p['a12m220L', 'Lchi_PP']
        fv_dict['p']['aw0']     = self.fit_result.p['a12m220L', 'aw0']
        fv_dict['x'] = dict()
        fv_dict['x'] = dict(self.fit_result.x['a12m220L'])
        for k in self.fit_result.p:
            if isinstance(k, str):
                fv_dict['p'][k] = self.fit_result.p[k]
        mpi = fv_dict['p']['mpi'].mean
        mk  = fv_dict['p']['mk'].mean
        me  = np.sqrt(4./3 * mk**2 - 1./3*mpi**2)
        fv_pred = []
        x   = []
        fv_fit_func = chipt.FitModel(self.model_list, _fv=self.fv, _FF=self.FF)
        for mL in np.arange(3.,10.1,.1):
            x.append(np.exp(-mL) / (mL)**1.5)
            fv_dict['x']['mpiL'] = mL
            fv_dict['x'][k] = mL * mk/mpi
            fv_dict['x'][k] = mL * me/mpi
            fv_pred.append(self.fitEnv._fit_function(fv_fit_func, fv_dict['x'], fv_dict['p']))
        x = np.array(x)
        y  = np.array([k.mean for k in fv_pred])
        dy = np.array([k.sdev for k in fv_pred])

        self.fig_Fv = plt.figure('FKFpi_vs_mL_'+self.model, figsize=self.fig_size)
        self.ax_fv  = plt.axes(self.plt_axes)
        self.ax_fv.fill_between(x, y-dy, y+dy, color=self.colors['a12'], alpha=0.4)
        markers = ['s','o','*']
        mL_ens = dict()
        xL_ens = dict()
        fL_ens = dict()
        for i_e,ens in enumerate(['a12m220L', 'a12m220', 'a12m220S']):
            if ens in self.switches['ensembles_fit']:
                c = color=self.colors['a12']
            else:
                c = 'k'
            mL_ens[ens] = self.fit_result.x[ens]['mpiL']
            y_data = self.fit_result.y[ens]
            self.ax_fv.errorbar(np.exp(-mL_ens[ens])/mL_ens[ens]**1.5, y_data.mean, yerr=y_data.sdev, \
                marker=markers[i_e], color=c, linestyle='None',label=r'$m_\pi L=%.2f$' %(mL_ens[ens]))
            # collect info for making text in band
            fv_dict['x']['mpiL'] = mL_ens[ens]
            fv_dict['x'][k] = mL_ens[ens] * mk/mpi
            fv_dict['x'][k] = mL_ens[ens] * me/mpi
            xL_ens[ens] = np.exp(-mL_ens[ens])/mL_ens[ens]**1.5
            fL_ens[ens] = self.fitEnv._fit_function(fv_fit_func, fv_dict['x'], fv_dict['p']).mean

        self.ax_fv.set_xlabel(r'$e^{-m_\pi L} / (m_\pi L)^{3/2}$',fontsize=self.fs_text)
        self.ax_fv.set_ylabel(r'$F_K / F_\pi$',fontsize=self.fs_text)
        self.ax_fv.legend(ncol=3, fontsize=self.fs_leg, columnspacing=0.5)
        self.ax_fv.vlines(0,1.12,1.15, color='k', lw=0.4)
        self.ax_fv.set_ylim(1.124, 1.144)
        self.ax_fv.set_xlim(-0.000,.0075)

        # Do a little of trig to get text to line up in band
        x_text  = (xL_ens['a12m220S'] + xL_ens['a12m220']) / 2
        def ml_x(mL):
            return np.exp(-mL)/mL**1.5
        mL_text = minimize_scalar(lambda x: (ml_x(x) -x_text)**2 ,bounds=(3,7), method='bounded').x
        fv_dict['x']['mpiL'] = mL_text
        fv_dict['x'][k] = mL_text * mk/mpi
        fv_dict['x'][k] = mL_text * me/mpi
        y_text = self.fitEnv._fit_function(fv_fit_func, fv_dict['x'], fv_dict['p']).mean
        # scale dy and dx by the limits of the plot to get angle right
        dx = (xL_ens['a12m220S'] - xL_ens['a12m220']) / (self.ax_fv.get_xlim()[1]-self.ax_fv.get_xlim()[0])
        dy = (fL_ens['a12m220S'] - fL_ens['a12m220']) / (self.ax_fv.get_ylim()[1]-self.ax_fv.get_ylim()[0])
        angle = 180/np.pi * np.arctan(dy / dx / self.gr) # remember the golden ratio scaling
        self.ax_fv.text(x_text, y_text - 0.0003, \
            r'a12m220: $\delta_{\rm FV}^{{\rm NLO}\ \chi{\rm PT}}(\epsilon_\pi^2, m_\pi L)$', \
            horizontalalignment='center', verticalalignment='center', \
            rotation=angle, fontsize=self.fs_text-1)

        if self.switches['save_figs']:
            plt.savefig('figures/'+'FKFpi_vs_mL_'+self.model+'.pdf',transparent=True)
