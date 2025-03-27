#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os, sys, shutil
import yaml

import gvar as gv


class BayesModelAvg:
    ''' Under Bayes Theorem, models which are used to fit the same dataset
        can easily be averaged together with the exp(logGBF) factor providing
        a weight, such that the relative weigth of any given model with respect
        to the most likely model (of those selected) is given by

        w_i = exp( logGBF_i - logGBF_max )

        The final uncertainty under such model averaging is given simply by a
        combination of the weighted variance from each fit, plus the variance
        over the models:

        mu  = sum_i w_i mu_i
        Var = sum_i w_i * var_i   + [ sum_j mu_j**2 * w_j - mu**2 ]

        This class performs this weighted average, and creates histograms of
        the final answer including splitting the historgram in various ways to
        understand the dominant contributions to the model average uncertainty
    '''
    def __init__(self, results):
        ''' results is a dictionary where the key is the model name and the
            value is an lsqfit object
        '''
        self.results = results
        # create a list of the keys so we gaurantee the same order
        self.r_list  = [a_res for a_res in self.results]
        self.weights = self.get_weights()

        # Figure formatting
        self.fig_width = 6.75 # in inches, 2x as wide as APS column
        self.gr        = 1.618034333 # golden ratio
        self.fig_size  = (self.fig_width, self.fig_width / self.gr)
        self.fig_size2 = (self.fig_width, self.fig_width * 1.58)
        self.plt_axes  = [0.14,0.14,0.858,0.858]
        self.fs_text   = 20 # font size of text
        self.fs_leg    = 16 # legend font size
        self.mrk_size  = '5' # marker size
        self.tick_size = 20 # tick size
        self.lw        = 1 # line width



    def get_weights(self):
        weights = []
        for a_res in self.r_list:
            weights.append(np.exp(self.results[a_res].logGBF))
        weights = np.array(weights)
        weights = weights / np.max(weights)
        weights = weights / weights.sum()
        return weights

    def print_weighted_models(self):
        i_weights = np.argsort(self.weights)[::-1]
        print(r"%37s & chi2/dof &   $Q$ &  logGBF& weight& $F_K/F_\pi$\\" %'model')
        print(r'\hline')
        for a_i, a_model in enumerate(np.array(self.r_list)[i_weights]):
            chi2   = self.results[a_model].chi2
            dof    = self.results[a_model].dof
            Q      = self.results[a_model].Q
            logGBF = self.results[a_model].logGBF
            w_i    = self.weights[i_weights][a_i]
            phys   = self.results[a_model].phys['FKFpi']
            print(r'%37s &  %.3f   &  %.3f&  %.3f&  %.3f&  %s\\' %(a_model.replace('_','\_'), chi2/dof, Q, logGBF, w_i, phys))

    def bayes_model_avg(self):
        self.pdf_x = np.arange(1.15,1.2301,.0001)
        avg = {k:0. for k in ['FKFpi', 'dF_iso_xpt', 'dF_iso_xpt2']}
        pdf = 0.
        cdf = 0.
        pdf_split = dict()
        pdf_split['PP'] = 0.
        pdf_split['PK'] = 0.
        pdf_split['KK'] = 0.
        pdf_split['PP_xpt'] = 0.
        pdf_split['PK_xpt'] = 0.
        pdf_split['KK_xpt'] = 0.
        pdf_split['ratio_PP'] = 0.
        pdf_split['no_ratio_PP'] = 0.
        pdf_split['ct_PP'] = 0.
        pdf_split['no_ct_PP'] = 0.
        results = []
        var_avg = dict()
        for i_r, a_res in enumerate(self.r_list):
            w_i = self.weights[i_r]
            a_i = self.results[a_res].phys
            results.append(a_i['FKFpi'])
            for k in avg:
                avg[k] += gv.gvar(w_i*a_i[k].mean, np.sqrt(w_i)*a_i[k].sdev)
            #avg.update({k:v += gv.gvar(w_i*a_i[k].mean, np.sqrt(w_i)*a_i[k].sdev) for k,v in avg.items()})
            uncertainties = uncertainty_breakdown(self.results[a_res],'FKFpi')
            for k in uncertainties:
                if k not in var_avg: var_avg[k] = 0.
                var_avg[k] += w_i * uncertainties[k]**2

            p = stats.norm.pdf(self.pdf_x, a_i['FKFpi'].mean, a_i['FKFpi'].sdev)
            pdf += w_i * p
            cdf += w_i * stats.norm.cdf(self.pdf_x, a_i['FKFpi'].mean, a_i['FKFpi'].sdev)
            FF = a_res.split('_')[-1]
            pdf_split[FF] += w_i * p
            if FF == 'PP':
                if '_ct_' not in a_res:
                    if 'ratio' in a_res:
                        pdf_split['ratio_PP'] += w_i * p
                    else:
                        pdf_split['no_ratio_PP'] += w_i * p
                    pdf_split['no_ct_PP'] += w_i * p
                else:
                    pdf_split['ct_PP'] += w_i * p
            if '_ct_' not in a_res:
                pdf_split[FF+'_xpt'] += w_i * p
        self.avg = avg
        self.pdf = pdf
        self.cdf = cdf
        self.pdf_split = pdf_split
        self.model_var  = np.sum(self.weights * np.array([r.mean**2 for r in results]))
        self.model_var += -self.avg['FKFpi'].mean**2
        print('-----------------------------------------------------------------------------------')
        print('%37s &         %s +- %.4f' %('Bayes Model Avg: FK/Fpi', self.avg['FKFpi'], np.sqrt(self.model_var)))
        for k in var_avg:
            e = '%.4f' %np.sqrt(var_avg[k])
            print('%37s           %9s     %s' %('',e[-2:],k))
        for k in ['dF_iso_xpt', 'dF_iso_xpt2']:
            if self.avg[k].mean < 0:
                print('%37s          %s   %s' %('',self.avg[k],k))
            else:
                print('%37s           %s    %s' %('',self.avg[k],k))
        self.avg['dF_iso_avg'] = avg_iso(self.avg['dF_iso_xpt'],self.avg['dF_iso_xpt2'])
        print('%37s          %s   %s' %('',self.avg['dF_iso_avg'],'dF_iso_avg'))
        #print('-----------------------------------------------------------------------------------')
        self.avg['FK+/Fpi+'] = self.avg['FKFpi'] + self.avg['dF_iso_avg']
        print('                      ----------------------------------------------------')
        print('%37s           %s +- %.4f     |' %('| FK+/Fpi+    =', self.avg['FK+/Fpi+'], np.sqrt(self.model_var)))
        print('                      ----------------------------------------------------')

    def plot_bma_hist(self,hist_type,save_fig=False):
        hist = plt.figure('hist_'+hist_type, figsize=self.fig_size)
        ax   = plt.axes(self.plt_axes)
        # histogram splitting
        if hist_type == 'FF':
            ax.plot(self.pdf_x, self.pdf, color='k')
            ax.fill_between(x=self.pdf_x, y1=self.pdf, color='k', alpha=.2)
            # 95%
            lidx95 = abs(self.cdf-0.025).argmin()
            uidx95 = abs(self.cdf-0.975).argmin()
            ax.fill_between(x=self.pdf_x[lidx95:uidx95],y1=self.pdf[lidx95:uidx95],\
                facecolor='k',edgecolor='k',alpha=0.1)
            #68%
            lidx68 = abs(self.cdf-0.158655254).argmin()
            uidx68 = abs(self.cdf-0.841344746).argmin()
            ax.fill_between(x=self.pdf_x[lidx68:uidx68],y1=self.pdf[lidx68:uidx68],\
                facecolor='k',edgecolor='k',alpha=0.1)
            # black lines
            ax.errorbar(x=[self.pdf_x[lidx95],self.pdf_x[lidx95]],\
                y=[0,self.pdf[lidx95]],color='k')#,lw=0.5)
            ax.errorbar(x=[self.pdf_x[uidx95],self.pdf_x[uidx95]],\
                y=[0,self.pdf[uidx95]],color='k')#,lw=0.5)
            ax.errorbar(x=[self.pdf_x[lidx68],self.pdf_x[lidx68]],\
                y=[0,self.pdf[lidx68]],color='k')#,lw=0.5)
            ax.errorbar(x=[self.pdf_x[uidx68],self.pdf_x[uidx68]],\
                y=[0,self.pdf[uidx68]],color='k')#,lw=0.5)

            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['PP'],\
                color='r',alpha=0.6,label=r'$F^2 \rightarrow F_\pi^2$')
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['PK'],\
                color='g',alpha=0.6,label=r'$F^2 \rightarrow F_\pi F_K$')
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['KK'],\
                color='b',alpha=0.6,label=r'$F^2 \rightarrow F_K^2$')
            ax.set_ylabel(r'Bayes Model Avg PDF',fontsize=self.fs_text)
        elif hist_type == 'FF_xpt':
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['PP_xpt'],\
                color='r',alpha=0.6,label=r'$F^2 \rightarrow F_\pi^2$')
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['PK_xpt'],\
                color='g',alpha=0.6,label=r'$F^2 \rightarrow F_\pi F_K$')
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['KK_xpt'],\
                color='b',alpha=0.6,label=r'$F^2 \rightarrow F_K^2$')
            ax.set_ylabel(r'PDFs from XPT fits',fontsize=self.fs_text)
        elif hist_type == 'ratio':
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['no_ratio_PP'],\
                color='b',alpha=0.6,label=r'w/o ratio fit $(F_\pi^2)$')
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['ratio_PP'],\
                color='r',alpha=0.6,label=r'w/  ratio fit $(F_\pi^2)$')
            ax.set_ylabel(r'PDFs from ratio or not',fontsize=self.fs_text)
        elif hist_type == 'ct':
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['ct_PP'],\
                color='r',alpha=0.6,label=r'N2LO=ct $(F_\pi^2)$')
            ax.fill_between(x=self.pdf_x,y1=self.pdf_split['no_ct_PP'],\
                color='b',alpha=0.6,label=r'N2LO=XPT $(F_\pi^2)$')
            ax.set_ylabel(r'PDFs from N2LO ct or full XPT',fontsize=self.fs_text)

        ax.set_xlim([1.1675,1.2275])
        ax.set_ylim(ymin=0)
        ax.set_xlabel(r'$F_K / F_\pi$',fontsize=self.fs_text)
        ax.legend(fontsize=self.fs_leg)
        if save_fig:
            plt.savefig('figures/hist_'+hist_type+'.pdf', transparent=True)


def uncertainty_breakdown(result,key,print_error=False):
    stat = dict()
    xpt  = dict()
    disc = dict()
    for k in result.prior:
        if isinstance(k, tuple):
            stat[k] = result.prior[k]
        elif 's' in k:
            disc[k] = result.prior[k]
        else:
            xpt[k]  = result.prior[k]
    uncertainties = dict()
    uncertainties['stat_xy']    = result.phys[key].partialsdev(result.y,stat)
    uncertainties['xpt']        = result.phys[key].partialsdev(xpt)
    uncertainties['cont']       = result.phys[key].partialsdev(disc)
    uncertainties['phys_point'] = result.phys[key].partialsdev(result.phys_point)
    if print_error:
        for k in uncertainties:
            print('%10s   %f' %(k,uncertainties[k]))
    else:
        return uncertainties

def check_for_duplicates(list_of_elems):
    ''' Check if given list contains any duplicates '''
    if len(list_of_elems) == len(set(list_of_elems)):
        return False
    else:
        return True


def sys_models(switches):
    def check_model(sys_val,models,nnlo=False,nnnlo=False):
        for model in models:
            if 'taylor' not in model:
                new_model = model+sys_val
                if nnlo:
                    if (sys_val not in model) and (new_model not in models):
                        if nnnlo:
                            if 'nnlo' in model and 'nnnlo' not in model:
                                models.append(new_model)
                        else:
                            if 'nnlo' in model:
                                models.append(new_model)
                else:
                    if (sys_val not in model) and (new_model not in models):
                        models.append(new_model)

    models = switches['ansatz']['models'].copy()
    if switches['sys']['FV']:
        check_model('_FV',models)
    if switches['sys']['alphaS']:
        check_model('_alphaS',models,nnlo=True)
    if switches['sys']['nnlo_ct']:
        check_model('_ct',models,nnlo=True)
    if switches['sys']['logSq']:
        check_model('_logSq',models,nnlo=True)
    if switches['sys']['a4']:
        check_model('_a4',models,nnlo=True,nnnlo=True)
    if switches['sys']['ratio']:
        for model in models:
            model_ratio = model.replace('xpt','xpt-ratio').replace('ma','ma-ratio')
            if '-ratio' not in model and model_ratio not in models:
                models.append(model_ratio)
    models_FPK = []
    for model in models:
        if switches['sys']['Lam_chi']:
            for FPK in switches['scales']:
                models_FPK.append(model+'_'+FPK)
        else:
            models_FPK.append(model+'_'+switches['scale'])
    if switches['debug_models']:
        for model in models_FPK:
            print(model)
    print(len(models_FPK),'models')
    print('Duplicate models?',check_for_duplicates(models_FPK))
    return models_FPK

def gather_model_elements(model):
    eft    = model.split('_')[0]
    order  = model.split('_')[1]
    FF     = model.split('_')[-1]
    fv     = 'FV'     in model
    alphaS = 'alphaS' in model
    ct     = 'ct'     in model
    a4     = 'a4'     in model

    if FF not in ['PP','PK','KK']:
        sys.exit('unrecognized FF choice [PP, PK, KK]: '+FF)

    if 'ratio' in eft or 'longform' in eft:
        model_elements = [eft.replace('-','_')+'_nlo']
    else:
        model_elements = [eft+'_nlo']
    if eft == 'taylor':
        if order in ['nnlo', 'nnnlo']:
            model_elements += ['nnlo_ct']
            if alphaS:
                model_elements += ['nnlo_alphaS']
        if order in ['nnnlo']:
            model_elements += ['nnnlo_ct']
    else:
        if order in ['nnlo','nnnlo']:
            if alphaS:
                model_elements += ['nnlo_alphaS']
            if ct:
                model_elements += ['nnlo_ct']
            else:
                model_elements += ['nnlo_ct','xpt_nnlo_logSq','xpt_nnlo_log','xpt_nnlo_FF_'+FF]
                if 'ratio' in eft:
                    model_elements += ['xpt_nnlo_ratio']


            if a4 and order == 'nnlo':
                model_elements += ['nnnlo_a4']
            if order == 'nnnlo':
                model_elements += ['nnnlo_ct']

    return model_elements, FF, fv

def prior_width_scan(model, fitEnv, fit_model, priors, switches):
    new_priors = dict(priors)
    p2_vals = [2, 5, 10]
    p3_vals = [2, 5, 10]

    logGBF_array = []
    if os.path.exists('data/saved_prior_search.yaml'):
        with open('data/saved_prior_search.yaml','r') as fin:
            prior_grid = yaml.safe_load(fin.read())
            if model in prior_grid:
                for k1 in prior_grid[model]:
                    for k2 in prior_grid[model][k1]:
                        for k3 in prior_grid[model][k1][k2]:
                            for k4 in prior_grid[model][k1][k2][k3]:
                                logGBF_array.append([k1, k2, k3, k4, float(prior_grid[model][k1][k2][k3][k4])])
        shutil.copyfile('data/saved_prior_search.yaml','data/saved_prior_search.yaml.bak')
    else:
        prior_grid = dict()

    if model not in prior_grid:
        prior_grid[model] = dict()
    nnlo_x_lst = ['p_4','k_4']
    nnlo_a_lst = ['s_4','saS_4']
    n3lo_x_lst = ['kp_6','k_6','p_6']
    n3lo_a_lst = ['s_6','sk_6','sp_6']

    n_p2 = len(p2_vals)
    n_p3 = len(p3_vals)
    i_search = 1
    i_tot = n_p2**2 * n_p3**2
    for i2x,p2x in enumerate(p2_vals):
        if p2x not in prior_grid[model]:
            prior_grid[model][p2x] = dict()
        for k2x in nnlo_x_lst:
            new_priors[k2x] = gv.gvar(0,p2x)
        for i2a,p2a in enumerate(p2_vals):
            if p2a not in prior_grid[model][p2x]:
                prior_grid[model][p2x][p2a] = dict()
            for k2a in nnlo_a_lst:
                new_priors[k2a] = gv.gvar(0,p2a)
            for i3x,p3x in enumerate(p3_vals):
                if p3x not in prior_grid[model][p2x][p2a]:
                    prior_grid[model][p2x][p2a][p3x] = dict()
                for k3x in n3lo_x_lst:
                    new_priors[k3x] = gv.gvar(0,p3x)
                for i3a,p3a in enumerate(p3_vals):
                    for k3a in n3lo_a_lst:
                        new_priors[k3a] = gv.gvar(0,p3a)
                    if p3a not in prior_grid[model][p2x][p2a][p3x]:
                        tmp_result = fitEnv.fit_data(new_priors)
                        lgbf = float(tmp_result.logGBF)
                        prior_grid[model][p2x][p2a][p3x][p3a] = lgbf
                        logGBF_array.append([ p2x, p2a, p3x, p3a, lgbf] )
                    else:
                        lgbf = float(prior_grid[model][p2x][p2a][p3x][p3a])
                        logGBF_array.append([ p2x, p2a, p3x, p3a, lgbf] )
                    if switches['prior_verbose']:
                        print('%4d / %4d: nnlo_x = %.2f,   nnlo_a = %.2f,   n3lo_x = %.2f,   n3lo_a = %.2f:   logGBF=%f' \
                            %(i_search, i_tot, p2x, p2a, p3x, p3a, lgbf))
                    else:
                        sys.stdout.write("%4d / %4d\r" %(i_search, i_tot))
                        sys.stdout.flush()
                    i_search += 1
    logGBF_array = np.array(logGBF_array)
    i_l_max = np.argmax(logGBF_array[:,4])
    vals = ''
    vals = vals.join("& %5s   " %str(k) for k in logGBF_array[i_l_max][:-1])
    vals = vals + "& %f" %logGBF_array[i_l_max][-1]
    print('%37s' %model, vals)
    prior_file = open('data/saved_prior_search.yaml', 'w')
    yaml.dump(prior_grid, prior_file)
    prior_file.close()

def avg_iso(a,b):
    ''' We know this correction is negative - so this routing assumes that.
        - Take the larger in magnitude term
        - Take the largest uncertainty
        - add 25% for SU(3) breaking
    '''
    max_err = max([a.sdev, b.sdev])
    avg_err = 0.5*(a+b).sdev
    sig     = np.sqrt( (max_err**2 - avg_err**2) / ((a+b).mean/2)**2)
    if a < b:
        return 0.5*(a+b) * gv.gvar(1, sig) * gv.gvar(1,.25) + (a-b).mean/2
    else:
        return 0.5*(a+b) * gv.gvar(1, sig) * gv.gvar(1,.25) + (b-a).mean/2
