import lsqfit
import numpy as np
import gvar as gv
import sys
import os
import functools

import fitter.special_functions as sf

class fitter(object):

    def __init__(self, prior, fit_data, model_info, observables, ensemble_mapping=None, prior_interpolation=None, simultaneous_extrapolation=True):
        self.prior = prior
        self.prior_interpolation = prior_interpolation
        self.fit_data = fit_data
        self.model_info = model_info.copy()
        self.observables = observables
        
        # attributes of fitter object to fill later
        self.empbayes_grouping = None
        self._counter = {'iters' : 0, 'evals' : 0} # To force empbayes_fit to converge?
        self._fit_interpolation = {} # cache results
        self._simultaneous_interpolation = False
        self._simultaneous_extrapolation = simultaneous_extrapolation
        self._ensemble_mapping = ensemble_mapping # Necessary for LO t0, w0 interpolations 


    def __str__(self):
        return str(self.fit)


    @functools.cached_property
    def fit(self):
        models = self._make_models()
        y_data = {self.model_info['name']+'_'+obs : self.y[obs] for obs in self.observables}
        prior = self._make_prior()

        fitter = lsqfit.MultiFitter(models=models)
        fit = fitter.lsqfit(data=y_data, prior=prior, fast=False, mopt=False)

        return fit


    def fit_interpolation(self, simultaneous_interpolation=False):
        if simultaneous_interpolation not in self._fit_interpolation:
            make_gvar = lambda g : gv.gvar(gv.mean(g), gv.sdev(g))
            data = {}
            for obs in self.observables:
                if obs == 'w0':
                    data[self.model_info['name']+'_interpolation_w0'] = 1 / make_gvar(self.fit_data['a/w'])
                elif obs == 't0':
                    data[self.model_info['name']+'_interpolation_t0'] = make_gvar(self.fit_data['t/a^2'])

                if simultaneous_interpolation:
                    data[self.model_info['name']+'_'+obs] = self.y[obs]

            models = self._make_models(interpolation=True, simultaneous_interpolation=simultaneous_interpolation)
            prior = self._make_prior(interpolation=True, simultaneous_interpolation=simultaneous_interpolation)

            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)

            self._fit_interpolation[simultaneous_interpolation] = fit

        return self._fit_interpolation[simultaneous_interpolation]

    @functools.cached_property
    def y(self):
        make_gvar = lambda g : gv.gvar(gv.mean(g), gv.sdev(g))

        output = {}
        for obs in self.observables:
            if obs == 'w0':
                if self.model_info['chiral_cutoff'] == 'Fpi':
                    output['w0'] = self.fit_data['mO'] / self.fit_data['a/w']
                else: # In Omega-type fits, shifting lam_chi will also shift y, leading to nonsensical fits 
                    output['w0'] = make_gvar(self.fit_data['mO'] / self.fit_data['a/w'])
            elif obs == 't0':
                if self.model_info['chiral_cutoff'] == 'Fpi':
                    output['t0'] = np.sqrt(self.fit_data['t/a^2']) *self.fit_data['mO'] 
                else: # In Omega-type fits, shifting lam_chi will also shift y, leading to nonsensical fits 
                    output['t0'] = make_gvar(np.sqrt(self.fit_data['t/a^2']) *self.fit_data['mO'])

        return output


    def _empbayes_groupings(self):
        zkeys = {}

        if self.empbayes_grouping == 'all':
            for param in self.prior:
                if param != 'c0': # Have a good prior for this already
                    zkeys[param] = [param]

        elif self.empbayes_grouping == 'order':
            zkeys['chiral_nlo'] = ['A_l', 'A_s']
            zkeys['chiral_n2lo'] = ['A_ll', 'A_ss', 'A_ls', 'A_ll_g']
            zkeys['chiral_n3lo'] = ['A_lll', 'A_lls', 'A_lss', 'A_sss', 'A_lll_g', 'A_lls_g', 'A_lll_gg']
            zkeys['latt_nlo'] = ['A_a', 'A_alpha']
            zkeys['latt_n2lo'] = ['A_aa', 'A_al', 'A_as']
            zkeys['latt_n3lo'] = ['A_aaa', 'A_aal', 'A_aas', 'A_all', 'A_als', 'A_ass']

        elif self.empbayes_grouping == 'disc':
            zkeys['chiral'] = ['A_l', 'A_s', 'A_ll', 'A_ss', 'A_ls', 'A_ll_g', 'A_lll', 'A_lls', 'A_lss', 'A_sss', 'A_lll_g', 'A_lls_g', 'A_lll_gg']
            zkeys['disc'] = ['A_a', 'A_alpha', 'A_aa', 'A_al', 'A_as', 'A_aaa', 'A_aal', 'A_aas', 'A_all', 'A_als', 'A_ass']

        elif self.empbayes_grouping == 'alphas':
            zkeys['alphas'] = ['A_alpha']

        elif self.empbayes_grouping == 'alphas_eps2a':
            zkeys['alphas'] = ['A_alpha', 'A_a']

        elif self.empbayes_grouping == 'disc_only':
            zkeys['disc'] = ['A_a', 'A_aa', 'A_al', 'A_as']

        all_keys = [obs+'::'+key for key in [k for g in zkeys for k in zkeys[g]] for obs in self.observables]
        prior_keys = list(self._make_prior())
        ignored_keys = set(all_keys) - set(prior_keys)

        # Don't determine empirical priors in param not in model
        for group in zkeys:
            for key in ignored_keys:
                if key in ignored_keys and key in zkeys[group]:
                    zkeys[group].remove(key)

        # Remove empty groupings
        for group in list(zkeys):
            if len(zkeys[group]) == 0:
                del(zkeys[group])
        
        return zkeys


    @functools.cache
    def _make_empbayes_fit(self, empbayes_grouping='order'):
        self.empbayes_grouping = empbayes_grouping
        self._counter = {'iters' : 0, 'evals' : 0}

        z0 = gv.BufferDict()
        for group in self._empbayes_groupings():
            for obs in self.observables:
                z0[(obs, group)] = 0 # separate zkeys for each observable


        # Might need to change minargs default values for empbayes_fit to converge:
        # tol=1e-8, svdcut=1e-12, debug=False, maxit=1000, add_svdnoise=False, add_priornoise=False
        # Note: maxit != maxfev. See https://github.com/scipy/scipy/issues/3334
        # For Nelder-Mead algorithm, maxfev < maxit < 3 maxfev?

        # For debugging. Same as 'callback':
        # https://github.com/scipy/scipy/blob/c0dc7fccc53d8a8569cde5d55673fca284bca191/scipy/optimize/optimize.py#L651
        def analyzer(arg):
            self._counter['evals'] += 1
            print("\nEvals: ", self._counter['evals'], arg,"\n")
            print(type(arg[0]))
            return None
        
        models = self._make_models()
        fitter = lsqfit.MultiFitter(models=models)

        fit, z = fitter.empbayes_fit(z0, fitargs=self._make_fitargs, maxit=200, analyzer=None, tol=0.1)
        print(z)
        return fit


    def _make_fitargs(self, z):
        y_data = {self.model_info['name']+'_'+obs : self.y[obs] for obs in self.observables}
        prior = self._make_prior()

        # Helps with convergence (minimizer doesn't use extra digits -- bug in lsqfit?)
        capped = lambda x, x_min, x_max : np.max([np.min([x, x_max]), x_min])

        zkeys = self._empbayes_groupings()
        val_min = 1e-2
        val_max = 1e3
        for (obs, group) in z.keys():
            for param in prior.keys():
                if param.split('::')[0] == obs and param.split('::')[-1] in zkeys[group]:
                    prior[param] = gv.gvar(0, capped(np.exp(z[obs, group]), val_min, val_max))

        self._counter['iters'] += 1
        print(self._counter['iters'], ' ', z)

        return dict(data=y_data, prior=prior)


    def _make_models(self, model_info=None, interpolation=False, y_data=None, simultaneous_interpolation=False):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])
        if interpolation:
            model_info_interpolation = {
                'name' : model_info['name'] + '_interpolation',
                'chiral_cutoff': 'Fpi',
                'order': 'n2lo',
                'latt_ct': 'n2lo',
                'include_log': False,
                'include_log2': False,
                'include_fv': False,
                'include_alphas': False,
                'exclude': []
            }

            for obs in self.observables:
                datatag = model_info_interpolation['name']+'_'+obs
                models = np.append(models, model_interpolation(datatag=datatag, model_info=model_info_interpolation, ens_mapping=self._ensemble_mapping, observable=obs))

            if not simultaneous_interpolation:
                return models

        for obs in self.observables:
            datatag = model_info['name']+'_'+obs
            models = np.append(models, model(datatag=datatag, model_info=model_info, observable=obs))

        return models


    def _make_prior(self, fit_data=None, interpolation=False, simultaneous_interpolation=False):
        if fit_data is None:
            fit_data = self.fit_data

        output = gv.BufferDict()
        if interpolation or simultaneous_interpolation:
            prior = self.prior_interpolation

            for obs in self.observables:
            
                for aXX in np.unique([ens[:3] for ens in self._ensemble_mapping]):
                    output[obs+'::c0'+aXX] = prior[obs]['c0'+aXX]

                # add priors for other LECs
                for key in set(list(prior[obs])).difference(['c0a06', 'c0a09', 'c0a12', 'c0a15']):
                    output[obs+'::'+key] = prior[obs][key]

            for key in ['mpi', 'mk', 'lam_chi']:
                output[key] = fit_data[key]

            if not simultaneous_interpolation:
                return output

        prior = self.prior

        newprior = gv.BufferDict()
        for obs in self.observables:

            # xpt terms
            # lo
            newprior['c0'] = prior[obs]['c0']

            # nlo
            if self.model_info['order'] in ['nlo', 'n2lo', 'n3lo']:
                newprior['A_l'] = prior[obs]['A_l']
                newprior['A_s'] = prior[obs]['A_s']
                newprior['A_a'] = prior[obs]['A_a']

            # n2lo
            if self.model_info['order'] in ['n2lo', 'n3lo']:
                newprior['A_aa'] = prior[obs]['A_aa']
                newprior['A_al'] = prior[obs]['A_al']
                newprior['A_as'] = prior[obs]['A_as']
                newprior['A_ll'] = prior[obs]['A_ll']
                newprior['A_ls'] = prior[obs]['A_ls']
                newprior['A_ss'] = prior[obs]['A_ss']

                if self.model_info['include_log']:
                    newprior['A_ll_g'] = prior[obs]['A_ll_g']

            # n3lo
            if self.model_info['order'] in ['n3lo']:
                newprior['A_aaa'] = prior[obs]['A_aaa']
                newprior['A_aal'] = prior[obs]['A_aal']
                newprior['A_aas'] = prior[obs]['A_aas']
                newprior['A_all'] = prior[obs]['A_all']
                newprior['A_als'] = prior[obs]['A_als']
                newprior['A_ass'] = prior[obs]['A_ass']

                newprior['A_lll'] = prior[obs]['A_lll']
                newprior['A_lls'] = prior[obs]['A_lls']
                newprior['A_lss'] = prior[obs]['A_lss']

                newprior['A_sss'] = prior[obs]['A_sss']

                if self.model_info['include_log']:
                    newprior['A_lll_g'] = prior[obs]['A_lll_g']
                    newprior['A_lls_g'] = prior[obs]['A_lls_g']
                if self.model_info['include_log2']:
                    newprior['A_lll_gg'] = prior[obs]['A_lll_gg']

            # latt terms
            if self.model_info['latt_ct'] in ['nlo', 'n2lo', 'n3lo']:
                newprior['A_a'] = prior[obs]['A_a']
            if self.model_info['latt_ct'] in ['n2lo', 'n3lo']:
                newprior['A_aa'] = prior[obs]['A_aa']
            if self.model_info['latt_ct'] in ['n3lo']:
                newprior['A_aaa'] = prior[obs]['A_aaa']

            # alpha_s corrections
            if self.model_info['include_alphas']:
                newprior['A_alpha'] = prior[obs]['A_alpha']

            for key in self.model_info['exclude']:
                if key in newprior.keys():
                    del(newprior[key])

            for key in newprior:
                output[obs+'::'+key] = newprior[key]

        # Move fit_data into prior
        for key in ['mpi', 'mk', 'lam_chi', 'L', 'alpha_s']:
            if key in fit_data:
                output[key] = fit_data[key]

        if self.model_info['eps2a_defn'] == 'w0_original':
            output['eps2_a'] = fit_data['a/w:orig']**2 / 4
        elif self.model_info['eps2a_defn'] == 'w0_improved':
            output['eps2_a'] = fit_data['a/w:impr']**2 / 4
        elif self.model_info['eps2a_defn'] == 't0_original':
            output['eps2_a'] = 1 / fit_data['t/a^2:orig'] / 4
        elif self.model_info['eps2a_defn'] == 't0_improved':
            output['eps2_a'] = 1 / fit_data['t/a^2:impr'] / 4
        elif self.model_info['eps2a_defn'] == 'variable':
            for obs in self.observables:
                if obs == 'w0':
                    output['w0::eps2_a'] = fit_data['a/w']**2 / 4
                elif obs == 't0':
                    output['t0::eps2_a'] = 1 / fit_data['t/a^2'] / 4

        return output


class model(lsqfit.MultiFitterModel):

    def __init__(self, datatag, model_info, observable, **kwargs):
        super(model, self).__init__(datatag)

        # Model info
        self.debug = False
        self.model_info = model_info
        self.observable = observable
        self._p = None

    
    def key(self, key):
        if key.startswith(self.observable):
            return key
        elif key in self._p:
            return key
        else:
            return self.observable+'::'+key


    def fitfcn(self, p, fit_data=None, xi=None, debug=None):
        if debug:
            self.debug = debug
            self.debug_table = {}

        if fit_data is not None:
            for key in fit_data.keys():
                p[key] = fit_data[key]

        self._p = p

        for key in self.model_info['exclude']:
            p[key] = 0

        # Variables
        if xi is None:
            xi = {}
        if 'l' not in xi:
            xi['l'] = (p['mpi'] / p['lam_chi'])**2
        if 's' not in xi:
            xi['s'] = (2 *p['mk']**2 - p['mpi']**2) / p['lam_chi']**2
        if 'a' not in xi:
            if self.model_info['eps2a_defn'] == 'variable':
                xi['a'] = p[self.key('eps2_a')] #p['a/w']**2 / 4
            else:
                xi['a'] = p['eps2_a']


        # lo
        output = p[self.key('c0')]

        if self.debug:
            self.debug_table['lo_ct'] = output

        # nlo
        if self.model_info['order'] in ['nlo', 'n2lo', 'n3lo']:
            output += self.fitfcn_nlo_ct(p, xi)
            if self.model_info['include_alphas']:
                output += self.fitfcn_nlo_latt_alphas(p, xi)
                
        elif self.model_info['latt_ct'] in ['nlo', 'n2lo', 'n3lo']: 
            output += self.fitfcn_nlo_latt_ct(p, xi)
            if self.model_info['include_alphas']:
                output += self.fitfcn_nlo_latt_alphas(p, xi)

        # n2lo 
        if self.model_info['order'] in ['n2lo', 'n3lo']:
            output += self.fitfcn_n2lo_ct(p, xi)
            if self.model_info['include_log']:
                output += self.fitfcn_n2lo_log(p, xi)

        elif self.model_info['latt_ct'] in ['n2lo', 'n3lo']:
            output += self.fitfcn_n2lo_latt_ct(p, xi)

        # n3lo
        if self.model_info['order'] in ['n3lo']:
            output += self.fitfcn_n3lo_ct(p, xi)
            if self.model_info['include_log']:
                output += self.fitfcn_n3lo_log(p, xi)
            if self.model_info['include_log2']:
                output += self.fitfcn_n3lo_log_sq(p, xi)
                
        elif self.model_info['latt_ct'] in ['n3lo']:
            output += self.fitfcn_n3lo_latt_ct(p, xi)



        for key in self.model_info['exclude']:
            del(p[key])

        if debug:
            #print(gv.tabulate(self.debug_table))
            temp_string = ''
            for key in self.debug_table:
                temp_string +='  % .15f:  %s\n' %(gv.mean(self.debug_table[key]), key)
            temp_string +='   -----\n'
            temp_string +='  % .15f:  %s\n' %(gv.mean(output), 'total')
            print(temp_string)

        return output


    def fitfcn_nlo_ct(self, p, xi):
        output = p[self.key('A_l')] *xi['l'] + p[self.key('A_s')] *xi['s'] + p[self.key('A_a')] *xi['a']

        if self.debug:
            self.debug_table['nlo_ct'] = output

        return output


    def fitfcn_n2lo_ct(self, p, xi):     
        output = ( 
            + p[self.key('A_aa')] *xi['a'] *xi['a']
            + p[self.key('A_al')] *xi['a'] *xi['l']
            + p[self.key('A_as')] *xi['a'] *xi['s']
            + p[self.key('A_ll')] *xi['l'] *xi['l']
            + p[self.key('A_ls')] *xi['l'] *xi['s']
            + p[self.key('A_ss')] *xi['s'] *xi['s']
        )

        if self.debug:
            self.debug_table['n2lo_ct'] = output

        return output


    def fitfcn_n2lo_log(self, p, xi):
        if self.model_info['include_fv']:
            output = p[self.key('A_ll_g')] *xi['l']**2 *sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10)
        else:
            output = p[self.key('A_ll_g')] *xi['l']**2 *np.log(xi['l'])

        if self.debug:
            self.debug_table['n2lo_log'] = output

        return output


    def fitfcn_n3lo_ct(self, p, xi):

        output = (
            + p[self.key('A_aaa')] *xi['a'] *xi['a'] *xi['a']
            + p[self.key('A_aal')] *xi['a'] *xi['a'] *xi['l']
            + p[self.key('A_aas')] *xi['a'] *xi['a'] *xi['s']
            + p[self.key('A_all')] *xi['a'] *xi['l'] *xi['l']
            + p[self.key('A_als')] *xi['a'] *xi['l'] *xi['s']
            + p[self.key('A_ass')] *xi['a'] *xi['s'] *xi['s']

            + p[self.key('A_lll')] *xi['l'] *xi['l'] *xi['l']
            + p[self.key('A_lls')] *xi['l'] *xi['l'] *xi['s']
            + p[self.key('A_lss')] *xi['l'] *xi['s'] *xi['s']

            + p[self.key('A_sss')] *xi['s'] *xi['s'] *xi['s']
        )

        if self.debug:
            self.debug_table['n3lo_ct'] = output

        return output


    def fitfcn_n3lo_log(self, p, xi):
        if self.model_info['include_fv']:
            output = (
                + p[self.key('A_lll_g')] *xi['l']**3 *sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10)
                + p[self.key('A_lls_g')] *xi['l']**2 *xi['s'] *sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10)
            )
        else:
            output = (
                + p[self.key('A_lll_g')] *xi['l']**3 *np.log(xi['l'])
                + p[self.key('A_lls_g')] *xi['l']**2 *xi['s'] *np.log(xi['l'])
            )

        if self.debug:
            self.debug_table['n3lo_log'] = output

        return output


    def fitfcn_n3lo_log_sq(self, p, xi):
        if self.model_info['include_fv']:
            output = p[self.key('A_lll_gg')] *xi['l']**3 *(
                + (sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10))**2
                #- (np.log(xi['l']))**2
            )
        else:
            output = p[self.key('A_lll_gg')] *xi['l']**3 *(np.log(xi['l']))**2

        if self.debug:
            self.debug_table['n3lo_log_sq'] = output

        return output


    def fitfcn_nlo_latt_alphas(self, p, xi):
        output = p[self.key('A_alpha')] *xi['a'] *p['alpha_s']

        if self.debug:
            self.debug_table['nlo_alphas'] = output

        return output


    def fitfcn_nlo_latt_ct(self, p, xi):
        output = p[self.key('A_a')] *xi['a']

        if self.debug:
            self.debug_table['nlo_latt'] = output

        return output


    def fitfcn_n2lo_latt_ct(self, p, xi):
        output = p[self.key('A_aa')] *xi['a']**2

        if self.debug:
            self.debug_table['n2lo_latt'] = output

        return output


    def fitfcn_n3lo_latt_ct(self, p, xi):
        output = p[self.key('A_aaa')] *xi['a']**3

        if self.debug:
            self.debug_table['n3lo_latt'] = output

        return output


    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]


class model_interpolation(lsqfit.MultiFitterModel):

    def __init__(self, datatag, model_info, ens_mapping=None, observable=None, **kwargs):
        super(model_interpolation, self).__init__(datatag)

        # Model info
        self.model_info = model_info
        self.ens_mapping = ens_mapping
        self.observable = observable
        self._p = None

    
    def key(self, key):
        if key.startswith(self.observable):
            return key
        elif key in self._p:
            return key
        else:
            return self.observable+'::'+key


    def fitfcn(self, p, fit_data=None, xi=None, latt_spacing=None, observable=None):
        if fit_data is not None:
            for key in fit_data.keys():
                p[key] = fit_data[key]

        self._p = p

        for key in self.model_info['exclude']:
            p[key] = 0


        # Variables
        if xi is None:
            xi = {}
        if 'l' not in xi:
            xi['l'] = (p['mpi'] / p['lam_chi'])**2
        if 's' not in xi:
            xi['s'] = (2 *p['mk']**2 - p['mpi']**2) / p['lam_chi']**2

        y_ch = self.fitfcn_lo_ct(p, xi, latt_spacing)

        if 'a' not in xi:
            if self.observable == 'w0':
                xi['a'] =  1 / (2 *y_ch)**2
            elif self.observable == 't0':
                xi['a'] =  1 / (4 *y_ch)

        # lo
        #output = w0ch_a

        # nlo
        #output += w0ch_a *self.fitfcn_nlo_ct(p, xi)

        # n2lo
        #output += w0ch_a *self.fitfcn_n2lo_ct(p, xi)
        #output += w0ch_a *self.fitfcn_n2lo_log(p, xi)

        #return output

        #print(np.sqrt(xi['a'] ))

        #if isinstance(w0ch_a[0], gv._gvarcore.GVar):
        #    print(gv.evalcorr([w0ch_a[0], self.fitfcn_lo_ct(p, latt_spacing)[0]]))

        #output = 1 / (2 *np.sqrt(xi['a'])) *(
        output = y_ch *(
            + 1
            + self.fitfcn_nlo_ct(p, xi)
            + self.fitfcn_n2lo_ct(p, xi)
            + self.fitfcn_n2lo_log(p, xi)
        )

        #print('good')
        return output


    def fitfcn_lo_ct(self, p, xi, latt_spacing=None):

        if latt_spacing == 'a06':
            output = p[self.key('c0a06')]
        elif latt_spacing == 'a09':
            output= p[self.key('c0a09')]
        elif latt_spacing == 'a12':
            output = p[self.key('c0a12')]
        elif latt_spacing == 'a15':
            output = p[self.key('c0a15')]

        else:
            output = xi['l'] *xi['s'] *0 # returns correct shape
            for j, ens in enumerate(self.ens_mapping):
                if ens[:3] == 'a06':
                    output[j] = p[self.key('c0a06')]
                elif ens[:3] == 'a09':
                    output[j] = p[self.key('c0a09')]
                elif ens[:3] == 'a12':
                    output[j] = p[self.key('c0a12')]
                elif ens[:3] == 'a15':
                    output[j] = p[self.key('c0a15')]
                else:
                    output[j] = 0

        return output


    def fitfcn_nlo_ct(self, p, xi):
        output = (
            + p[self.key('k_l')] *xi['l'] 
            + p[self.key('k_s')] *xi['s'] 
            + p[self.key('k_a')] *xi['a']
        )
        return output


    def fitfcn_n2lo_ct(self, p, xi):     
        output = ( 
            + p[self.key('k_aa')] *xi['a'] *xi['a']
            + p[self.key('k_al')] *xi['a'] *xi['l']
            + p[self.key('k_as')] *xi['a'] *xi['s']
            + p[self.key('k_ll')] *xi['l'] *xi['l']
            + p[self.key('k_ls')] *xi['l'] *xi['s']
            + p[self.key('k_ss')] *xi['s'] *xi['s']
        )
        return output


    def fitfcn_n2lo_log(self, p, xi):
        output = p[self.key('k_ll_g')] *xi['l']**2 *np.log(xi['l'])
        return output


    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]