import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import copy

# Set defaults for plots
import matplotlib as mpl
'''
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = [6.75, 6.75/1.618034333]
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True
'''

import fitter.fitter as fit

class fit_manager(object):

    def __init__(self, phys_point_data, fit_data=None, prior=None, model_info=None, simultaneous=False):

        for ens in sorted(list(fit_data)):
            if model_info['chiral_cutoff'] == 'mO':
                fit_data[ens]['lam_chi'] = fit_data[ens]['mO']
                phys_point_data['lam_chi'] = phys_point_data['mO']
            elif model_info['chiral_cutoff'] == 'Fpi':
                fit_data[ens]['lam_chi'] = 4 *np.pi *fit_data[ens]['Fpi']
                phys_point_data['lam_chi'] = 4 *np.pi *phys_point_data['Fpi']

        self.ensembles = list(sorted(fit_data))
        self.model_info = model_info
        self.fit_data = fit_data
        self.fitter = {}
        if simultaneous:
            self.fitter['w0'] = fitter_dict(fit_data=fit_data, input_prior=prior, observables=['w0', 't0'])[model_info]
            self.fitter['t0'] = self.fitter['w0']
        else:
            self.fitter['w0'] = fitter_dict(fit_data=fit_data, input_prior=prior, observables=['w0'])[model_info]
            self.fitter['t0'] = fitter_dict(fit_data=fit_data, input_prior=prior, observables=['t0'])[model_info]

        self.simultaneous = simultaneous
        self._input_prior = prior
        self._phys_point_data = phys_point_data
        self._fit = {}


    def __str__(self):
        output = "Model: %s" %(self.model)
        if self.simultaneous:
            output += '   [simultaneous]'

        observables = ['w0', 't0']
        if self.simultaneous:
            observables.append('t0w0')

        for obs in observables:
            output += '\n---\n'

            if obs == 'w0':
                output += "\nw0: %s\n\n" %(self.w0)
                for a_xx in ['a06', 'a09', 'a12', 'a15']:
                    w0_a = self.interpolate_w0a(latt_spacing=a_xx)
                    output += '  w0/{}: {}'.format(a_xx, w0_a).ljust(22)  + '=> %s/fm: %s\n'%(a_xx, self.w0 / w0_a)

            elif obs == 't0':
                output += "\nsqrt(t0): %s\n\n" %(self.sqrt_t0)
                for a_xx in ['a06', 'a09', 'a12', 'a15']:
                    t0_a2 = self.interpolate_t0a2(latt_spacing=a_xx)
                    output += '  t0/{}^2: {}'.format(a_xx, t0_a2).ljust(22)  + '=> %s/fm: %s\n'%(a_xx, self.sqrt_t0 / np.sqrt(t0_a2))

            elif obs == 't0w0':
                output += "\nsqrt(t0)/w0: %s\n" %(self.sqrt_t0 / self.w0)

            if not self.simultaneous:
                output += '\nParameters:\n'
                my_str = self.fit[obs].format(pstyle='m')
                for item in my_str.split('\n'):
                    for key in self.fit_keys[obs]:
                        re = key+' '
                        if re in item:
                            output += item + '\n'

                output += '\n'
                output += self.fit[obs].format(pstyle=None)

            output += '\nError Budget:\n'
            max_len = np.max([len(key) for key in self.error_budget[obs]])
            for key in {k: v for k, v in sorted(self.error_budget[obs].items(), key=lambda item: item[1], reverse=True)}:
                if obs == 'w0':
                    val_sdev = self.w0.sdev
                elif obs == 't0':
                    val_sdev = self.sqrt_t0.sdev
                elif obs == 't0w0':
                    val_sdev = gv.sdev(self.sqrt_t0 / self.w0)
                else:
                    val_sdev = None
                output += '  '
                output += key.ljust(max_len+1)
                output += '{: .1%}\n'.format((self.error_budget[obs][key]/val_sdev)**2).rjust(7)

        if self.simultaneous:
            output += '\n---\n\nParameters:\n'
            my_str = self.fit['w0'].format(pstyle='m')
            for item in my_str.split('\n'):
                for key in self.fit_keys['w0']:
                    re = key+' '
                    if re in item:
                        output += item + '\n'

            output += '\n'
            output += self.fit['w0'].format(pstyle=None)

        return output


    @property
    def error_budget(self):
        return self._get_error_budget()


    def _get_error_budget(self, verbose=False, **kwargs):
        output = None

        observable_list = ['w0', 't0']
        label_list = ['w0', 't0']
        if self.simultaneous:
            observable_list.append('w0')
            label_list.append('t0w0')

        for label, observable in zip(label_list, observable_list):
            # Fill these out
            phys_keys = list(self.phys_point_data)
            stat_key = 'lam_chi' # Since the input data is mostly correlated, only need uncorrelated x data

            if verbose:
                print('Warning: verbose flag may be inaccurate')
                if output is None:
                    output = ''

                inputs = {}

                # xpt/chiral contributions
                inputs.update({str(param)+' [disc]' : self.fitter[observable].fit.prior[param] for param in self.fitter[observable].fit.prior 
                     if param not in phys_keys and param not in ['w0::eps2_a', 't0::eps2_a'] and 'a' in param })
                
                inputs.update({str(param)+' [xpt]' : self.fitter[observable].fit.prior[param] for param in self.fitter[observable].fit.prior 
                     if param not in phys_keys and param not in ['w0::eps2_a', 't0::eps2_a'] and 'a' not in param })

                # phys point contributions
                inputs.update({str(param)+' [phys]' : self.phys_point_data[param] for param in list(phys_keys)})
                del(inputs['lam_chi [phys]'])

                # stat contribtions
                inputs.update({'x [stat]' : self._get_prior(stat_key)[observable]})
                if 'eps2_a' in self.fitter[observable].fit.prior:
                    inputs.update({'a [stat]' : self.fitter[observable].fit.prior['eps2_a']})
                inputs.update({str(obs)+'[stat]' : self.fitter[observable].y[obs] for obs in self.fitter[observable].y})
                
                if kwargs is None:
                    kwargs = {}
                kwargs.setdefault('percent', False)
                kwargs.setdefault('ndecimal', 10)
                kwargs.setdefault('verify', True)

                if label == 'w0':
                    output += 'observable: ' + label + '\n' + gv.fmt_errorbudget(outputs={'w0' : self.w0}, inputs=inputs, **kwargs)
                    value = self.w0
                elif label == 't0':
                    output += 'observable: ' + label + '\n' + gv.fmt_errorbudget(outputs={'t0' : self.sqrt_t0}, inputs=inputs, **kwargs)
                    value = self.sqrt_t0

                output += 'total:  ' +str(gv.sdev(value)) +'\n'
                output += 'summed: ' +str(np.sqrt(np.sum([value.partialsdev(inputs[key])**2 for key in inputs])))

                output += '\n---\n'

            else: 
                if output is None:
                    output = {}

                output[label] = {}
                if label == 'w0':
                    value = self.w0

                elif label == 't0':
                    value = self.sqrt_t0

                elif label == 't0w0':
                    value = self.sqrt_t0 / self.w0

                output[label]['disc'] = value.partialsdev(
                    [self.fitter[observable].fit.prior[param] for param in self.fitter[observable].fit.prior 
                     if param not in phys_keys and param not in ['w0::eps2_a', 't0::eps2_a'] and 'a' in param]
                )
                output[label]['chiral'] = value.partialsdev(
                    [self.fitter[observable].fit.prior[param] for param in self.fitter[observable].fit.prior 
                    if param not in phys_keys and param not in ['w0::eps2_a', 't0::eps2_a'] and 'a' not in param]
                )
                output[label]['phys'] = value.partialsdev(
                    [self.phys_point_data[param] for param in list(phys_keys)]
                )
                output[label]['stat'] = value.partialsdev(
                    [self.fitter[observable].fit.prior[key] for key in ['eps2_a'] if key in self.fitter[observable].fit.prior]
                    + [self._get_prior(stat_key)[observable]] 
                    + [self.fitter[observable].y[obs] for obs in self.fitter[observable].y]
                )

        return output


    @property
    def fit(self):
        if 'w0' not in self._fit:
            temp_fit = self.fitter['w0'].fit
            self._fit['w0'] = temp_fit

        if 't0' not in self._fit: 
            temp_fit = self.fitter['t0'].fit
            self._fit['t0'] = temp_fit

        return self._fit

    @property
    def fit_info(self):
        fit_info = {}
        fit_info['w0'] = {
            'name' : self.model,
            'w0' : self.w0,
            'logGBF' : self.fit['w0'].logGBF,
            'chi2/df' : self.fit['w0'].chi2 / self.fit['w0'].dof,
            'Q' : self.fit['w0'].Q,
            'phys_point' : self.phys_point_data,
            'error_budget' : self.error_budget['w0'],
            'prior' : self.prior['w0'],
            'posterior' : self.posterior['w0'],
        }
        fit_info['t0'] = {
            'name' : self.model,
            'sqrt_t0' : self.sqrt_t0,
            'logGBF' : self.fit['t0'].logGBF,
            'chi2/df' : self.fit['t0'].chi2 / self.fit['t0'].dof,
            'Q' : self.fit['t0'].Q,
            'phys_point' : self.phys_point_data,
            'error_budget' : self.error_budget['t0'],
            'prior' : self.prior['t0'],
            'posterior' : self.posterior['t0'],
        }
        if self.simultaneous:
            fit_info['t0w0'] = {
                'name' : self.model,
                'sqrt_t0/w0' : self.sqrt_t0/self.w0,
                'logGBF' : self.fit['t0'].logGBF,
                'chi2/df' : self.fit['t0'].chi2 / self.fit['t0'].dof,
                'Q' : self.fit['t0'].Q,
                'phys_point' : self.phys_point_data,
                'error_budget' : self.error_budget['t0w0'],
                'prior' : self.prior['t0'],
                'posterior' : self.posterior['t0'],
            }

        # Need to keep track of correlation between spacings 
        latt_spacings = np.unique([ens[:3] for ens in self.ensembles])
        for axx in latt_spacings:
            fit_info['w0']['w0/a:'+str(axx)] = self.interpolate_w0a(latt_spacing=axx, simultaneous_interpolation=True)
            fit_info['t0']['t0/a2:'+str(axx)] = self.interpolate_t0a2(latt_spacing=axx, simultaneous_interpolation=True)

        return fit_info

    # Returns names of LECs in prior/posterior
    @property
    def fit_keys(self):
        output = {}

        observables = ['w0', 't0']
        keys1 = [obs+'::'+key for obs in ['w0', 't0'] for key in list(self._input_prior[obs].keys())]
        for obs in observables:
            keys2 = list(self.fit[obs].p.keys())
            output[obs] = np.intersect1d(keys1, keys2)
        return output

    @property
    def model(self):
        return self.model_info['name']

    @property
    def phys_point_data(self):
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return copy.deepcopy(self._phys_point_data)
        else:
            return self._phys_point_data[parameter]

    @property
    def posterior(self):
        return self._get_posterior()

    # Returns dictionary with keys fit parameters, entries gvar results
    def _get_posterior(self, param=None):
        output = {}
        if param not in [None, 'all']:
            for obs in ['w0', 't0']:
                output[obs] = {}
                if param in self.fit[obs].p:
                    output[obs] = self.fit[obs].p[param]
                elif obs+'::'+param in self.fit[obs].p:
                    output[obs] = self.fit[obs].p[obs+'::'+param]
                else:
                    raise ValueError('Not a valid posterior key.')
            return output

        for obs in ['w0', 't0']: 
            if param is None:
                temp = {param : self.fit[obs].p[param] for param in self.fit_keys[obs]}
            elif param == 'all':
                temp = self.fit[obs].p
            else:
                temp = self.fit[obs].p[param]

            output[obs] = gv.BufferDict()

            for pkey in temp:
                keys = pkey.split('::')
                if len(keys) == 1:
                    output[obs][pkey] = temp[pkey]
                elif keys[0] == obs:
                    output[obs][keys[1]] = temp[pkey]

        return output

    @property
    def prior(self):
        return self._get_prior()

    def _get_prior(self, param=None):
        output = {}
        if param not in [None, 'all']:
            for obs in ['w0', 't0']:
                output[obs] = {}
                if param in self.fit[obs].prior:
                    output[obs] = self.fit[obs].prior[param]
                elif obs+'::'+param in self.fit[obs].prior:
                    output[obs] = self.fit[obs].prior[obs+'::'+param]
                else:
                    raise ValueError('Not a valid prior key.')
            return output

        for obs in ['w0', 't0']: 
            if param is None:
                temp = {param : self.fit[obs].prior[param] for param in self.fit_keys[obs]}
            elif param == 'all':
                temp = self.fit[obs].prior
            else:
                temp = self.fit[obs].prior[param]

            output[obs] = gv.BufferDict()

            for pkey in temp:
                keys = pkey.split('::')
                if len(keys) == 1:
                    output[obs][pkey] = temp[pkey]
                elif keys[0] == obs:
                    output[obs][keys[1]] = temp[pkey]

        return output

    @property
    def sqrt_t0(self):
        return self.fitfcn(fit_data=copy.deepcopy(self.phys_point_data), observable='t0') / self.phys_point_data['mO'] *self.phys_point_data['hbarc']


    @property
    def w0(self):
        return self.fitfcn(fit_data=copy.deepcopy(self.phys_point_data), observable='w0') / self.phys_point_data['mO'] *self.phys_point_data['hbarc']

    def _extrapolate_to_ens(self, ens=None, phys_params=None, observable=None):
        if phys_params is None:
            phys_params = []

        extrapolated_values = {}
        for j, ens_j in enumerate(self.ensembles):
            posterior = {}
            xi = {}
            if ens is None or (ens is not None and ens_j == ens):
                for param in self.fit[observable].p:
                    shape = self.fit[observable].p[param].shape
                    if param in phys_params:
                        a_fm = self.w0 / self.interpolate_w0a(ens_j[:3])
                        posterior[param] = self.phys_point_data[param] / self.phys_point_data['hbarc'] *a_fm
                    elif shape == ():
                        posterior[param] = self.fit[observable].p[param]
                    else:
                        posterior[param] = self.fit[observable].p[param][j]

                if 'alpha_s' in phys_params:
                    posterior['alpha_s'] = self.phys_point_data['alpha_s']

                if 'xi_l' in phys_params:
                    xi['l'] = self.phys_point_data['mpi']**2 / self.phys_point_data['lam_chi']**2
                if 'xi_s' in phys_params:
                    xi['s'] = (2 *self.phys_point_data['mk']**2 - self.phys_point_data['mpi']**2)/ self.phys_point_data['lam_chi']**2
                if 'xi_a' in phys_params:
                    xi['a'] = 0

                if ens is not None:
                    return self.fitfcn(posterior=posterior, fit_data={}, xi=xi, observable=observable)
                else:
                    extrapolated_values[ens_j] = self.fitfcn(posterior=posterior, fit_data={}, xi=xi, observable=observable)
        return extrapolated_values


    def _fmt_key_as_latex(self, key):
        convert = {
            # data parameters
            'a/w0' : r'$a$ (fm)',
            'L' : r'$L$ (fm)',
            'mpi' : r'$m_\pi$',
            'mk' : r'$m_K$',

            'V' : r'$e^{-m L} / \sqrt{m L}$',
        }

        if key in convert.keys():
            return convert[key]
        else:
            return key


    def fitfcn(self, fit_data=None, posterior=None, xi=None, debug=False, observable=None):
        if fit_data is None:
            fit_data = copy.deepcopy(self.phys_point_data)

        if posterior is None:
            posterior = copy.deepcopy(self.posterior[observable])

        models = self.fitter[observable]._make_models()
        for mdl in models:
            if mdl.observable == observable:
                return mdl.fitfcn(p=posterior, fit_data=fit_data, xi=xi, debug=debug)

    # observable = 'w0' or 't0'
    def fitfcn_interpolation(self, latt_spacing, observable, fit_data=None, posterior=None, xi=None, simultaneous_interpolation=True):
        if fit_data is None:
            fit_data = copy.deepcopy(self.phys_point_data)

        prior_keys = [observable +'::' + key for key in list(self._input_prior[observable+'_interpolation'])]
        param_keys = set(prior_keys).intersection(set(list(self.fitter[observable].fit_interpolation(simultaneous_interpolation).p)))

        posterior = {param : self.fitter[observable].fit_interpolation(simultaneous_interpolation).p[param] 
            for param in prior_keys if param in param_keys}

        for mdl in  self.fitter[observable]._make_models(interpolation=True, y_data=None):
            if mdl.observable == observable:
                return mdl.fitfcn(p=posterior, fit_data=fit_data, xi=xi, latt_spacing=latt_spacing)
            

    def fmt_error_budget(self, **kwargs):
        return self._get_error_budget(verbose=True, **kwargs)


    def interpolate_w0a(self, latt_spacing, simultaneous_interpolation=True):
        return self.fitfcn_interpolation(latt_spacing=latt_spacing, simultaneous_interpolation=simultaneous_interpolation, observable='w0')


    def interpolate_t0a2(self, latt_spacing, simultaneous_interpolation=True):
        return self.fitfcn_interpolation(latt_spacing=latt_spacing, simultaneous_interpolation=simultaneous_interpolation, observable='t0')


    def optimize_prior(self, empbayes_grouping='order'):
        prior = {}
        for observable in ['w0', 't0']:
            temp_prior = self.fitter[observable]._make_empbayes_fit(empbayes_grouping).prior
            prior[observable] = gv.BufferDict()
            for key in self.fit_keys[observable]:
                prior[observable][key] = temp_prior[key]

        return prior


    # Takes keys from posterior (eg, 'A_l' and 'A_s')
    def plot_error_ellipsis(self, x_key, y_key, observable):
        x = self._get_posterior(x_key)[observable]
        y = self._get_posterior(y_key)[observable]


        fig, ax = plt.subplots()

        corr = '{0:.3g}'.format(gv.evalcorr([x, y])[0,1])
        std_x = '{0:.3g}'.format(gv.sdev(x))
        std_y = '{0:.3g}'.format(gv.sdev(y))
        text = ('$R_{x, y}=$ %s\n $\sigma_x =$ %s\n $\sigma_y =$ %s' %(corr,std_x,std_y))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        C = gv.evalcov([x, y])
        eVe, eVa = np.linalg.eig(C)
        for e, v in zip(eVe, eVa.T):
            plt.plot([gv.mean(x)-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x)],
                     [gv.mean(y)-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y)],
                     'k-', lw=2)

        #plt.scatter(x-np.mean(x), y-np.mean(y), rasterized=True, marker=".", alpha=100.0/self.bs_N)
        #plt.scatter(x, y, rasterized=True, marker=".", alpha=100.0/self.bs_N)

        plt.grid()
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlabel(x_key.replace('_', '\_'), fontsize = 24)
        plt.ylabel(y_key.replace('_', '\_'), fontsize = 24)

        fig = plt.gcf()
        plt.close()
        return fig


    def plot_fit(self, param, observable, show_legend=True, ylim=None, ):

        x = {}
        y = {}
        c = {}
            
        #fig = plt.figure(figsize=(6.75, 6.75/1.618034333))
        plt.axes([0.145,0.145,0.85,0.85])

        colors = {
            '06' : '#6A5ACD',#'#00FFFF',
            '09' : '#51a7f9',
            '12' : '#70bf41',
            '15' : '#ec5d57',
        }

        if observable == 'w0':
            latt_spacings = {a_xx[1:] : (1/(2 *self.interpolate_w0a(a_xx)))**2 for a_xx in ['a06', 'a09' , 'a12', 'a15']}
        elif observable == 't0':
            latt_spacings = {a_xx[1:] : 1/self.interpolate_t0a2(a_xx)/4 for a_xx in ['a06', 'a09' , 'a12', 'a15']}
        latt_spacings['00'] = gv.gvar(0,0)

        for j, xx in enumerate(reversed(latt_spacings)):
            xi = {}
            phys_data = self.phys_point_data
            phys_data['eps2_a'] = latt_spacings[xx]

            min_max = lambda mydict : (gv.mean(np.nanmin([mydict[key] for key in mydict.keys()])), 
                                       gv.mean(np.nanmax([mydict[key] for key in mydict.keys()])))


            if param in ['pi', 'l', 'p']:
                plt.axvline(gv.mean((phys_data['mpi'] / phys_data['lam_chi'])**2), ls='--', label='phys. point')
                min_max = min_max({ens : (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2 for ens in self.ensembles})
                xi['l'] = np.linspace(0.0001, min_max[1])
                x_fit = xi['l']

            elif param in ['k', 's']:
                plt.axvline(gv.mean(((2 *phys_data['mk']**2 - phys_data['mpi']**2) / phys_data['lam_chi']**2)), ls='--', label='Phys point')
                min_max = min_max({ens : (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2 for ens in self.ensembles})
                xi['s'] = np.linspace(min_max[0], min_max[1])
                x_fit = xi['s']

            elif param == 'a':
                plt.axvline(0, label='phys. point', ls='--')
                if self.model_info['eps2a_defn'] == 'w0_original':
                    eps2_a_arr = [self.fit_data[ens]['a/w:orig']**2 / 4 for ens in self.ensembles] 
                elif self.model_info['eps2a_defn'] == 'w0_improved':
                    eps2_a_arr = [self.fit_data[ens]['a/w:impr']**2 / 4 for ens in self.ensembles] 
                elif self.model_info['eps2a_defn'] == 't0_original':
                    eps2_a_arr = [1 / self.fit_data[ens]['t/a^2:orig'] / 4  for ens in self.ensembles] 
                elif self.model_info['eps2a_defn'] == 't0_improved':
                    eps2_a_arr = [1 / self.fit_data[ens]['t/a^2:impr'] / 4  for ens in self.ensembles] 
                elif self.model_info['eps2a_defn'] == 'variable':
                    if observable == 'w0':
                        eps2_a_arr = [self.fit_data[ens]['a/w']**2 / 4 for ens in self.ensembles] 
                    elif observable == 't0':
                        eps2_a_arr = [1 / self.fit_data[ens]['t/a^2'] / 4  for ens in self.ensembles] 
                xi['a'] = np.linspace(0, gv.mean(np.max(eps2_a_arr)))
                x_fit = xi['a']

                
            y_fit = self.fitfcn(posterior=self.posterior[observable], fit_data=phys_data, xi=xi, observable=observable)

            # For LO fits
            if not hasattr(y_fit, "__len__"):
                y_fit = np.repeat(y_fit, len(x_fit))


            pm = lambda g, k : gv.mean(g) + k *gv.sdev(g)
            if xx != '00' and param != 'a':
                plt.fill_between(pm(x_fit, 0), pm(y_fit, -1), pm(y_fit, +1), color=colors[xx], alpha=0.4)
            elif xx == '00' and param != 'w0mO':
                plt.fill_between(pm(x_fit, 0), pm(y_fit, -1), pm(y_fit, +1), facecolor='None', edgecolor='k', alpha=0.6, hatch='/')
            else:
                pass

        for ens in self.ensembles:

            if param in ['pi', 'l', 'p']:
                x[ens] = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_s', 'alpha_s'], observable=observable))
                label = r'$\xi_l$'
                if self.model_info['chiral_cutoff'] == 'Fpi':
                    label = r'$l^2_F = m_\pi^2 / (4 \pi F_\pi)^2$'

            elif param in ['k', 's']:
                x[ens] = (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_l', 'alpha_s'], observable=observable))
                label = r'$\xi_s$'
                if self.model_info['chiral_cutoff'] == 'Fpi':
                    label = r'$s^2_F$'

            elif param == 'a':
                if self.model_info['eps2a_defn'] == 'w0_original':
                    x[ens] = self.fit_data[ens]['a/w:orig']**2 / 4
                    label = r'$\epsilon^2_a = (a / 2 w_{0,\mathrm{orig}})^2$'
                elif self.model_info['eps2a_defn'] == 'w0_improved':
                    x[ens] = self.fit_data[ens]['a/w:impr']**2 / 4
                    label = r'$\epsilon^2_a = (a / 2 w_{0,\mathrm{impr}})^2$'
                elif self.model_info['eps2a_defn'] == 't0_original':
                    x[ens] = 1 / self.fit_data[ens]['t/a^2:orig'] / 4
                    label = r'$\epsilon^2_a = t_{0,\mathrm{orig}} / 4 a^2$'
                elif self.model_info['eps2a_defn'] == 't0_improved':
                    x[ens] = 1 / self.fit_data[ens]['t/a^2:impr'] / 4
                    label = r'$\epsilon^2_a = t_{0,\mathrm{impr}} / 4 a^2$'
                elif self.model_info['eps2a_defn'] == 'variable':
                    if observable == 'w0':
                        x[ens] = self.fit_data[ens]['a/w']**2 / 4
                        label = '$\epsilon^2_a = (a / 2 w_{0,\mathrm{var}})^2$'
                    elif observable == 't0':
                        x[ens] = 1 / self.fit_data[ens]['t/a^2'] / 4
                        label = '$\epsilon^2_a = t_{0,\mathrm{var}} / 4 a^2$'

                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_l', 'xi_s', 'alpha_s'], observable=observable))
                #label = '$\epsilon^2_a = (a / 2 w_0)^2$'

        for ens in reversed(self.ensembles):
            C = gv.evalcov([x[ens], y[ens]])
            eVe, eVa = np.linalg.eig(C)
            for e, v in zip(eVe, eVa.T):
                plt.plot([gv.mean(x[ens])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[ens])],
                        [gv.mean(y[ens])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[ens])],
                        color=colors[ens[1:3]], alpha=1.0, lw=2)
                plt.plot(gv.mean(x[ens]), gv.mean(y[ens]), 
                         color=colors[ens[1:3]], marker='o', mec='w', zorder=3)

        if show_legend:
            if param in ['pi', 'l', 'p']:
                labels = [
                    r'$a_{06}(l_F,s_F^{\rm phys})$',
                    r'$a_{09}(l_F,s_F^{\rm phys})$',
                    r'$a_{12}(l_F,s_F^{\rm phys})$',
                    r'$a_{15}(l_F,s_F^{\rm phys})$'
                ]
            elif param in ['k', 's']:
                labels = [
                    r'$a_{06}(l_F^{\rm phys},s_F)$',
                    r'$a_{09}(l_F^{\rm phys},s_F)$',
                    r'$a_{12}(l_F^{\rm phys},s_F)$',
                    r'$a_{15}(l_F^{\rm phys},s_F)$'
                ]
            elif param == 'a':
                labels = [
                    r'$a_{06}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{09}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{12}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{15}(l_F^{\rm phys},s_F^{\rm phys})$'
                ]
            handles = [
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['06']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['09']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['12']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['15'])
            ]
            plt.legend(handles=handles, labels=labels, ncol=2)#, bbox_to_anchor=(0,1), loc='lower left')

        #plt.grid()
        plt.xlabel(label)

        if observable == 'w0':
            plt.ylabel('$w_0 m_\Omega$')
        elif observable == 't0':
            plt.ylabel('$m_\Omega \sqrt{t_0 / a^2}$')

        if ylim is not None:
            plt.ylim(ylim)

        fig = plt.gcf()
        plt.close()
        return fig


    def plot_interpolation(self, latt_spacing, param='p', observable=None):
        x = {}
        y = {}
        c = {}
            
        #fig = plt.figure(figsize=(6.75, 6.75/1.618034333))
        #plt.axes([0.145,0.145,0.85,0.85])

        colors = {
            'a06' : '#6A5ACD',#'#00FFFF',
            'a09' : '#51a7f9',
            'a12' : '#70bf41',
            'a15' : '#ec5d57',
        }
        ensembles = [ens for ens in self.ensembles if ens[:3] == latt_spacing]

        xi = {}
        phys_data = copy.deepcopy(self.phys_point_data)

        min_max = lambda mydict : (gv.mean(np.nanmin([mydict[key] for key in mydict.keys()])), 
                                    gv.mean(np.nanmax([mydict[key] for key in mydict.keys()])))


        if param in ['pi', 'l', 'p']:
            plt.axvline(gv.mean((phys_data['mpi'] / phys_data['lam_chi'])**2), ls='--', label='phys. point')
            min_max = min_max({ens : (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2 for ens in ensembles})
            xi['l'] = np.linspace(0.0001, min_max[1])
            x_fit = xi['l']

            xlabel = r'$\xi_l$'
            if self.model_info['chiral_cutoff'] == 'Fpi':
                xlabel = r'$l^2_F = m_\pi^2 / (4 \pi F_\pi)^2$'

        elif param in ['k', 's']:
            plt.axvline(gv.mean(((2 *phys_data['mk']**2 - phys_data['mpi']**2) / phys_data['lam_chi']**2)), ls='--', label='Phys point')
            min_max = min_max({ens : (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2 for ens in ensembles})
            xi['s'] = np.linspace(min_max[0], min_max[1])
            x_fit = xi['s']

            xlabel = r'$\xi_s$'
            if self.model_info['chiral_cutoff'] == 'Fpi':
                xlabel = r'$s^2_F$'
        
        y_fit = self.fitfcn_interpolation(fit_data=phys_data, xi=xi, latt_spacing=latt_spacing, observable=observable)

        # For LO fits
        if not hasattr(y_fit, "__len__"):
            y_fit = np.repeat(y_fit, len(x_fit))
        pm = lambda g, k : gv.mean(g) + k *gv.sdev(g)
        plt.fill_between(pm(x_fit, 0), pm(y_fit, -1), pm(y_fit, +1), color=colors[latt_spacing], alpha=0.4)

        
        for ens in ensembles:
            xi = {}
            xi['l'] = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
            xi['s'] = (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2

            if observable == 'w0':
                value_latt = 1 / self.fit_data[ens]['a/w']
            elif observable == 't0':
                value_latt = self.fit_data[ens]['t/a^2']
            value_fit = self.fitfcn_interpolation(latt_spacing=latt_spacing, xi=xi, observable=observable)

            if param in ['pi', 'l', 'p']:
                x[ens] = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
                xi['s'] = (2 *phys_data['mk']**2 - phys_data['mpi']**2) / phys_data['lam_chi']**2
                label = r'$a_{%s}(l_F,s_F^{\rm phys})$'%(ens[1:3])

            elif param in ['k', 's']:
                xi['l'] = (phys_data['mpi'] / phys_data['lam_chi'])**2
                x[ens] = (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2
                label = r'$a_{%s}(l_F^{\rm phys},s_F)$'%(ens[1:3])

            value_fit_phys = self.fitfcn_interpolation(latt_spacing=latt_spacing, xi=xi, observable=observable)

            y[ens] = value_latt + value_fit_phys - value_fit

            plt.errorbar(x=x[ens].mean, y=y[ens].mean, xerr=x[ens].sdev, yerr=y[ens].sdev, marker='o', color=colors[latt_spacing], capsize=0.0, mec='white',  elinewidth=3.0,)


        handles = [
            plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors[latt_spacing])
        ]
        plt.legend(handles=handles, labels=[label])#, bbox_to_anchor=(0,1), loc='lower left')


        plt.xlabel(xlabel)
        if observable == 'w0':
            plt.ylabel('$w_0 / a$')
        elif observable == 't0':
            plt.ylabel('$t_0 / a^2$')

        fig = plt.gcf()
        plt.close()
        return fig



    def plot_parameters(self, xparam, yparam=None):
        if yparam is None:
            yparam = 'w0mO'

        x = {}
        y = {}
        c = {}
            
        colors = {
            '06' : '#6A5ACD',#'#00FFFF',
            '09' : '#51a7f9',
            '12' : '#70bf41',
            '15' : '#ec5d57',
        }
        #c = {abbr : colors[ens[1:3]] for abbr in self.ensembles}

        for ens in self.ensembles:
            for j, param in enumerate([xparam, yparam]):
                if param == 'w0':
                    value = self.fit_data[ens]['mO'] / self.fit_data[ens]['a/w']
                    label = '$w_0 m_\Omega$'
                elif param == 't0':
                    value = np.sqrt(self.fit_data[ens]['t/a^2']) *self.fit_data[ens]['mO']
                    label = '$m_\Omega \sqrt{t/a^2}$'

                elif param == 'l':
                    value = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
                    label= r'$\xi_l$'
                elif param == 's':
                    value = ((2 *self.fit_data[ens]['mk'] - self.fit_data[ens]['mpi'])/ self.fit_data[ens]['lam_chi'])**2
                    label = r'$\xi_s$'
                elif param == 'a':
                    value = self.fit_data[ens]['a/w']**2 / 4
                    label = r'$\xi_a$'
                elif param == 'mpi':
                    value = self.fit_data[ens]['mpi']
                    label = '$am_\pi$'


                if j == 0:
                    x[ens] = value
                    xlabel = label
                elif j == 1:
                    y[ens] = value
                    ylabel = label

        for ens in self.ensembles:
            C = gv.evalcov([x[ens], y[ens]])
            eVe, eVa = np.linalg.eig(C)
            for e, v in zip(eVe, eVa.T):
                plt.plot([gv.mean(x[ens])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[ens])],
                        [gv.mean(y[ens])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[ens])],
                        color=colors[ens[1:3]], alpha=1.0, lw=2)
                plt.plot(gv.mean(x[ens]), gv.mean(y[ens]), 
                         color=colors[ens[1:3]], marker='o', mec='w', zorder=3)


        #handles, labels = plt.gca().get_legend_handles_labels()
        #by_label = dict(zip(labels, handles))
        #plt.legend(by_label.values(), by_label.keys())#ncol=len(by_label), bbox_to_anchor=(0,1), loc='lower left')
        plt.grid()
        plt.xlabel(xlabel, fontsize = 24)
        plt.ylabel(ylabel, fontsize = 24)

        fig = plt.gcf()
        plt.close()
        return fig


    def plot_qq(self, observable):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.fit[observable].qqplot_residuals()

        fig = plt.gcf()
        plt.close()
        return fig


    def shift_latt_to_phys(self, ens=None, phys_params=None, observable=None):
        value_shifted = {}
        for j, ens_j in enumerate(self.ensembles):
            if ens is None or ens_j == ens:
                for mdl in self.fit[observable].y:
                    if mdl.endswith(observable):
                        value_latt = self.fit[observable].y[mdl][j]
                value_fit = self._extrapolate_to_ens(ens_j, observable=observable)
                value_fit_phys = self._extrapolate_to_ens(ens_j, phys_params, observable=observable)

                value_shifted[ens_j] = value_latt + value_fit_phys - value_fit
                if ens is not None:
                    return value_shifted[ens_j]

        return value_shifted


class fitter_dict(dict):
    def __init__(self, fit_data, input_prior, observables):
        self.prior = {}
        self.prior_interpolation = {}

        for obs in observables:
            if obs == 'w0':
                self.prior[obs] = input_prior['w0']
                self.prior_interpolation[obs] = input_prior['w0_interpolation']
            elif obs == 't0':
                self.prior[obs] = input_prior['t0']
                self.prior_interpolation[obs] = input_prior['t0_interpolation']

        self.fit_data = fit_data
        self.observables = observables
        self.ensembles = sorted(list(fit_data))


    def __getitem__(self, model_info):
        if type(model_info) != str:
            key = model_info['name']
        else:
            key = model_info

        if key not in self:
            super().__setitem__(key, self._make_fitter(model_info))
            #print(model_info.values())
            return super().__getitem__(key)
            #return self.__getitem__(model_info)
        else:
            return super().__getitem__(key)


    def __str__(self):
        output = ''
        for model_name in list(self):
            output += '\n---\nModel: '+model_name+'\n\n'
            #output += str(self.__getitem__(key).fit)

            output += 'Parameters:\n'
            my_str = self.__getitem__(model_name).fit.format(pstyle='m')
            for item in my_str.split('\n'):
                for key in self.prior:
                    re = key+' '
                    if re in item:
                        output += item + '\n'

            output += '\n'
            output += self.__getitem__(model_name).fit.format(pstyle=None)

        return output


    def _make_fitter(self, model_info):
        prepped_data = self._make_fit_data(model_info)
        #print(model_info)
        fitter = fit.fitter(
            fit_data=prepped_data, 
            prior=self.prior, 
            prior_interpolation=self.prior_interpolation, 
            model_info=model_info, 
            observables=self.observables, 
            ensemble_mapping=self.ensembles)
        return fitter


    def _make_fit_data(self, model_info):
        prepped_data = {}
        for param in list(self.fit_data[self.ensembles[0]]):
            prepped_data[param] = np.array([
                self.fit_data[ens][param] 
                for ens in self.ensembles
            ])
        if model_info['chiral_cutoff'] == 'mO':
            prepped_data['lam_chi'] = prepped_data['mO']
        elif model_info['chiral_cutoff'] == 'Fpi':
            prepped_data['lam_chi'] = 4 *np.pi *prepped_data['Fpi']
        return prepped_data



    