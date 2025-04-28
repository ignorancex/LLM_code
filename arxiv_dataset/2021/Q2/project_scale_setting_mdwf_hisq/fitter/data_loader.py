import numpy as np
import gvar as gv
import datetime
import re
import os
import yaml
import h5py
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

# Set defaults for plots
import matplotlib as mpl
'''
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = (6.75, 6.75/1.618034333)
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True
'''

class data_loader(object):

    def __init__(
            self, 
            collection=None, 
            models=None, 
            excluded_ensembles=None, 
            empirical_priors=None,
            data_file=None, 
            use_charm_reweighting=None,
            use_milc_aw0=None,
            improved_observables=None,
            simultaneous=None):

        self.project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

        # Set options for fits
        # Default values
        if collection is None:
            name = str(datetime.datetime.now())
            for c in [' ', ':', '.', '-']:
                name = name.replace(c, '_')
        else:
            name = collection

        defaults = {
            'name' : name, 
            'models' : [
                'Fpi_n2lo_w0orig',
                'Fpi_n2lo_alphas_w0orig',
                'Fpi_n2lo_alphas_fv_w0orig',
                'Fpi_n2lo_fv_w0orig',
                'Fpi_n2lo_log_w0orig',
                'Fpi_n2lo_log_alphas_w0orig',
                'Fpi_n2lo_log_alphas_fv_w0orig',
                'Fpi_n2lo_log_fv_w0orig',
                'Fpi_n3lo_w0orig',
                'Fpi_n3lo_alphas_w0orig',
                'Fpi_n3lo_alphas_fv_w0orig',
                'Fpi_n3lo_fv_w0orig',
                'Fpi_n3lo_log_log2_w0orig',
                'Fpi_n3lo_log_log2_alphas_w0orig',
                'Fpi_n3lo_log_log2_alphas_fv_w0orig',
                'Fpi_n3lo_log_log2_fv_w0orig',
                'Om_n2lo_w0orig',
                'Om_n2lo_alphas_w0orig',
                'Om_n2lo_alphas_fv_w0orig',
                'Om_n2lo_fv_w0orig',
                'Om_n2lo_log_w0orig',
                'Om_n2lo_log_alphas_w0orig',
                'Om_n2lo_log_alphas_fv_w0orig',
                'Om_n2lo_log_fv_w0orig',
                'Om_n3lo_w0orig',
                'Om_n3lo_alphas_w0orig',
                'Om_n3lo_alphas_fv_w0orig',
                'Om_n3lo_fv_w0orig',
                'Om_n3lo_log_log2_w0orig',
                'Om_n3lo_log_log2_alphas_w0orig',
                'Om_n3lo_log_log2_alphas_fv_w0orig',
                'Om_n3lo_log_log2_fv_w0orig'],
            'excluded_ensembles' : None,
            'data_file' : 'omega_pi_k_spec',
            'empirical_priors' : None,
            'use_charm_reweighting' : False,
            'use_milc_aw0' : False,
            'improved_observables' : True,
            'simultaneous' : False
        }

        # Override values with those from yaml file
        try:
            with open(self.project_path+'/results/'+name+'/settings.yaml') as file:
                output = yaml.safe_load(file)
            defaults.update(output)
        except FileNotFoundError:
            pass

        print(defaults)

        # Override default/yaml values with those from initializing data_loader object
        collection = {
            'name' : name, 
            'models' : models,
            'excluded_ensembles' : excluded_ensembles,
            'data_file' : data_file,
            'empirical_priors' : empirical_priors,
            'use_charm_reweighting' : use_charm_reweighting,
            'use_milc_aw0' : use_milc_aw0,
            'improved_observables' : improved_observables,
            'simultaneous' : simultaneous,
        }
        for key in collection:
            if collection[key] is None:
                collection[key] = defaults[key]

        self.collection = collection
        self._bs_data = None


    @property
    def bs_data(self):
        if self._bs_data is None:

            with h5py.File(self.project_path +'/data/omega_pi_k_spec.h5', 'r') as f:
                ensembles = sorted(list(f.keys()))

            if self.collection['excluded_ensembles'] is not None:
                for ens in self.collection['excluded_ensembles']:
                    ensembles.remove(ens)
            #ensembles.remove('a12m220ms')
            #ensembles.remove('a06m310L')

            #print('load data')
            data = {}
            with h5py.File(self.project_path +'/data/omega_pi_k_spec.h5', 'r') as f:
                for ens in ensembles:
                    data[ens] = {}

                    # Old charm reweighting data
                    #if self.use_charm_reweighting:
                    #    data[ens]['mO'] = f[ens]['m_omega_reweight'][:]
                    #else:
                    #    data[ens]['mO'] = f[ens]['m_omega'][:]
                    data[ens]['mO'] = f[ens]['m_omega'][:]

                    to_gvar = lambda arr : gv.gvar(arr[0], arr[1])
                    if self.collection['improved_observables']:
                        data[ens]['a/w'] = 1.0 / to_gvar(f[ens]['w0a_callat_imp'][:])
                        data[ens]['t/a^2'] = to_gvar(f[ens]['t0aSq_imp'][:])
                    else:
                        data[ens]['a/w'] = 1.0 / to_gvar(f[ens]['w0a_callat'][:])
                        data[ens]['t/a^2'] = to_gvar(f[ens]['t0aSq'][:])


                    if self.collection['use_milc_aw0']:
                        data[ens]['a/w'] = to_gvar(f[ens]['aw0_milc'][:])
                    
                    # For fixed eps2_a = (a / 2 w)^2
                    data[ens]['a/w:orig'] = 1.0 / to_gvar(f[ens]['w0a_callat'][:])
                    data[ens]['a/w:impr'] = 1.0 / to_gvar(f[ens]['w0a_callat_imp'][:])
                    data[ens]['t/a^2:orig'] = to_gvar(f[ens]['t0aSq'][:])
                    data[ens]['t/a^2:impr'] = to_gvar(f[ens]['t0aSq_imp'][:])

                    # arrays
                    for param in ['Fpi', 'mk', 'mpi']:
                        data[ens][param] = f[ens][param][:]

                    # scalars
                    for param in ['L', 'alpha_s']:
                        data[ens][param] = f[ens][param][()]
                        
                if self.collection['use_charm_reweighting']:
                    data['a06m310L']['mO'] = f['a06m310L']['m_omega_reweight'][:]
                    data['a06m310L']['mk'] = f['a06m310L']['mk_reweight'][:]
                    data['a06m310L']['mpi'] = f['a06m310L']['mpi_reweight'][:]


            if self.collection['data_file'] == 'omega_pi_k_spec':
                self._bs_data = data
            else:
                h5_filepath = self.project_path +'/data/' +self.collection['data_file'] +'.h5'
                with h5py.File(h5_filepath, 'r') as f:        
                    for ens in ensembles:
                        # Fix mismatched naming schemes
                        if ens == 'a12m220ms':
                            data['a12m220ms']['mO'] = f['a12m220_ms']['mO'][:]
                        elif ens in list(f.keys()):
                            data[ens]['mO'] = f[ens]['mO'][:]

                self._bs_data = data

        return self._bs_data


    @property
    def gv_data(self):
        gv_data = {}

        for ens in sorted(self.bs_data):
            gv_data[ens] = gv.BufferDict()
            for param in ['mO', 'mpi', 'mk', 'Fpi']:
                gv_data[ens][param] = self.bs_data[ens][param][1:]

            gv_data[ens] = gv.dataset.avg_data(gv_data[ens], bstrap=True) 

            for param in ['mO', 'mpi', 'mk', 'Fpi']: 
                gv_data[ens][param] = gv_data[ens][param] - gv.mean(gv_data[ens][param]) + self.bs_data[ens][param][0]
            
            gv_data[ens]['a/w'] = self.bs_data[ens]['a/w']
            gv_data[ens]['t/a^2'] = self.bs_data[ens]['t/a^2']

            gv_data[ens]['a/w:orig'] = self.bs_data[ens]['a/w:orig']
            gv_data[ens]['a/w:impr'] = self.bs_data[ens]['a/w:impr']
            gv_data[ens]['t/a^2:orig'] = self.bs_data[ens]['t/a^2:orig']
            gv_data[ens]['t/a^2:impr'] = self.bs_data[ens]['t/a^2:impr']

            gv_data[ens]['L'] = gv.gvar(self.bs_data[ens]['L'], self.bs_data[ens]['L'] / 10**6)
            gv_data[ens]['alpha_s'] = gv.gvar(self.bs_data[ens]['alpha_s'], self.bs_data[ens]['alpha_s'] / 10**6)

        return gv_data


    @property
    def phys_point_data(self):
        phys_point_data = {
            'eps2_a' : gv.gvar(0),
            'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.inf),
            'hbarc' : gv.gvar(197.3269804),

            # Flag 2019
            # 'Fpi' : gv.gvar('92.07(57)'),
            # 'mpi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            # 'mk' : gv.gvar('494.2(3)'), # '495.6479(92)'

            # Edinburgh concensus
            'Fpi' : gv.gvar('92.07(57)'),
            'mpi' : gv.gvar('135.00000(1)'), # '138.05638(37)'
            'mk' : gv.gvar( '494.60000(1)'), # '495.6479(92)'
            'Fpi' : gv.gvar('130.50000(1)') / np.sqrt(2),

            #'mss' : gv.gvar('688.5(2.2)'), # Taken from arxiv/1303.1670
            'mO' : gv.gvar('1672.43(32)') # PDG
        }
        return phys_point_data


    def _pickle_fit_info(self, fit_info):
        for obs in list(fit_info):
            model = fit_info[obs]['name']
            if not os.path.exists(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
                os.makedirs(self.project_path +'/results/'+ self.collection['name'] +'/pickles/')
            filename = self.project_path +'/results/'+ self.collection['name'] +'/pickles/'+ obs +'_'+ model +'.p'

            output = {}
            if obs == 'w0':
                output['w0'] = fit_info[obs]['w0']
            elif obs == 't0':
                output['sqrt_t0'] = fit_info[obs]['sqrt_t0']
            elif obs == 't0w0':
                output['sqrt_t0/w0'] = fit_info[obs]['sqrt_t0/w0']

            output['logGBF'] = gv.gvar(fit_info[obs]['logGBF'])
            output['chi2/df'] = gv.gvar(fit_info[obs]['chi2/df'])
            output['Q'] = gv.gvar(fit_info[obs]['Q'])

            for key in fit_info[obs]['prior'].keys():
                output['prior:'+key] = fit_info[obs]['prior'][key]

            for key in fit_info[obs]['posterior'].keys():
                output['posterior:'+key] = fit_info[obs]['posterior'][key]

            for key in fit_info[obs]['phys_point'].keys():
                # gvar can't handle integers -- entries not in correlation matrix
                output['phys_point:'+key] = fit_info[obs]['phys_point'][key]

            for key in fit_info[obs]['error_budget']:
                output['error_budget:'+key] = gv.gvar(fit_info[obs]['error_budget'][key])

            for key in fit_info[obs]:
                if key.startswith('w0/a:') or key.startswith('t0/a2:'):
                    output[key] = fit_info[obs][key]

            gv.dump(output, filename)
        return None


    def _unpickle_fit_info(self, mdl_key):
        filepath = self.project_path +'/results/'+ self.collection['name'] +'/pickles/'+ mdl_key +'.p'
        if os.path.isfile(filepath):
            return gv.load(filepath)
        else:
            return None


    def get_fit_collection(self):
        if os.path.exists(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
            output = {}

            pickled_models = []
            for file in os.listdir(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
                if(file.endswith('.p')):
                    pickled_models.append(file.split('.')[0])

            for mdl_key in pickled_models:
                fit_info_mdl_key = self._unpickle_fit_info(mdl_key=mdl_key)
                model = mdl_key.split('_', 1)[1]

                obs = mdl_key.split('_')[0]
                if obs not in output:
                    output[obs] = {}

                output[obs][model] = {}
                output[obs][model]['name'] = model
                if obs == 'w0':
                    output[obs][model]['w0'] = fit_info_mdl_key['w0']
                    output[obs][model]['w0/a'] = {}
                elif obs == 't0':
                    output[obs][model]['sqrt_t0'] = fit_info_mdl_key['sqrt_t0']
                    output[obs][model]['t0/a2'] = {}
                elif obs == 't0w0':
                    output[obs][model]['sqrt_t0/w0'] = fit_info_mdl_key['sqrt_t0/w0']

                output[obs][model]['logGBF'] = fit_info_mdl_key['logGBF'].mean
                output[obs][model]['chi2/df'] = fit_info_mdl_key['chi2/df'].mean
                output[obs][model]['Q'] = fit_info_mdl_key['Q'].mean
                output[obs][model]['prior'] = {}
                output[obs][model]['posterior'] = {}
                output[obs][model]['phys_point'] = {}
                output[obs][model]['error_budget'] = {}

                for key in fit_info_mdl_key.keys():
                    if key.startswith('prior'):
                        output[obs][model]['prior'][key.split(':')[-1]] = fit_info_mdl_key[key]
                    elif key.startswith('posterior'):
                        output[obs][model]['posterior'][key.split(':')[-1]] = fit_info_mdl_key[key]
                    elif key.startswith('phys_point'):
                        output[obs][model]['phys_point'][key.split(':')[-1]] = fit_info_mdl_key[key]
                    elif key.startswith('error_budget'):
                        output[obs][model]['error_budget'][key.split(':')[-1]] = fit_info_mdl_key[key].mean

                    elif key.startswith('w0/a') and obs == 'w0':
                        output[obs][model]['w0/a'][key.split(':')[-1]] = fit_info_mdl_key[key]
                    elif key.startswith('t0/a2') and obs == 't0':
                        output[obs][model]['t0/a2'][key.split(':')[-1]] = fit_info_mdl_key[key]



            return output


    def get_model_info_from_name(self, name):
        model_info = {}
        model_info['name'] = name

        if name.startswith('Fpi'):
            model_info['chiral_cutoff'] = 'Fpi'
        elif name.startswith('Om'):
            model_info['chiral_cutoff'] = 'mO'
        

        # Order
        if '_nlo' in name:
            model_info['order'] = 'nlo'
            model_info['latt_ct'] = 'nlo'
        elif '_n2lo' in name:
            model_info['order'] = 'n2lo'
            model_info['latt_ct'] = 'n2lo'
        elif '_n3lo' in name:
            model_info['order'] = 'n3lo'
            model_info['latt_ct'] = 'n3lo'
        else:
            model_info['order'] = 'lo'
            model_info['latt_ct'] = 'lo'

        # Order for latt terms
        if '_a2' in name:
            model_info['latt_ct'] = 'nlo'
        elif '_a4' in name:
            model_info['latt_ct'] = 'n2lo'
        elif '_a6' in name:
            model_info['latt_ct'] = 'n3lo'

        # Include other corrections
        model_info['include_log'] = bool('_log_' in name or name.endswith('log'))
        model_info['include_log2'] = bool('_log2' in name)
        model_info['include_fv'] = bool('_fv' in name)
        model_info['include_alphas'] = bool('_alphas' in name)

        model_info['exclude'] = []

        # Definition of eps2_a
        if '_w0orig' in name:
            model_info['eps2a_defn'] = 'w0_original'
        elif '_w0impr' in name:
            model_info['eps2a_defn'] = 'w0_improved'
        elif '_t0orig' in name:
            model_info['eps2a_defn'] = 't0_original'
        elif '_t0impr' in name:
            model_info['eps2a_defn'] = 't0_improved'
        elif '_variable' in name:
            model_info['eps2a_defn'] = 'variable'
        else:
            model_info['eps2a_defn'] = None

        model_info['name']  = self.get_model_name_from_info(model_info)
        return model_info


    def get_model_name_from_info(self, model_info):
        if model_info['chiral_cutoff'] == 'mO':
            name = 'Om'
        elif model_info['chiral_cutoff'] == 'Fpi':
            name = 'Fpi'

        if model_info['order'] == 'lo':
            name += '_lo'
        elif model_info['order'] == 'nlo': 
            name += '_nlo'
        elif model_info['order'] == 'n2lo': 
            name += '_n2lo'
        elif model_info['order'] == 'n3lo': 
            name += '_n3lo'

        if model_info['include_log']:
            name += '_log'
        if model_info['include_log2']:
            name += '_log2'

        if 'latt_ct' in model_info:
            if model_info['latt_ct'] != model_info['order']:
                if model_info['latt_ct'] == 'nlo':
                    name += '_a2'
                elif model_info['latt_ct'] == 'n2lo':
                    name += '_a4'
                elif model_info['latt_ct'] == 'n3lo':
                    name += '_a6'

        if model_info['include_alphas']:
            name += '_alphas'

        if model_info['include_fv']:
            name += '_fv'

        # Definition of eps2_a
        if model_info['eps2a_defn'] == 'w0_original':
            name += '_w0orig'
        elif model_info['eps2a_defn'] == 'w0_improved':
            name += '_w0impr'
        elif model_info['eps2a_defn'] == 't0_original':
            name += '_t0orig'
        elif model_info['eps2a_defn'] == 't0_improved':
            name += '_t0impr'
        elif model_info['eps2a_defn'] == 'variable':
            name += '_variable'

        return name


    def get_prior(self, model=None, model_info=None, default=False):

        def get_default_prior(model, fit_type):
            with open(self.project_path+'/data/default_priors.yaml') as file:
                prior_file = yaml.safe_load(file)

                model_info = self.get_model_info_from_name(model)
                chiral_cutoff = model_info['chiral_cutoff']

                if (fit_type == 'w0') and (chiral_cutoff == 'Fpi'):
                    key = 'w0_default_Fpi'
                elif (fit_type == 'w0') and (chiral_cutoff == 'mO'):
                    key = 'w0_default_mO'

                elif (fit_type == 't0') and (chiral_cutoff == 'Fpi'):
                    key = 't0_default_Fpi'
                elif (fit_type == 't0') and (chiral_cutoff == 'mO'):
                    key = 't0_default_mO'

                elif fit_type == 'w0_interpolation':
                    key = 'w0_interpolation'
                elif fit_type == 't0_interpolation':
                    key = 't0_interpolation'

                else:
                    print('Error: requested prior does not exist!')
                    return None

                output = gv.gvar(prior_file[key])
                self.save_prior(mdl_key=key, prior=output)

                return output


        if (model is None) and (model_info is not None):
            model = self.get_model_name_from_info(model_info)

        filepath = self.project_path+'/results/'+self.collection['name']+'/priors.yaml'

        if os.path.isfile(filepath):
            with open(filepath) as file:
                prior_file = yaml.safe_load(file)
                chiral_cutoff = self.get_model_info_from_name(model)['chiral_cutoff']

                prior = {}

                # Load prior for w0 extrapolation
                if 'w0_'+model in prior_file:
                    prior['w0'] = gv.gvar(prior_file['w0_'+model])
                elif ('w0_default_Fpi' in prior_file) and (chiral_cutoff == 'Fpi'):
                    prior['w0'] = gv.gvar(prior_file['w0_default_Fpi'])
                elif ('w0_default_mO' in prior_file) and (chiral_cutoff == 'mO'):
                    prior['w0'] = gv.gvar(prior_file['w0_default_mO'])
                else:
                    prior['w0'] = get_default_prior(model=model, fit_type='w0')

                # Load prior for t0 extrapolation
                if 't0_'+model in prior_file:
                    prior['t0'] = gv.gvar(prior_file['t0_'+model])
                elif ('t0_default_Fpi' in prior_file) and (chiral_cutoff == 'Fpi'):
                    prior['t0'] = gv.gvar(prior_file['t0_default_Fpi'])
                elif ('t0_default_mO' in prior_file) and (chiral_cutoff == 'mO'):
                    prior['t0'] = gv.gvar(prior_file['t0_default_mO'])
                else:
                    prior['t0'] = get_default_prior(model=model, fit_type='t0')


                if 'w0_interpolation' in prior_file:
                    prior['w0_interpolation'] = gv.gvar(prior_file['w0_interpolation'])
                else:
                    prior['w0_interpolation'] = get_default_prior(model=model, fit_type='w0_interpolation')

                if 't0_interpolation' in prior_file:
                    prior['t0_interpolation'] = gv.gvar(prior_file['t0_interpolation'])
                else:
                    prior['t0_interpolation'] = get_default_prior(model=model, fit_type='t0_interpolation')

                return prior
                
        else:
            prior = {}
            prior['w0'] = get_default_prior(model=model, fit_type='w0')
            prior['t0'] = get_default_prior(model=model, fit_type='w0')
            prior['w0_interpolation'] = get_default_prior(model=model, fit_type='w0_interpolation')
            prior['t0_interpolation'] = get_default_prior(model=model, fit_type='t0_interpolation')

            return prior


    def plot_qq(self, ens, param):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.probplot(self.bs_data[ens][param], dist='norm', plot=ax)

        fig = plt.gcf()
        plt.close()

        return fig


    def save_fit_info(self, fit_info):
        self._pickle_fit_info(fit_info)
        return None


    def save_results_summary(self, output_string):

        # converts a dictionary of dictionaries to a markdown table
        def dict_dict_to_table(dict_dict, column0='', exclude=[], sort_key=None):
            header = []
            for item in dict_dict:
                header = np.union1d(header, list(dict_dict[item]))

            header = sorted(header, key=sort_key)
            for key in exclude:
                if key in header:
                    header.remove(key)
                

            table = '| %s |' %(column0)
            for key in header:
                table += ' %s |' %(key)
            table += '\n'

            table += '| --- |'
            for key in header:
                table += ' --- |'
            table += '\n'

            for item in dict_dict:
                table += '| %s |' %(item)
                for key in header:
                    if key in dict_dict[item]:
                        table += ' %s |' %(str(dict_dict[item][key]))
                    else:
                        table += '  |'
                table += '\n'

            return table

        filename = self.project_path +'/results/'+ self.collection['name'] + '/README.md'

        models = self.collection['models']
        filepath = self.project_path +'/results/'+ self.collection['name'] +'/pickles/'
        observables = np.unique([file.split('.')[0].split('_')[0] for file in os.listdir(filepath)])

        prior_table = ''
        for obs in observables:
            if obs in ['w0', 't0']:
                prior_table += 'observable: %s \n' %(obs)
                #priors_models = {mdl : self.get_prior(model=mdl)[obs] for mdl in models}
                priors_models = {}
                priors_models['default_Fpi'] = self.get_prior(model='Fpi_n3lo_fv', default=True)[obs]
                priors_models['default_Om'] = self.get_prior(model='Om_n3lo_fv', default=True)[obs]

                length = lambda x : len(x.split('_')[1]) if len(x.split('_')) > 1 else 0
                prior_table += dict_dict_to_table(priors_models, column0='model', sort_key=length)
                prior_table += '\n\n'

        for obs in observables:
            if obs in ['w0', 't0']:
                priors_models = {obs+'_interpolation' : self.get_prior(model=models[0])[obs+'_interpolation']}
                prior_table += dict_dict_to_table(priors_models, column0='model', sort_key=length)
                prior_table += '\n\n'

        # Generate table for models    
        #models = self.collection['models']
        #models_descriptions = {mdl : self.get_model_info_from_name(mdl) for mdl in models}
        #models_table = dict_dict_to_table(models_descriptions, exclude=['name'], column0='model')

        # Generate table for input params
        inputs_table = dict_dict_to_table(self.gv_data, column0='ens')

        #output_string  = '### Priors\n' +prior_table+ '\n### Model List\n' +models_table+ '\n### Inputs\n' +inputs_table+ '\n' +output_string
        output_string  = '### Priors\n' +prior_table+ '\n### Inputs\n' +inputs_table+ '\n' +output_string

        #return output_string

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                file_content = file.read()
            if '## Autogenerated' in file_content:
                file_content = re.sub(r'## Autogenerated\s(.*\s*)*\Z',
                                      '## Autogenerated\n'+output_string,
                                      file_content)
                with open(filename, 'w') as file:
                    print(file_content)
                    file.write(file_content)

                return None

        # else
        with open(filename, 'a+') as file:
            file_content = '\n## Autogenerated\n'
            file_content += output_string
            file.write(file_content)

        return None


    def save_fig(self, fig=None, output_filename=None, filetype='svg'):

        if fig is None:
            print('Nothing here!')
            return None

        if output_filename is None:
            if not os.path.exists(os.path.normpath(self.project_path+'/tmp/')):
                os.makedirs(os.path.normpath(self.project_path+'/tmp/'))
            output_file = os.path.normpath(self.project_path+'/tmp/temp.'+filetype)
        else:
            output_file = os.path.normpath(self.project_path+'/results/'+self.collection['name']+'/'+output_filename+'.'+filetype)

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        fig.savefig(output_file, bbox_inches='tight', transparent=True)
        return None


    def save_settings(self):
        filepath = self.project_path+'/results/'+self.collection['name']+'/settings.yaml'
        Path(self.project_path+'/results/'+self.collection['name']).mkdir(parents=True, exist_ok=True)

        output = self.collection.copy()
        del(output['name'])
        with open(filepath, 'w') as file:
            yaml.dump(output, file)

        return None


    def save_prior(self, mdl_key, prior):
        filepath = self.project_path+'/results/'+self.collection['name']+'/priors.yaml'
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                yaml_content = yaml.safe_load(file) or {}
                yaml_content.update({mdl_key : {str(key) : str(prior[key]) for key in prior}})
        else:
            yaml_content = {mdl_key : {str(key) : str(prior[key]) for key in prior}}

        with open(filepath, 'w') as file:
            yaml.safe_dump(yaml_content, file)

        return None
