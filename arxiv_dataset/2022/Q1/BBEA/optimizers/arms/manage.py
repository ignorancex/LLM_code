from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import random
import json

from xoa.commons import *
from optimizers.choosers import load_chooser
from optimizers.arms.select import ArmSelector


class SurrogateModelManager(object):

    def __init__(self, space, config, verifier=None):
        self.config = config
        self.search_space = space
        
        self.selector = None
        self.mode = None
        self.spec = None

        self.models = self.configure(config, verifier)

    def get_default_models(self, config):
        models = []
        if 'arms' in config:
            models = list(set([a['model'] for a in config['arms']]))
        
        # Add default modeling methods
        if not 'NONE' in models:
            models.append('NONE')

        #if not 'GP' in models:
        #    models.append('GP')

        #if not 'RF' in models:
        #    models.append('RF')

        #if not 'TPE' in models:
        #    models.append('TPE')

        #if not 'EVO' in models:
        #    models.append('EVO')

        # experimental option
        #if not 'DNGO' in opts:
        #    opts.append('DNGO')
        #if not 'GN' in opts:
        #    opts.append('GN')
        #if not 'BNS' in opts:
        #    opts.append('BNS')

        return models

    def configure(self, config, verifier=None):        
        choosers = {}
        models = self.get_default_models(config)

        # Global response shaping setting 
        
        space_config = {}
        if 'search_space' in config:
            space_config = config['search_space']            
            

        # Local response shaping setting
        for m in models:
            options = ''
            model_type = None #XXX:Not determined yet

            # response shaping options
            shaping_options = ''
            if 'shaping_func' in space_config:
                shaping_options += ',shaping_func={}'.format(space_config['shaping_func'])
            if 'shaping_alpha' in space_config:
                shaping_options += ',alpha={}'.format(space_config['shaping_alpha'])
            if shaping_options != '': 
                debug("Grobal shaping option: {}".format(shaping_options))
            
            if '-LE' in m:
                shaping_options = ',shaping_func=log_err'
            elif '-HLE' in m:
                if 'shaping_alpha' in space_config:
                    shaping_options = ',shaping_func=hybrid_log,alpha={}'.format(space_config['shaping_alpha'])
                else:
                    shaping_options = ',shaping_func=hybrid_log'
            elif '-PTE' in m:
                shaping_options = ',shaping_func=power_transform'
            elif '-NLE' in m:
                shaping_options = shaping_options + ',shaping_func=no_shaping'

            
            if 'GP' in m:
                # max sample size hyperparameters
                gp_options = 'max_obs=200'
                
                # parse GP model hyperparameter options
                if 'n300' in m:
                    gp_options = 'max_obs=300'
                elif 'n400' in m:
                    gp_options = 'max_obs=400'                
                elif 'n500' in m:
                    gp_options = 'max_obs=500'
                
                # MCMC sampling 
                if 'm' in m:
                    gp_options += ',mcmc_iters=1'
                    if 'm5' in m:
                        gp_options += ',mcmc_iters=5' 

                if 'WGP' in m:
                    model_type = 'WGP'

                    if 'WGPL32' in m:
                        gp_options = gp_options + ",covar=Linear+Matern32"
                    elif 'WGP32' in m:
                        gp_options = gp_options + ",covar=Matern32"
                    else:
                        gp_options = gp_options + ",covar=Matern52"
                    
                    if 'NWGP' in m:
                        gp_options += ',warp=0' # No input warping
                elif 'GPDN' in m:
                    model_type = 'GN'
                else:
                    # Snoek's model implementation
                    model_type = 'GP' 
                    
                    if 'a0' == m:
                        gp_options = gp_options + ',trade_off=0.01,v=0.2'
                    elif 'a1' in m:
                        gp_options = gp_options + ',trade_off=0.01,v=0.2'
                    elif 'a2' in m:
                        gp_options = gp_options + ',trade_off=1.0,v=0.1'
                    
                debug("{} model setting: {}".format(model_type, gp_options))
                options = gp_options + shaping_options
            

            elif 'RF' in m:
                options = "max_features=auto" + shaping_options
                model_type = 'RF'                            
            elif 'DNGO' in m:
                options = "n_hypers=0" + shaping_options
                model_type = 'DNGO' 
            elif 'BNS' in m:
                options = "num_ensemble=5" + shaping_options
                model_type = 'BNS' 
            elif 'TPE' in m:
                model_type = 'TPE'
                options = ''
            elif 'HEBO' in m:
                model_type = 'HEBO'
                if not 'shaping_func' in shaping_options:
                    options = 'shaping_func=power_transform' + shaping_options
                else:
                    options = shaping_options
            elif 'EVO' in m:
                evo_options = ''
                if 'n_init_pop' in space_config:
                    evo_options += ',n_init_pop={}'.format(space_config['n_init_pop'])
                if 'n_parent' in space_config:
                    n_p = space_config['n_parent']
                    try:
                        n_parent = int(n_p)
                        evo_options += ',n_parent={}'.format(n_parent)
                    except Exception as ex:
                        warn("sample size convert error: {}".format(m))                
                
                model_type = 'EVO'
                options = 'acq_func=RE' + evo_options
                
            elif 'NONE' in m:
                model_type = 'NONE'
                options = 'acq_func=RANDOM'  
            
            if not model_type:
                error("Not supported model: {}".format(m))

            choosers[m] = load_chooser(self.search_space, model_type, options, verifier=verifier)

        return choosers

    def reset(self, mode, spec):
        self.mode = mode
        self.spec = spec
        
        if mode == 'DIV' or mode == 'ADA':            
            debug("Multi-armed bandit mode.")
            self.selector = ArmSelector(spec, self.config, self.search_space, self.models)
        elif 'search_space' in self.config:
            space_spec = self.config['search_space']
            if 'evolve_div' in space_spec:
                debug("Single-armed bandit mode with diversified evolution.")
                self.selector = ArmSelector('RANDOM', self.config, self.search_space, self.models)
            elif 'resample' in space_spec:
                debug("Single-armed bandit mode with resampling candidates.")
                self.selector = ArmSelector('SEQ', self.config, self.search_space, self.models)
            else:
                debug("Single-armed bandit mode.")
                self.selector = None                               
        else:            
            debug("Single-armed bandit mode.")
            self.selector = None
        
        self.cur_index = 0

        for m in self.models:
            self.models[m].reset()

    def get_arm_list(self):
        if self.selector:
            return self.selector.arms
        else:
            return [ {"model": self.mode, "acq_func": self.spec } ]

    def get_models(self):
        return self.models

    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError("Invalid model name: {} in {}".format(model_name, self.models.keys()))

    def get_others_best(self, min_epoch=0):
        others = self.selector.get_other_arms()
        others_selection = []
        tpe = None 
        for s in others:            
            model_name = s['model']
            if model_name != 'TPE' or model_name != 'HEBO' or model_name != 'EVO':
                acq_func = s['acq_func']            
                next_index, _ = self.models[model_name].next(acq_func, min_epoch)
                debug("Selected next candidate by {}({}) is {}".format(model_name, acq_func, next_index))
                others_selection.append(next_index)
            else:
                tpe = s               
        
        if tpe != None:
            model_name = tpe['model']
            acq_func = tpe['acq_func']
            next_index, _ = self.models[model_name].next(acq_func, min_epoch)
            debug("Selected next candidate by {}({}) is {}".format(model_name, acq_func, next_index))
            others_selection.append(next_index)            

        return others_selection # XXX: selection can be duplicated

    def get_selected_model(self, index):
        self.cur_index = index
        
        if self.mode == 'DIV' or self.mode == 'ADA':
            return self.selector.get_arm(index)
        else:
            return self.mode, self.spec 

    def get_stats(self):
        if self.selector != None:
            return self.selector.get_stats()
        else:
            return 0, self.cur_index     

    def feedback(self, index, value, opt, metric):
        if value == None:
            # XXX: when no response returned
            return
        
        if metric != 'accuracy':
            value = -1.0 * value # XXX:make 

        if self.mode == 'DIV' or self.mode == 'ADA':
            self.selector.update_reward(index, value, opt)

