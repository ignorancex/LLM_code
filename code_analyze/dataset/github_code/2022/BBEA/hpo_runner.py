from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
\

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # XXX:cleansing console message

import numpy as np
from xoa.commons import *
from xoa.spaces import *

from optimizers import create_emulator, create_runner
from trainers import create_trainer, create_verifier 

ALL_OPT_MODELS = ['NONE', 
                  'WGP', 'WGP32', 'WGPL32',  # Input wrapped GP using GPy 
                  'NWGP', 'NWGP32', 'NWGPL32', # No input wrapped GP using GPy 
                  'WGP-PTE', 'WGPL32-PTE', # Input wrapped and output transformation
                  # speed-up version (use of a few MC sampling)
                  'WGPL32m', 'WGPL32m-HLE', 'WGPL32m-PTE', 
                  'NWGPL32m', 'NWGPL32m-HLE', 'NWGPL32m-PTE',                    
                
                  'RF', 'RF-LE', 'RF-HLE', 'RF-PTE'    
                  ]

ACQ_FUNCS = ['RANDOM', 'EI', 'PI', 'UCB', # typical acquisition function types
             ]

DIV_SPECS = ['SEQ', 'RANDOM', # naive diversification strategies 
             ]

def load_search_space(space_path, max_size=None):
    hpv_list = []
    if not 'search_space.npz' in space_path:
        if space_path.endswith('/'):
            space_path = space_path + 'search_space.npz'
        else:
            space_path = space_path + '/search_space.npz'
    
    try:
        hist_f = np.load(space_path, allow_pickle=True)
        hpv_dict = hist_f['hpv'].tolist() # XXX: np.array to python dict
        for k in hpv_dict.keys():
            hpv_list.append(hpv_dict[k])
            if max_size != None and len(hpv_list) > max_size:
                break
    
    except Exception as ex:
        warn("Loading hyperparameters from prior space failed: {}".format(ex))

    debug("Number of configurations: {}".format(len(hpv_list)))
    return hpv_list


def run(args, save=True):

    run_cfg = args['run_config']
    space_spec = run_cfg['search_space']
    hp_cfg = args['hp_config']
    hp_cfg_dict = hp_cfg.get_dict()
    hp_cfg_name = args['hp_cfg_name']
    s = None
    m = None        
    result = []
            
    if not 'use_lookup' in run_cfg or not run_cfg['use_lookup']:
        v = None        
        surrogate = None
        hpv_list = None
        dataset = None
        if 'dataset' in hp_cfg_dict:
            dataset = hp_cfg_dict['dataset']

        if 'config' in hp_cfg_dict:
            if 'surrogate' in hp_cfg_dict['config']:
                surrogate = hp_cfg_dict['config']['surrogate']                
                if 'verification' in space_spec and space_spec['verification']:
                    info("Pre-verification candidates on {}".format(surrogate))                                           
                    v = create_verifier(surrogate, run_cfg)
        
        if 'prior_space' in space_spec:
            hpv_list = load_search_space(space_spec['prior_space'])
            i_start = 0
            if 'seed' in space_spec:
                i_start = space_spec['seed']

            if 'num_samples' in space_spec:
                if len(hpv_list) > space_spec['num_samples'] + i_start:
                    i_end = space_spec['num_samples'] + i_start
                    hpv_list = hpv_list[i_start:i_end]
                else:
                    error("Invalid space setting! # of config in the prior space is less than the spec.")
                
            info("{} predefined configruations are from {}".format(len(hpv_list), space_spec['prior_space']))

        s = create_search_space(hp_cfg_dict, space_spec, hpv_list=hpv_list, verifier=v)
        t = create_trainer(s, hp_cfg, run_cfg, builder=v, surrogate=surrogate, dataset=dataset)
        
        debug("Search space is created as {}".format(space_spec))
        m = create_runner(s, t, 
                        args['exp_crt'], 
                        args['exp_goal'], 
                        args['exp_time'],
                        goal_metric= args['goal_metric'],
                        num_resume=args['rerun'],
                        save_internal=args['save_internal'],
                        run_config=run_cfg,
                        hp_config=hp_cfg
                        )
    else:
        s = create_space_from_table(hp_cfg_name, space_spec)
        if s == None:
            raise ValueError("Loading tabular dataset failed: {}".format(hp_cfg_name))

        debug("Surrogate space is constructed using pre-evaluated configurations: {}.".format(hp_cfg_name))            

        m = create_emulator(s, 
                            args['exp_crt'], 
                            args['exp_goal'], 
                            args['exp_time'],
                            goal_metric=args['goal_metric'],
                            num_resume=args['rerun'],
                            save_internal=args['save_internal'],
                            run_config=run_cfg)


    if not args['mode'] in ALL_OPT_MODELS + ['DIV']:
        raise ValueError('unsupported mode: {}'.format(args['mode']))

    if not args['spec'] in ACQ_FUNCS + DIV_SPECS:
        raise ValueError('unsupported spec: {}'.format(args['spec']))

    result = m.play(args['mode'], args['spec'], args['num_trials'], 
                    save=save)
    #m.print_best(result)
    
    if args['save_space'] == True:
        if s != None:
            # save final search space
            s.save() 


