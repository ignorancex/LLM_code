from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fnmatch
import argparse

from xoa.commons import *
from xoa.commons.logger import *
from hpo_runner import run

# Global variables
HP_CONF_PATH = './hp_conf/'
ALG_CONF_PATH = './opt_conf/'
DNN_BENCHMARKS = ['CIFAR10-ResNet', 'CIFAR10-VGG', 'CIFAR100-VGG', 'MNIST-LeNet1', 'MNIST-LeNet2', 'PTB-LSTM']

def validate_args(args):

    valid = {}

    if args.debug_mode:          
        set_log_level('debug')

    hp_cfg_path = args.hp_config_dir
    hp_cfg_name = args.hp_config

    if hp_cfg_name.endswith('.json'):
        hp_cfg_name = hp_cfg_name[:-5]

    # hp_config is a json file
    hp_cfg_path = hp_cfg_path + hp_cfg_name + '.json'    
    hp_cfg = read_hyperparam_config(hp_cfg_path)
    if hp_cfg is None:
        raise ValueError('Invaild hyperparameter configuration: {}'.format(hp_cfg_path))
    
    bbea_cfg = 'BBEA'
    run_cfg = read_config(bbea_cfg, path=ALG_CONF_PATH)
    if not validate_config(run_cfg):
        raise ValueError('Invaild algorithm configuration: {}'.format(bbea_cfg))  
    else:
        info("Algorithm configuration: {}".format(bbea_cfg))

    if not 'title' in run_cfg:
        run_cfg['title'] = bbea_cfg

    if not "early_term_rule" in run_cfg:
        run_cfg["early_term_rule"] = args.early_term_rule

    if args.num_trials > 1:
        run_cfg['benchmark_mode'] = True
        run_cfg['report_type'] = 'validation'
    else:
        run_cfg['benchmark_mode'] = False
        run_cfg['report_type'] = 'test'

    if 'NAS' in hp_cfg_name or 'HPO' in hp_cfg_name:
        run_cfg['use_lookup'] = False
    elif hp_cfg_name in DNN_BENCHMARKS:
        # For use of DNN benchmark tasks
        run_cfg['use_lookup'] = True
        run_cfg['report_type'] = 'test'
    
    for attr, value in vars(args).items():
        # override argument options 
        if attr in run_cfg:
            valid[str(attr)] = run_cfg[str(attr)]
        else:
            valid[str(attr)] = value  

    space_spec = {}
    if 'search_space' in run_cfg:
        space_spec = run_cfg['search_space']
    
    # Search space initialization
    if 'NAS101' in hp_cfg_name:
        space_spec['sample_method'] = 'Sobol'
        space_spec['num_samples'] = 100000
        space_spec['verification'] = True
    elif 'NAS201' in hp_cfg_name or 'MLP' in hp_cfg_name:
        space_spec['sample_method'] = 'cat_grid'
    
    if not 'order' in space_spec:
        space_spec['order'] = None
    if not 'num_samples' in space_spec:
        if not 'sample_method' in space_spec:
            space_spec['num_samples'] = 20000
    if not 'seed' in space_spec:
        space_spec['seed'] = 1
    if not 'prior_history' in space_spec:
        space_spec['prior_history'] = None
    
    valid['search_space'] = space_spec
    valid['hp_cfg_name'] = hp_cfg_name
    valid['hp_config'] = hp_cfg
    valid['run_config'] = run_cfg

    valid['mode'] = run_cfg['mode']
    valid['spec'] = run_cfg['spec']
    valid['num_trials'] = args.num_trials
    
    # Default target settings
    valid['exp_crt'] = 'TIME'
    valid['exp_goal'] = 0.0
    valid['goal_metric'] = 'error'

    # Disable internal options
    valid['save_internal'] = False
    valid['save_space'] = False
    valid['rerun'] = 0

    return valid


def main(args):
    try:
        run_args = validate_args(args)
        run(run_args)
          
    except Exception as ex:
        error("Runtime exception: {}".format(ex))
        error(traceback.format_exc())  


if __name__ == "__main__":
    
    default_early_term_rule = 'DecaTercet' # 'PentaTercet' or 'TetraTercet' can be used

    parser = argparse.ArgumentParser()

    # Debug option
    parser.add_argument('-dm', '--debug_mode', action='store_true',
                        help='Set debugging mode.')                        

    # Benchmark mode options
    parser.add_argument('-nt', '--num_trials', type=int, default=1,
                        help='The total number of repeated runs. The default setting is "1".')


    # Early termination rule
    parser.add_argument('-etr', '--early_term_rule', default=default_early_term_rule, type=str,
                        help='Early termination rule.\nA name of compound rule, such as "PentaTercet" or "DecaTercet", can be used.\nThe default setting is {}.'.format(default_early_term_rule))


    # Search space configurations
    parser.add_argument('-hd', '--hp_config_dir', default=HP_CONF_PATH, type=str,
                        help='Hyperparameter space configuration directory.\n'+\
                        'The default setting is "{}"'.format(HP_CONF_PATH))                        

    # Mandatory arguments
    parser.add_argument('hp_config', type=str, help='Hyperparameter space configuration file name.')    
    parser.add_argument('exp_time', type=str, help='The maximum runtime when an HPO run expires.')


    args = parser.parse_args()

    main(args)
    
