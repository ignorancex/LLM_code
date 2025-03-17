import time
import random
import traceback

from xoa.commons.logger import * 
from xoa.spaces.candidates import CandidateSetGenerator
from xoa.spaces.adaptive_space import AdaptiveConfigurationSpace

##########################################################################
# Search space operations

def create_space_from_table(surrogate_name, space_setting={}):
    try:
        from lookup.dnnbench import get_lookup_loader
        from xoa.spaces.preeval_space import SurrogatesSpace
        grid_order = None
        if 'order' in space_setting:
            grid_order = space_setting['order']    
        l = get_lookup_loader(surrogate_name, grid_order=grid_order)
        s = SurrogatesSpace(l, space_setting)
        debug("Surrogate space created from {} tabular benchmark".format(surrogate_name))
        return s
    except Exception as ex:
        error("Preevaluated dataset {} is not suppported!".format(surrogate_name))
        debug(traceback.format_exc())


def create_search_space(hp_cfg_dict, space_setting={}, hpv_list=None, verifier=None):
    
    start_time = time.time()
    if not 'num_samples' in space_setting:
        space_setting['num_samples'] = 20000

    if not 'sample_method' in space_setting:
        space_setting['sample_method'] = 'Sobol'

    if not 'seed' in space_setting:
        space_setting['seed']  = 1 # basic seed number
    elif space_setting["seed"] == 'random':
            max_bound = 100000 # XXX: tentatively designed bound
            space_setting["seed"] = random.randint(1, max_bound)

    if not 'prior_history' in space_setting:
        space_setting['prior_history'] = None

    if 'dataset' in hp_cfg_dict and 'model' in hp_cfg_dict:
        prefix = "{}-{}".format(hp_cfg_dict['dataset'], hp_cfg_dict['model'])
    else:
        prefix = "{}-{}".format(space_setting['sample_method'], space_setting['seed'])
    name = "{}-{}".format(prefix, time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))

    with_default = False
    if 'starts_from_default' in space_setting:
        with_default = space_setting['starts_from_default']

    if hpv_list == None:    
        hvg = CandidateSetGenerator(hp_cfg_dict, space_setting, 
                                            use_default=with_default, verifier=verifier)
        hpv_list = hvg.generate()

    s = AdaptiveConfigurationSpace(name, hp_cfg_dict, hpv_list, space_setting=space_setting)
    end_time = time.time() - start_time
    debug("Search space {} has been created ({:.0f}s)".format(name, end_time))
    
    return s


def connect_remote_space(space_url, cred):
    try:
        from xoa.spaces.remote_space import RemoteParameterSpace
        debug("Connecting remote space: {}".format(space_url))
        return RemoteParameterSpace(space_url, cred)
    except Exception as ex:
        warn("Fail to connect remote search space: {}".format(ex))
        return None  
