from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import validators as v
import random as rd

from xoa.commons.logger import *
from xoa.commons.hp_cfg import HyperparameterConfiguration
from .proxy import *



########################################################################
# Surrogate DNN trainer for pre-evaluated benchmark datasets
########################################################################


def create_trainer(space, hp_config, run_config, builder=None, surrogate=None, dataset=None):    
    kwargs = {}

    if isinstance(hp_config, dict):
        hp_config = HyperparameterConfiguration(hp_config)

    if 'benchmark_mode' in run_config and run_config['benchmark_mode'] == True:
        kwargs["surrogate"] = surrogate

    if surrogate != None:
        info("Pre-evaluated dataset: {}".format(surrogate))
        if 'NAS-Bench-101' in surrogate:
            if builder == None:
                builder = get_nas_builder(surrogate, run_config)
            return get_nas_emulator(101, builder, space, run_config)
        elif 'NAS-Bench-201' in surrogate:
            if builder == None:
                builder = get_nas_builder(surrogate, run_config, dataset)
            return get_nas_emulator(201, builder, space, run_config)
        elif 'Benchmark' in surrogate:
            return get_fcnet_emulator(space, run_config, surrogate)
    else:
        raise NotImplementedError("Not supported benchmark dataset!")


