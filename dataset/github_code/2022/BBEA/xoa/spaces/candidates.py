import os
import numpy as np
import pandas as pd
import random
import copy
import time
import traceback

from xoa.commons.logger import *
from xoa.commons.hp_cfg import HyperparameterConfiguration
from xoa.samplers import *


class CandidateSetGenerator(object):

    def __init__(self, config, spec, use_default=False, verifier=None):
        if type(config) == dict:
            self.config = HyperparameterConfiguration(config)
        else:
            self.config = config
        self.params = self.config.get_param_names()
        self.spec = spec
        self.use_default = use_default
        self.sampler = None
        self.verifier = verifier

        if use_default:
            default = self.config.get_default_vector()
            debug("Default value setting: {}".format(default))
            norm_vec = self.config.convert("arr", "norm_arr", default)
            hpv = self.config.convert("arr", "list", default)
            
            self.grid = np.array([norm_vec])
            self.hpvs = np.array([hpv])            
            self.schemata = np.zeros(self.hpvs.shape)
            self.generations = np.array([0,])
        else:
            self.grid = np.array([])
            self.hpvs = np.array([])            
            self.schemata = np.array([])
            self.generations = np.array([])
        
    def get_param_vectors(self):
        return self.grid

    def get_hp_vectors(self):        
        return self.hpvs

    def get_schemata(self):        
        return self.schemata

    def get_generations(self):
        return self.generations

    def get_sampler(self, spec):

        if 'sample_method' in spec:
            if spec['sample_method'] == 'uniform':
                return UniformRandomSampler(self.config, spec, verifier=self.verifier)
            elif spec['sample_method'] == 'cat_grid':
                return CartesianGridSampler(self.config, spec, verifier=self.verifier)                
            elif spec['sample_method'] == 'latin':
                return LatinHypercubeSampler(self.config, spec, verifier=self.verifier)
            elif spec['sample_method'] == 'local':
                return NeighborSampler(self.config, spec, verifier=self.verifier)
            elif spec['sample_method'] == 'partial':
                return PartialNeighborSampler(self.config, spec, verifier=self.verifier)                
            elif spec['sample_method'] == 'genetic':
                return EvolutionarySampler(self.config, spec, verifier=self.verifier)                                             
            elif spec['sample_method'] != 'Sobol':
                warn("Not supported sampling method: {}. Use of Sobol sequences instead.".format(spec['sample_method']))
        else:
            debug("The sampling will be performed using Sobol sequences.")
        return SobolSequenceSampler(self.config, spec, verifier=self.verifier)

    def generate(self):
        
        try:
            s_t = time.time()
            self.sampler = self.get_sampler(self.spec)
            #debug("Candidates will be generated using {}...".format(g.get_name()))
            grid = self.sampler.generate()
            schemata = self.sampler.get_schemata()
            gen = self.sampler.get_generations()
            self.spec['num_samples'] = self.sampler.get_sample_size()

            if self.use_default:
                self.grid = np.concatenate((self.grid, grid))
                self.schemata = np.concatenate((self.schemata, schemata))
                self.generations = np.concatenate((self.generations, gen))
            else:
                self.grid = grid
                self.schemata = schemata
                self.generations = gen

            self.hpvs = self.config.convert('grid', 'hpv_list', self.grid)
            #debug("{} samples are populated ({:.0f}s)".format(len(self.hpvs), time.time() - s_t))
            #debug("HPV list: {}".format(self.hpvs))

            return self.hpvs

        except Exception as ex:
            warn("Failed to generate candidates: {}".format(ex))
            debug(traceback.format_exc())
            return []
