import numpy as np
import random
import traceback
import copy

from xoa.commons.logger import *
from xoa.commons.converter import OneHotVectorTransformer

from .proto import SamplerProtype

class PartialNeighborSampler(SamplerProtype):

    def __init__(self, config, spec, verifier=None):        
        self.name = 'Partial local sampling'
               
        super(PartialNeighborSampler, self).__init__(config, spec, verifier)
        
        self.candidate = self.validate(spec['female']) # spec['female'] is list type
        if self.candidate == False:
            raise ValueError("Invalid best candidate!: {}".format(spec['female']))
        
        if 'ls_stdev' in spec:
            self.sd = spec['ls_stdev']
        else:
            self.sd = 0.5
        
        if not 'schema' in spec:
            self.schema = np.zeros(self.num_dim)
        else:
            self.schema = spec['schema'] 

        if 'generation' in spec:
            self.generation = spec['generation'] + 1 # inherits from best_candidate
        else:
            self.generation = 0

        if 'valid_params' in spec:
            for p in spec['valid_params']:
                if not p in self.params:
                    raise ValueError('Invalid param name: {}'.format(p))
            self.valid_params = spec['valid_params']
        else:
            self.valid_params = self.params

        if 'valid_types' in spec:
            self.valid_types = spec['valid_types']
        else:
            self.valid_types = ['discrete', 'continuous', 'categorical', 'preordered'] # all available value types

        if 'arch_type_only' in spec:
            self.arch_type_only = spec['arch_type_only']
        else:
            self.arch_type_only = False

        # select hyperparameter dimensions that are valid
        self.valid_dims = []
        for param in self.valid_params:            
            vt = self.config.get_value_type(param)
            if vt in self.valid_types:
                if self.arch_type_only:
                    if self.config.is_architectural(param):
                        self.valid_dims.append(param)
                else:
                    self.valid_dims.append(param)
        
        self.schemata = []
        self.generations = []

    def generate(self):
        np.random.seed(self.seed)         
        nc_list = []

        try:
            for i in range(self.num_samples):
                schema = copy.copy(self.schema)
                nc, n_i = self.perturb(len(self.valid_dims))
                schema[n_i] = 1                            
                nc2 = self.config.convert("dict", "norm_arr", nc)  
                nc_list.append(nc2)
                self.schemata.append(schema)
                self.generations.append(self.generation)
        except Exception as ex:
            warn("Partial local search sampling failed:{}".format(ex))
            debug(traceback.format_exc())
        finally:
            return np.array(nc_list)

    def get_schemata(self):
        return self.schemata

    def get_generations(self):
        return self.generations

    def perturb(self, num_dim, excluded_index=None):
        i = 0
        if num_dim > 1:
            i = random.randint(0, num_dim - 1) # choose param index
        
            if excluded_index != None and excluded_index < num_dim - 1:
                # choose the other param index
                while i == excluded_index:
                    i = random.randint(0, num_dim - 1) 
        
        hp_name = self.valid_dims[i]

        vt = self.config.get_value_type(hp_name)
        t = self.config.get_type(hp_name)
        r = self.config.get_range(hp_name)

        p_val = self.candidate[hp_name] # the value of choosen param
        np_val = None

        ''' returns perturbed value as dictionary type ''' 
        ovt = OneHotVectorTransformer(self.config)
        n_val = ovt.encode(vt, t, r, p_val)
        if vt == 'categorical': 
            try:
                # force to choose any others                
                ot_opts = np.delete(r, n_val.index(1.0), 0)
                np_val = np.random.choice(ot_opts)
            except Exception as ex:
                debug("Perturbation failed: {}".format(ex))
                return self.perturb(num_dim, excluded_index=i)
        elif vt == 'preordered':
            try:
                # force to choose any others                
                ot_opts = np.delete(r, r.index(p_val), 0)
                np_val = np.random.choice(ot_opts)
            except Exception as ex:
                debug("Perturbation failed: {}".format(ex))
                return self.perturb(num_dim, excluded_index=i)            
        else:
            n_retry = 0
            MAX_RETRY = 100
            while True: # force to one exchange neighbourhood
                r_val = np.random.normal(n_val, self.sd) # random draw from normal
                if r_val < 0.:
                    r_val = 0.0
                elif r_val > 1.:
                    r_val = 1.0                
                un_val = ovt.decode(vt, t, r, r_val)
                # Value check                        
                if un_val < r[0] or un_val > r[-1]:
                    warn("{} is not in {}".format(un_val, r))
                    continue

                if p_val != un_val: # check parameter changed
                    np_val = un_val
                    break
                elif n_retry > MAX_RETRY:
                    # exit to avoid too many iteration
                    np_val = un_val
                    break
                else:
                    n_retry += 1                    

        nc = copy.copy(self.candidate)
        nc[hp_name] = np_val

        if self.verify(nc) == False:
            return self.perturb(num_dim, excluded_index=i)
        else:
            return nc, i
