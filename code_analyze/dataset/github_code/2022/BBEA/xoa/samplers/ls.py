import numpy as np
import random
import traceback
import copy

from xoa.commons.logger import *
from xoa.commons.converter import OneHotVectorTransformer

from .proto import SamplerProtype


class NeighborSampler(SamplerProtype):
    def __init__(self, config, spec, verifier=None):        
        self.name = 'Local sampling'
               
        super(NeighborSampler, self).__init__(config, spec, verifier)
        
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

        self.schemata = []
        self.generations = []

    def generate(self):
        np.random.seed(self.seed)         
        nc_list = []

        try:
            for i in range(self.num_samples):
                schema = copy.copy(self.schema)
                nc, n_i = self.perturb(self.num_dim)
                schema[n_i] = 1                            
                nc2 = self.config.convert("dict", "norm_arr", nc)  
                nc_list.append(nc2)
                self.schemata.append(schema)
                self.generations.append(self.generation)
        except Exception as ex:
            warn("Local search sampling of {} failed:{}".format(self.candidate, ex))
            debug(traceback.format_exc())
        finally:
            return np.array(nc_list)

    def get_schemata(self):
        return self.schemata

    def get_generations(self):
        return self.generations

    def perturb(self, num_dim, excluded_index=None):
        
        i = random.randint(0, num_dim - 1) # choose param index
        if excluded_index != None:
            # choose the other param index
            while i == excluded_index:
                i = random.randint(0, num_dim - 1) 

        ''' returns perturbed value as dictionary type ''' 
        ovt = OneHotVectorTransformer(self.config)
        hp_name = self.params[i] # hyperparameter name

        vt = self.config.get_value_type(hp_name)
        t = self.config.get_type(hp_name)
        r = self.config.get_range(hp_name)

        p_val = self.candidate[hp_name] # the value of choosen param
        np_val = None

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

