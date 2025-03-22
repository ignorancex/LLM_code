import numpy as np
import traceback

from xoa.commons.logger import *

class SamplerProtype(object):
    def __init__(self, config, spec, verifier=None):
        
        if 'num_samples' in spec:
            self.num_samples = spec['num_samples']
        else:
            self.num_samples = 20000 

        if 'seed' in spec:
            seed = spec['seed']
            if spec['seed'] == 'random':
                seed = np.random.randint(self.num_samples)
            self.seed = seed
        else:
            self.seed = 1 

        self.config = config
        self.params = config.get_param_names()
        self.num_dim = len(self.params)
        self.verifier = verifier
        self.verified = False       

    def get_sample_size(self):
        return self.num_samples

    def get_name(self):
        if self.name:
            return self.name
        else:
            return "Undefined"

    def is_verified(self):
        if self.verified == None:            
            return True # XXX: No verifier is treated as all of candidate are verified
        else:
            return self.verified

    def validate(self, candidate):

        if type(candidate) != dict:
            candidate = self.config.convert("arr", "dict", candidate)        

        cand = {}
        try:
            # Type forcing
            for k in candidate:
                if not k in self.config.get_param_names():
                    raise ValueError("{} is not in {}".format(k, self.params))
                v = candidate[k]
                t = eval(self.config.get_type(k))
                v = t(v)
                # Value check
                r_k = self.config.get_range(k)
                vt = self.config.get_value_type(k)
                if vt == 'categorical' or vt == 'preordered':
                    if not v in r_k:
                        raise ValueError("{} is not in {}".format(v, r_k)) 
                else:
                    if v < r_k[0] or v > r_k[-1]:
                        raise ValueError("{} is not in {}".format(v, r_k))
                cand[k] = v 

            if self.verifier != None:
                self.verified = True            
                if not self.verify(cand):
                    debug("Verification failed: {}".format(cand))
                    return False

            return cand            
        
        except Exception as ex:
            warn("Candidate validation failed:{}".format(ex))
            return False

    def is_duplicated(self, candidates, hpv_to_check):
        if type(hpv_to_check) == dict:
            hpv_to_check = self.config.convert('dict', 'arr', hpv_to_check)
        pvc = self.config.convert('arr', 'one_hot', hpv_to_check)

        for c in candidates:
            hpv = c['hpv']
            pv = self.config.convert('arr', 'one_hot', hpv)
            dist = np.linalg.norm(np.array(pv) - np.array(pvc))
            if dist < 1e-5: # XXX: ignore very small difference
                #debug("Duplicated configuration: {}".format(hpv_to_check))
                return True

        return False

    def verify(self, cand): # cand is dict type
        
        if self.verifier != None:
            return self.verifier.verify(cand)
        else:
            return True # skip verification when no verifier set up

    def get_schemata(self):
        '''return empty schemata'''
        return np.array(np.zeros((self.num_samples, self.num_dim)))

    def get_generations(self):
        '''return all zeros'''
        return np.array(np.zeros(self.num_samples))

    def generate(self):
        ''' returns M * N normalized vectors '''
        raise NotImplementedError("This method should return given samples # * parameters # dimensional normalized array.")
