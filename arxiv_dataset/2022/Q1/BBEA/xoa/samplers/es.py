import numpy as np
import random

from itertools import combinations
from itertools import combinations_with_replacement as cwr

from xoa.commons.logger import *

from .proto import SamplerProtype
from .ls import NeighborSampler


class EvolutionarySampler(SamplerProtype):

    def __init__(self, config, spec, verifier=None):
        
        if not 'mutation_ratio' in spec:
            self.m_ratio = .1 # default mutation ratio
        else:
            self.m_ratio = spec['mutation_ratio']
                
        self.male = spec['male']
        if 'gen' in spec: 
            self.generation = spec['gen'] + 1 # set offsprings' generation
        else:
            self.generation = 0
        if 'schema' in spec:
            self.m_schema = [ int(f) for f in spec['schema'] ] # type forcing
        else:
            self.m_schema = [ 0 for i in self.male ]

        self.female = spec['female']
        
        self.schemata = []
        self.generations = []
        self.name = 'evolutionary sampling'        

        super(EvolutionarySampler, self).__init__(config, spec, verifier)

    def generate(self):
        random.seed(self.seed)
        evol_grid = []
        self.schemata = []
        candidates = []
        n_dim = len(self.male)

        candidates = self.cross_over_mp(self.num_samples)
        #candidates = self.cross_over_uniform(self.num_samples)
        #n_remains = self.num_samples - len(candidates)
        
        offsprings = []
        while len(offsprings) < self.num_samples:
            m = random.sample(candidates, 1)[0]
            if np.random.rand() <= self.m_ratio:
                m_ = self.mutate(m)
                if m_ != False: 
                    offsprings.append(m_)
                else:
                    offsprings.append(m) # add normal case
            else:
                offsprings.append(m)

        for o in offsprings:            
            g = self.config.convert('arr', 'norm_arr', o['hpv'])
            evol_grid.append(g)
            self.schemata.append(o['schema'])
            self.generations.append(o['gen'])
        
        return np.array(evol_grid)

    def get_schemata(self):
        return np.array(self.schemata)

    def get_generations(self):
        return np.array(self.generations)

    def get_random_mask(self):
        schemeta = []
        masks = list(cwr([0, 1], self.num_dim))
        masks = masks[1:-1] # remove all zero or all one mask
        s = np.random.randint(len(masks))
        schema = list(masks[s])
        np.random.shuffle(schema)

        return schema

    def cross_over_uniform(self, num_child):
        # populate offsprings from parents
        offsprings = []
        n_invalid = 0
        n_duplicate = 0

        while len(offsprings) < num_child:
            try:             
                o_schema = self.get_random_mask()           
                o_hpv = [] # child hyperparam vector
                for i in range(len(o_schema)):
                    bit = o_schema[i]
                    if bit == 0: # inherit from male
                        o_hpv.append(self.male[i])
                    elif bit == 1: # inherit from female
                        o_hpv.append(self.female[i])
                    else:
                        raise ValueError("Invalid child schema: {}".format(o_schema))
                
                if self.validate(o_hpv) != False:
                    child = {"hpv": o_hpv, "schema": o_schema, "gen": self.generation }
                    offsprings.append(child)

            except Exception as ex:
                debug("Cross over failed: {}".format(ex))

        return offsprings # list item - {"hpv": [], "schema": []}

    def cross_over_mp(self, num_child):
        offsprings = []
        o_schemata = self.create_schemata(num_child * 2) # XXX: to avoid large size issue
        random.shuffle(o_schemata)
        n_offsprings = len(o_schemata)
        n_invalid = 0
        n_duplicate = 0

        if n_offsprings < num_child:
            debug("The # of possible offsprings is less then {}".format(num_child))
        
        for o_schema in o_schemata:
            o_hpv = [] # child hyperparam vector
            for i in range(len(o_schema)):
                bit = o_schema[i]
                if bit == 0: # inherit from male
                    o_hpv.append(self.male[i])
                elif bit == 1: # inherit from female
                    o_hpv.append(self.female[i])
                else:
                    raise ValueError("Invalid child schema: {}".format(o_schema))

            try:
                # validate new parameter                
                if self.validate(o_hpv) != False:
                    # check duplicated
                    if not self.is_duplicated(offsprings, o_hpv):
                        child = {"hpv": o_hpv, "schema": o_schema,"gen": self.generation }
                        offsprings.append(child)
                        if len(offsprings) == num_child:
                            break                         
                    else:
                        #debug("Duplicated configuration: {}".format(o_hpv))
                        n_duplicate += 1
                else:
                    #debug("Invalid config: {}".format(o_hpv))
                    n_invalid += 1
            
            except Exception as ex:
                warn("Cross over failed: {}".format(ex))
        
        if n_invalid > 0 or n_duplicate > 0:
            debug("P1: {}".format(self.male))
            debug("P2: {}".format(self.female))            
            debug("# of child: {}/{}, invalid: {}, duplicated: {}".format(len(offsprings), num_child, n_invalid, n_duplicate))
        return offsprings 

    def create_schemata(self, num_child=None):
        o_schemata = [] 
        # create all possible offsprings' schema
        n_params = len(self.m_schema)
        
        for i in range(1, n_params):
            for c in self.create_schema_list(n_params, i):
                o_schemata.append(c)
                if num_child != None and len(o_schemata) >= num_child:
                    break
        
        return o_schemata

    def create_schema_list(self, n_p, n_on):
        arr = []
        combi = combinations([ i for i in range(n_p)], n_on)
        for c in combi:
            a = [0 for i in range(n_p)]
            for i in c:
                a[i] = 1
            arr.append(a)
        
        random.shuffle(arr)
        return arr

    def mutate(self, cand):
        
        # mutate this candidate
        spec = { 'num_samples': 1 }
        spec['female'] = cand['hpv']
        spec['generation'] = self.generation
        spec['seed'] = self.seed
        
        lsg = NeighborSampler(self.config, spec)
        hpv_dict, n_i = lsg.perturb(self.num_dim) # return dict type
        r_schema = cand['schema']
        
        # XOR operation in n_i
        if r_schema[n_i] == 1:
            r_schema[n_i] = 0
        elif r_schema[n_i] == 0:
            r_schema[n_i] = 1
        else:
            debug("Invalid schema: {}".format(r_schema))
            return False
        
        if self.validate(hpv_dict) != False:
            m_cand = { "hpv": self.config.convert("dict", "arr", hpv_dict), 
                    "schema": r_schema, 
                    "gen": self.generation
                    } 
            return m_cand
        else:
            debug("Invalid configuration: {}".format(hpv_dict))
            return False
        