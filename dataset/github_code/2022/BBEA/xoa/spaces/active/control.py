import os
import time
import json
import copy
import sys
import random
import traceback

import numpy as np
from itertools import combinations

from xoa.commons.logger import * 
from xoa.spaces.candidates import CandidateSetGenerator


class CandidateSetController(object):

    def __init__(self, space, verifier=None):
        self.search_space = space
        self.search_space.restore_candidates()
        self.verifier = verifier

    def append(self, spec, total_samples):
        if not 'add' in spec:
            raise ValueError("Invalid search space spec!")
            return False

        add_spec = spec['add']
        method_type = 'uniform'
        number = 1
        cur_size = len(self.search_space.get_candidates())
        
        # parsing append scheme
        try:
            if type(add_spec) == str:
                if '[' in add_spec and  ']' in add_spec:
                    o_i = add_spec.find('[')
                    e_i = add_spec.find(']')
                    method_type = add_spec[:o_i]
                    remains = add_spec[o_i+1:e_i]
                    if remains == 'refill':
                        number = total_samples - cur_size
                    else:
                        number = int(remains)
            elif type(add_spec) == int:
                method_type = 'uniform'
                number = int(add_spec)
                
            else:
                raise TypeError("Invalid method: {}".format(add_spec))
                return False

        except Exception as ex:
            warn("{}, Add a candidate by uniform random.".format(ex))
            method_type = 'uniform'
            number = 1
        
        total_samples = number
        
        if 'verification' in spec and spec['verification']:
            verifier = self.verifier
        else:
            verifier = None 

        spec = copy.copy(spec)
        spec['sample_method'] = method_type
        spec['num_samples'] = number
        # Randomizes the seed value
        spec['seed'] = random.randint(cur_size, cur_size * 2)
        if method_type == 'Sobol':
            spec['num_skips'] = number * random.randint(cur_size, cur_size * 2)
        
        hvg = CandidateSetGenerator(self.search_space.get_hp_config(), spec, verifier=verifier)
        hvg.generate()    
        hpvs = hvg.get_hp_vectors()

        
        if total_samples != len(hpvs):
            warn("[Add] # of items to be added: {}, actual: {}".format(total_samples, len(valid_hpvs)))

        if len(hpvs) > 0:
            self.search_space.expand(hpvs)
            return True
        else:
            warn("No candidate added by {}: {}".format(add_spec, spec))
            return False

    def intensify(self, spec):
        try:
            
            total_samples = spec['num_samples']
            if total_samples <= 0:
                return False
                
            if 'verification' in spec and spec['verification']:
                verifier = self.verifier
            else:
                verifier = None 

            spec['seed'] = random.randint(0, total_samples)

            hvg = CandidateSetGenerator(self.search_space.get_hp_config(), spec, verifier=verifier)
            hvg.generate()    
            hpvs = hvg.get_hp_vectors()
            schemata = hvg.get_schemata()
            gen_counts = hvg.get_generations()

            if total_samples != len(hpvs):
                warn("[Intensify] # of items to be added: {}, actual: {}".format(total_samples, len(hpvs)))
            if len(hpvs) > 0:
                self.search_space.expand(hpvs, schemata, gen_counts)
                return True
            return False
        
        except Exception as ex:
            warn("Exception raised on intensifying samples: {}".format(ex))
            return False

    def evolve(self, spec):
        try:
            total_samples = spec['num_samples']
            if 'verification' in spec and spec['verification']:
                verifier = self.verifier
            else:
                verifier = None 
            
            spec['seed'] = random.randint(0, total_samples)
            hvg = CandidateSetGenerator(self.search_space.get_hp_config(), spec, verifier=verifier)
            hvg.generate()    
            hpvs = hvg.get_hp_vectors()
            schemata = hvg.get_schemata()
            gen_counts = hvg.get_generations()

            if total_samples != len(hpvs):
                warn("[Evolve] # of items to be added: {}, actual: {}".format(total_samples, len(hpvs)))
            if len(hpvs) > 0:
                self.search_space.expand(hpvs, schemata, gen_counts)
                return True
            else:
                return False
        except Exception as ex:
            warn("Exception raised on evolving samples: {}".format(ex))
            debug(traceback.format_exc())
            return False     

    def remove_evaluated(self, hpvs):
        ''' check whether hpv is already in the history '''
        ds_t = time.time()
        n_dup = 0
        valid_hpvs = []
        for i in range(len(hpvs)):
            hpv = hpvs[i]
            if self.search_space.is_evaluated(hpv):
                n_dup += 1
            else:
                valid_hpvs.append(hpv)
            
        debug("# of duplicates: {} (Takes {:.1f} sec.)".format(n_dup, time.time() - ds_t))
        return valid_hpvs

    def remove(self, method, estimates):
        if method == 'all_candidates':            
            self.search_space.set_candidates([]) # FIXME:to avoid expensive operation
            return True
        
        if not 'candidates' in estimates or not 'acq_funcs' in estimates:
            warn("Samples can not be removed without estimated values")
            return False
        
        if not '[' in method or not ']' in method:
            warn("Invalid method format: {}".format(method))
            return False
        else:
            o_i = method.find('[')
            e_i = method.find(']')

            try:
                method_type = method[:o_i]
                number = int(method[o_i+1:e_i])
                #debug("Removing {} by {}".format(number, method_type))
            
            except Exception as ex:
                warn("Invalid method name: {}".format(method))
                return False

            try:
                start_t = time.time()
                cands = np.array(estimates['candidates']) # has index
                est_values = np.array(estimates['acq_funcs']) # estimated performance by acquistion function
                if len(cands) < number:
                    raise ValueError("Invalid method - removing {} samples can not performed over {} candidates".format(number, len(cands)))

                # supported method type: worst, except_top 
                if method_type == 'worst':
                    # find worst {number} items to delete
                    #worst_k = est_values.argsort()[:number][::1] # FIXME: takes long time to sort                
                    worst_k = np.argpartition(est_values, number)[:number]
                    s_t = time.time()
                    self.search_space.remove(cands[worst_k])
                    #debug("Worst item remove time: {:.2f}".format(time.time() - s_t))
                elif method_type == 'except_top':
                    # find top {number} items to be saved
                    #top_k = est_values.argsort()[-1 * number:][::-1] # FIXME: takes long time to sort
                    top_k = np.argpartition(est_values, -1 * number)[-1 * number:]
                    remains = np.setdiff1d(cands, cands[top_k]) # remained items
                    self.search_space.remove(remains)
                else:
                    raise ValueError("Invalid method type: {}".format(method))
                    return False

            except Exception as ex:
                warn("Removing sample failed: {}".format(ex))
                return False
            
            return True

    def resample(self, estimates, space_cfg):
        s_t = time.time()
        total_samples = 20000

        if "num_samples" in space_cfg:
            total_samples = space_cfg["num_samples"]
        elif "num_samples" in space_cfg["search_space"]:
            total_samples = space_cfg["search_space"]["num_samples"]
        else:
            warn("Use of default sample size: {}".format(total_samples))
                
        if estimates == None or not 'candidates' in estimates:
            debug("No Estimated value is available to do resampling.")
        else:    
            if 'remove' in space_cfg:
                start_t = time.time()
                ds = space_cfg["remove"]
                if self.remove(ds, estimates):
                    debug("Candidates have been removed by {} ({:.1f} sec)".format(ds, time.time() - start_t))

            if 'add' in space_cfg:
                start_t = time.time()
                if self.append(space_cfg, total_samples):
                    debug("Candidates have been added by {} ({:.0f} sec)".format(space_cfg['add'], time.time() - start_t))

            if 'intensify' in space_cfg:
                start_t = time.time()

                ns = 10 # XXX:initial size
                pn = 10 # parent size
                
                try:
                    if type(space_cfg['intensify']) == int:
                        ns = space_cfg['intensify']
                    elif ']' in space_cfg['intensify']:
                        # Tokenize evolve strategies
                        e_spec = space_cfg['intensify']
                        o_i = e_spec.find('[')
                        e_i = e_spec.find(']')
                        method = e_spec[:o_i]
                        if not 'top-' in method:
                            raise NotImplementedError("Not supported mutation method: {}. Currently top-* supported only.".format(method))
                        else:
                            pn = int(method[4:])
                        ns = int(e_spec[o_i+1:e_i])
                    else:
                        raise ValueError("Invalid mutation strategy: {}".format(space_cfg['intensify']))
                except Exception as ex:
                    warn("Fail to decode mutation strategy: {}".format(ex))
                    debug(traceback.format_exc())
                    raise ValueError(ex)
                cands = np.array(estimates['candidates']) # has index
                est_values = np.array(estimates['acq_funcs']) # estimated performance by acquistion function
                top_k = est_values.argsort()[-1*pn:][::-1]
                
                if pn > 1:
                    try:                                         
                        top_k_est_values = est_values[top_k]
                        norm_values = top_k_est_values / sum(top_k_est_values)                    
                        
                        debug("Top-k values: {}, normalized: {}".format(top_k_est_values, norm_values))
                        pop_sizes = np.array(norm_values) * ns
                        pop_sizes = pop_sizes.astype(np.int)
                    
                    except Exception as ex:
                        warn("Fail to calculate pop size: {}".format(ex))
                        debug(traceback.format_exc())                        
                else:
                    pop_sizes = np.array([ ns ])

                i = 0
                for k in cands[top_k]:
                    if self.search_space.is_existed(k):
                        spec = copy.copy(space_cfg)
                        spec['num_samples'] = pop_sizes[i]
                        spec['sample_method'] = 'local'
                        spec['seed'] = random.randint(0, total_samples)
                        spec['female'] = self.search_space.get_hpv(k)
                        spec['generation'] = self.search_space.get_generation(k)
                        self.intensify(spec)
                    else:
                        warn("Invalid candidate index: {}".format(k))
                    i += 0

                debug("Candidates have been intensified. ({:.1f} sec)".format(time.time() - start_t))

            if 'evolve' in space_cfg:
                start_t = time.time()
                spec = copy.copy(space_cfg)
                spec['sample_method'] = 'genetic'
                incum = self.search_space.get_incumbent() # has {"hpv":list, "schema": list, "gen": int}

                if not 'mutation_ratio' in spec: 
                    spec['mutation_ratio'] = .1

                ns = 10 # XXX:initial size
                pn = 1 # parents size
                                
                try:
                    spec['male'] = incum['hpv']
                    spec['schema'] = incum['schema']
                    spec['gen'] = incum['gen']

                    # Decode evolving method
                    if type(space_cfg['evolve']) == int:
                        ns = space_cfg['evolve']
                    elif ']' in space_cfg['evolve']:
                        # Tokenize evolve strategies
                        e_spec = space_cfg['evolve']
                        o_i = e_spec.find('[')
                        e_i = e_spec.find(']')
                        method = e_spec[:o_i]
                        if not 'top-' in method:
                            raise NotImplementedError("Not supported evolutionary method: {}. Currently top-* supported only.".format(method))
                        else:
                            pn = int(method[4:])
                        ns = int(e_spec[o_i+1:e_i])
                    else:
                        raise ValueError("Invalid evolutionary strategy: {}".format(space_cfg['evolve']))
                except Exception as ex:
                    warn("Fail to decode evolutionary strategy: {}".format(ex))
                    debug(traceback.format_exc())
                    raise ValueError(ex)

                # Count # of promising samples using estimated values
                cands = np.array(estimates['candidates']) # has index
                est_values = np.array(estimates['acq_funcs']) # estimated performance by acquistion function
                top_k = est_values.argsort()[-1*pn:][::-1]
                if pn > 1:
                    try:                                         
                        top_k_est_values = est_values[top_k]
                        norm_values = top_k_est_values / sum(top_k_est_values)                    
                        
                        debug("Top-k values: {}, normalized: {}".format(top_k_est_values, norm_values))
                        pop_sizes = np.array(norm_values) * ns
                        pop_sizes = pop_sizes.astype(np.int)
                    
                    except Exception as ex:
                        warn("Fail to calculate pop size: {}".format(ex))
                        debug(traceback.format_exc())                        
                else:
                    pop_sizes = np.array([ ns ])

                debug("Evolutionary strategy: {}, population to evolve: {}".format(space_cfg['evolve'], pop_sizes))
                
                # Evolve samples
                i = 0
                for k in cands[top_k]:
                    spec['num_samples'] = pop_sizes[i]
                    spec['female'] = self.search_space.get_hpv(k)
                    spec['schema'] = self.search_space.get_schema(k)
                    if spec['num_samples'] > 0:
                        self.evolve(spec)
                    i += 0
                debug("Candidates have been evolved. ({:.1f} sec)".format(time.time() - start_t))

            # FIXME:obsolete feature. do not use below 
            if 'evolve_div' in space_cfg:
                start_t = time.time()
                spec = copy.copy(space_cfg)
                spec['sample_method'] = 'genetic'
                n_ts = spec['evolve_div']

                if not 'mutation_ratio' in spec: 
                    spec['mutation_ratio'] = .1                
                
                if not 'candidates' in estimates or not 'acq_funcs' in estimates \
                    or not 'all_selection' in estimates:
                    warn("No diversified evolution without estimated values")
                else:    
                    # diversified evolution                
                    selections = estimates['all_selection']
                    unique_selections = [] 
                    for s in list(set(selections)):
                        if self.search_space.is_existed(s):
                            unique_selections.append(s)

                    n_us = len(unique_selections) # number of unique selections
                    debug("[evolve_div] {} selected parents: {}".format(n_us, unique_selections))
                    if n_us == 0:
                        warn("No parents to evolve available!")
                        return
                    
                    try:
                        combi_list = list(combinations(unique_selections, 2))
                        
                        pop_size = int(n_ts / len(combi_list))
                        div_values = []
                        specs = []
                                        
                        for c in combi_list:
                            p1_index = c[0]
                            p2_index = c[1]
                            p1_vec = self.search_space.get_param_vectors(p1_index)
                            p2_vec = self.search_space.get_param_vectors(p2_index)
                            dist = np.linalg.norm(np.array(p1_vec) - np.array(p2_vec))
                            #debug("Distance btw {} & {}: {}".format(p1_index, p2_index, dist))
                            div_values.append(dist)
                            
                            s = copy.copy(spec)
                            s['male'] = self.search_space.get_hpv(p1_index)
                            s["schema"] = self.search_space.get_schema(p1_index)
                            s["gen"] = self.search_space.get_generation(p1_index)
                            
                            s['female'] = self.search_space.get_hpv(p2_index)
                            specs.append(s)

                        for i in range(len(specs)):
                            spec = specs[i]
                            div_v = div_values[i]
                            spec['num_samples'] = int(n_ts * div_v / sum(div_values))
                            if spec['num_samples'] > 0:
                                self.evolve(spec)
                    except Exception as ex:
                        warn('[evolve_div] exception raised:'.format(ex))
                        debug(traceback.format_exc())
            # FIXME:remove above code

            cand_size = len(self.search_space.get_candidates())
            debug("The # of candidates: {}, resampling time: {:.1f} sec.".format(cand_size, time.time() - s_t))  
