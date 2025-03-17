import random
import traceback

import numpy as np

from scipy.spatial import distance

from itertools import combinations
from operator import itemgetter

from xoa.commons.logger import * 
from xoa.spaces.candidates import CandidateSetGenerator
from xoa.samplers import EvolutionarySampler


class CandidateSetResampler(object):

    def __init__(self, space, modeller, verifier=None):
        self.search_space = space

        self.modeller = modeller
        self.verifier = verifier
        self.strategy = None

    def get_diversity(self, p1_index, p2_index, dist_type):
        p1 = self.search_space.get_param_vectors(p1_index)
        p2 = self.search_space.get_param_vectors(p2_index)
        
        if dist_type == 'l2':
            return distance.euclidean(p1, p2)
        elif dist_type == 'l1':
            return distance.cityblock(p1, p2)
        elif dist_type == 'cosine':
            return distance.cosine(p1, p2) # 0~1, the value closed to 1 refers to be independent each other
        else:
            raise ValueError("{} is not supported distance type. Use any of l2, l1, cosine.".format(dist_type))

    def get_top_k_indices(self, estimates, k):
        if estimates and 'candidates' in estimates and 'acq_funcs' in estimates:
            cands = np.array(estimates['candidates']).reshape(-1) # has index
            est_values = np.array(estimates['acq_funcs']).reshape(-1) # estimated performance by acquistion function
            top_k_values = np.argpartition(est_values, -1 * k)[-1 * k:]
            top_k_cands = cands[top_k_values]
            #debug('Top-{} indices: {}'.format(k, top_k_cands))
            return top_k_cands
        else:
            warn("No estimation values to sample candidates")
            return []

    def mutate(self, parent_index, seed_var=100, n_offspring=1):
        spec = { 'num_samples': n_offspring, 'sample_method': 'local'}
        spec['seed'] = random.randint(0, seed_var) # XXX:randomize!
        spec['female'] = self.search_space.get_hpv(parent_index)
        
        hvg = CandidateSetGenerator(self.search_space.get_hp_config(), spec, verifier=self.verifier) 
        hvg.generate()
        offspring = hvg.get_hp_vectors()
        if len(offspring) == 0:
            raise ValueError("Mutation operation failed!")
        
        #debug("{} -> {}".format(spec['female'], offspring))
        return offspring

    def partial_mutate(self, parent_index,
                        valid_params=None, 
                        valid_types=None, 
                        arch_type_only=False, 
                        seed_var=100):
        spec = { 'num_samples': 1, 'sample_method': 'partial'}
        spec['seed'] = random.randint(0, seed_var) # XXX:randomize!
        spec['female'] = self.search_space.get_hpv(parent_index)
        
        if valid_params:
            spec['valid_params'] = valid_params
        
        if valid_types:
            spec['valid_types'] = valid_types

        if arch_type_only:
            spec['arch_type_only'] = arch_type_only

        hvg = CandidateSetGenerator(self.search_space.get_hp_config(), spec, verifier=self.verifier) 
        hvg.generate()
        hpvs = hvg.get_hp_vectors()
        if len(hpvs) != 1:
            raise ValueError("Partial mutation failed!")
        offspring = hpvs[0]
        #debug("{} -> {}".format(spec['female'], offspring))
        return offspring

    def crossover(self, up_list, n_child, n_pop, 
                  distance='cosine', order_by='low', keep_parents=False):

        # pairing all possible parents and create next generation
        all_pairs = list(combinations(up_list, 2))
        p_size = len(all_pairs)
        debug("# of parents: {}, # of pairs: {}".format(len(up_list), p_size))
        hp_config = self.search_space.get_hp_config()
        next_gen_list = []
        
        while len(next_gen_list) < n_child:
            i = random.randint(0, p_size-1)
            p = all_pairs[i]
            spec = { 'sample_method': 'genetic' }
            p1_index = p[0]
            p2_index = p[1]
            spec['male'] = self.search_space.get_hpv(p1_index)
            spec["schema"] = self.search_space.get_schema(p1_index)
            spec["gen"] = self.search_space.get_generation(p1_index)            
            spec['female'] = self.search_space.get_hpv(p2_index)
            spec['num_samples'] = 1
            
            esg = EvolutionarySampler(hp_config, spec, self.verifier)
            offspring_list = esg.cross_over_uniform(1) # list item - {"hpv": [], "schema": []}
            dist = self.get_diversity(p1_index, p2_index, distance)
            next_gen_list.append({"diversity": dist, "offspring": offspring_list})

        #debug("Sorting offspring...")
        if order_by == 'low':
            # ordering next_gen_list by low diversity
            next_gen_list = sorted(next_gen_list, key=itemgetter('diversity'))            
        elif order_by == 'high':
            # ordering next_gen_list by high diversity
            next_gen_list = sorted(next_gen_list, key=itemgetter('diversity'), reverse=True)
        elif order_by == 'random':
            random.shuffle(next_gen_list)
        elif order_by != 'none':
            raise ValueError("Invalid ordering type: {}".format(order_by))

        if len(next_gen_list) > n_pop:
            next_gen_list = next_gen_list[:n_pop]

        # get evolved configuration index
        child_hpv_list = []
        for ng in next_gen_list:
            offspring = ng["offspring"]
            for o in offspring:
                child_hpv = o['hpv']
                child_hpv_list.append(child_hpv)
        debug("[CR] getting indices of offspring for {} configurations".format(len(child_hpv_list)))  
        offspring = self.search_space.expand(child_hpv_list)
        n_s = len(offspring)
        if keep_parents == True:
            offspring += up_list
            n_s = len(offspring)
            debug("[CR] {} parents will be survived: {} -> {}".format(len(up_list), len(child_hpv_list), n_s)) 
        
        if n_s < n_pop:
            warn("[CR] population size is less than {}: {}".format(n_pop, len(next_gen_list)))

        return offspring

    def crossover_uniform(self, up_list, n_child):

        # pairing all possible parents and create next generation
        all_pairs = list(combinations(up_list, 2))
        p_size = len(all_pairs)
        debug("# of parents: {}, # of pairs: {}".format(len(up_list), p_size))
        hp_config = self.search_space.get_hp_config()
        next_gen_list = []
        
        while len(next_gen_list) < n_child:
            i = random.randint(0, p_size-1)
            p = all_pairs[i]
            spec = { 'sample_method': 'genetic' }
            if type(p[0]) == int and type(p[0]):
                p1_index = p[0]
                p2_index = p[1]
                spec['male'] = self.search_space.get_hpv(p1_index)
                spec["schema"] = self.search_space.get_schema(p1_index)
                spec["gen"] = self.search_space.get_generation(p1_index)            
                spec['female'] = self.search_space.get_hpv(p2_index)
            else:
                debug("pair: {}".format(p))
                spec['male'] = p[0]
                spec['female'] = p[1]
            spec['num_samples'] = 1
            
            esg = EvolutionarySampler(hp_config, spec, self.verifier)
            offspring_list = esg.cross_over_uniform(1) # list item - {"hpv": [], "schema": []}
            next_gen_list.append({"offspring": offspring_list})

        # get evolved configuration index
        child_hpv_list = []
        for ng in next_gen_list:
            offspring = ng["offspring"]
            for o in offspring:
                child_hpv = o['hpv']
                child_hpv_list.append(child_hpv)
        debug("[CRU] getting indices of offspring for {} configurations".format(len(child_hpv_list)))  
        offspring = self.search_space.expand(child_hpv_list)

        return offspring

    def resample(self, s_method, n_cand, n_steps, vetoer=None):

        self.search_space.restore_candidates() # XXX:reset candidate set
        
        model_name = 'GP'  # default BO model
        acq_func_name = 'EI' # default acquisition function
        arms = self.modeller.get_arm_list()
        
        m_i = 0

        if 'UNIFORM' in s_method:
            model_name = 'ALL'
            acq_func_name = 'RANDOM'

        elif 'HISTORY' in s_method:
            model_name = 'HISTORY'
            acq_func_name = 'RANDOM'

        elif 'DIV-SEQ' in s_method:
            m_i = n_steps % len(arms)
            a = arms[m_i] 
            model_name = a['model']
            acq_func_name = a['acq_func']
            if vetoer != None and vetoer['model'] == model_name and vetoer['acq_func'] == acq_func_name:
                # get next model                
                m_i = (n_steps + 1) % len(arms)
                a = arms[m_i] 
                model_name = a['model']
                acq_func_name = a['acq_func']
                debug("Switching to {}-{}".format(model_name, acq_func_name))                
        elif 'DIV-RANDOM' in s_method:
            while True:
                m_i = random.randint(0, len(arms) - 1)
                a = arms[m_i] 
                model_name = a['model']
                acq_func_name = a['acq_func']
                if vetoer != None and vetoer['model'] != model_name and vetoer['acq_func'] != acq_func_name:
                    break
        elif 'DIV-DISP' in s_method:
            if self.strategy is None:
                from optimizers.arms.strategies import DisparityCheckStrategy
                choosers = self.modeller.get_models()
                self.strategy = DisparityCheckStrategy(arms, 
                        [0.0 for a in arms], [0 for a in arms],
                        choosers)
            m_i = self.strategy.next(n_steps)
            a = arms[m_i] 
            model_name = a['model']
            acq_func_name = a['acq_func']
        
        elif not 'DIV-' in s_method: # deterministic BO model
            m_i = s_method.rfind('-')
            last_token = s_method[m_i+1:]
            #print(last_token)
            # check last token 
            if 'M' in last_token or 'CO' in last_token:
                m_j = s_method.rfind('-', 0, m_i-1)
                model_name = s_method[:m_j]
                acq_func_name = s_method[m_j+1:m_i]
            else:
                model_name = s_method[:m_i]
                acq_func_name = s_method[m_i+1:]
        else:
            raise ValueError("Not supported model to sample: {}".format(s_method))

        d = len(self.search_space.get_completions())
        debug('Resampling with {}-{} (obs.# {})'.format(model_name, acq_func_name, d))

        try:
            parents = []
            samples = []
            if 'RANDOM' == acq_func_name:
                if model_name == 'HISTORY':
                    parents = np.array(self.search_space.get_completions())
                    if len(parents) < 10: # XXX:ignore when minimum history size
                        parents = []
                    else:
                        parents = np.random.choice(parents, n_cand)

                elif model_name == 'ALL':
                    all_parents = np.array(self.search_space.get_candidates())
                    parents = np.random.choice(all_parents, n_cand)
            else:
                model = self.modeller.get_model(model_name)
                _, estimates = model.next(acq_func_name)

                if estimates != None:
                    parents = self.get_top_k_indices(estimates, n_cand)

            if len(parents) > 0:

                # mutation based recombination strategies
                if '-XM' in s_method:
                    # performing mutation 0.5, none 0.5
                    offspring = []
                    keep_list = []

                    for p_index in parents.tolist():
                        if random.random() < 0.5:
                            # keep parent
                            keep_list.append(p_index)
                        else:
                            o_hpv = self.mutate(p_index, seed_var=n_cand)
                            for v in o_hpv:
                                if len(v) > 0: #XXX: skip empty list
                                    offspring.append(v)
                                else:
                                    debug('mutated offspring is invalid: {}'.format(v))
                                    keep_list.append(p_index)
                    if len(offspring) > 0:
                        samples = self.search_space.expand(offspring)
                        #debug('mutated candidates: {}'.format(samples))
                        #debug('selected candidates: {}'.format(keep_list))
                        samples += keep_list
                    else:
                        samples = keep_list
                    debug("Offsprings generated by mutation ({}) or origin ({})".format(len(offspring), len(keep_list)))
                elif '-M' in s_method:
                    # performing mutation 
                    offspring = []
                    for p_index in parents.tolist():
                        o_hpv = self.mutate(p_index, seed_var=n_cand)
                        for v in o_hpv:
                            offspring.append(v)

                    samples = self.search_space.expand(offspring)
                    debug("offsprings by mutation: {}".format(len(samples)))

                else:
                    # No variation operator
                    samples = parents
            
            if len(samples) > 0:                
                self.search_space.set_candidates(samples)

        except Exception as ex:
            error("Resampling candidate error at {} iterations: {}".format(n_steps, ex))
            debug(traceback.format_exc())
