import time

from xoa.commons.logger import * 

from .resample import CandidateSetResampler


class CandidateSetEnsembler(CandidateSetResampler):

    def __init__(self, space, modeller, verifier=None):        
        super(CandidateSetEnsembler, self).__init__(space, modeller, verifier)

    def ensemble(self, s_method, n_cand, n_steps, steps_after_best):
        s_t = time.time()
        n_p, n_c, dist, ord, kp = self.parse_method(s_method, n_cand, n_steps, steps_after_best)

        if '-ALL' in s_method:
            candidates = self.select_batch(n_cand)
            self.search_space.set_candidates(candidates)
        elif '-MUTATE' in s_method:
            candidates = self.mutate_batch(n_cand)
            self.search_space.set_candidates(candidates)
            pass        
        else:
            candidates = self.crossover_batch(n_cand, n_p, n_c, 
                                    distance=dist, order_by=ord, keep_parents=kp)
            self.search_space.set_candidates(candidates) 
        debug("Candidate reproduction took {:.1f} sec. to evolve {} candidates".format(time.time() - s_t, len(candidates)))

    def parse_method(self, s_method, n_cand, n_steps, steps_after_best):
        
        n_p = 60
        n_c = 3000
        if n_cand > n_c:
            n_c = n_cand

        dist = 'cosine'
        order = 'low' # make exploitative
        
        # TODO:add more experiment conditions
        if '-P600' in s_method:
            n_p = 600            
        elif '-P300' in s_method:
            n_p = 300
        elif '-P120' in s_method:
            n_p = 120
        elif '-P60' in s_method:
            n_p = 60
        elif '-P30' in s_method:
            n_p = 30

        if '-C20k' in s_method:
            n_c = 20000
        if '-C10k' in s_method:
            n_c = 10000
        elif '-C5k' in s_method:
            n_c = 5000  
        elif '-C3k' in s_method:
            n_c = 3000 
        elif '-C1k' in s_method:
            n_c = 1000 

        if '-L2' in s_method:
            dist = 'l2'
        elif '-L1' in s_method:
            dist = 'l1'

        if '-H' in s_method:
            order = 'high' # make explorative
        elif '-N' in s_method:
            order = 'none'
        elif '-R' in s_method:
            order = 'random'                
        elif '-S10' in s_method: # round-robin 10:1
            if n_steps % 10 == 1:
                order = 'high' # make explorative
            else:
                order = 'low'
        elif '-S5' in s_method: # round-robin 5:1
            if n_steps % 5 == 1:
                order = 'high' # make explorative
            else:
                order = 'low'
        elif '-S' in s_method: # round-robin 1:1
            if n_steps % 2 == 1:
                order = 'high' # make explorative
            else:
                order = 'low'
        elif '-ADA2' in s_method:
            rn = self.steps_after_best % 4
            if rn == 0:
                order = 'low' # make exploitative
            elif rn == 1:
                order = 'random' 
            elif rn == 2:
                order = 'high' # make explorative
            else:
                # no subsampling will be performed
                return
        elif '-ADA1' in s_method:
            if n_steps < 20:
                order = 'random'
            elif steps_after_best % 2 == 0:
                order = 'low' # make exploitative
            else:
                order = 'high' # make explorative
        elif '-ADA' in s_method:
            if n_steps < 20:
                order = 'random'
            elif steps_after_best < 3:
                order = 'low' # make exploitative
            else:
                order = 'high' # make explorative

        keep_parents = False
        if '-KP' in s_method:
            keep_parents = True

        return n_p, n_c, dist, order, keep_parents

    def select_batch(self, n_cand):
        s_t = time.time()
        arms = self.modeller.get_arm_list()
        ep_size = int(n_cand / len(arms))
        p_list = []
        for a in arms:
            model = self.modeller.get_model(a['model'])
            _, estimates = model.next(a['acq_func'])
            p_indices = self.get_top_k_indices(estimates, ep_size)
            if len(p_indices) > 0:
                p_list += p_indices.tolist()

        candidates = list(set(p_list))
        return candidates

    def mutate_batch(self, n_cand):
        parents = self.select_batch(n_cand)    
        offspring = []
        for p_index in parents:
            o_hpv = self.mutate(p_index, n_cand)
            offspring.append(o_hpv)

        sample_indices = self.search_space.expand(offspring)            
        return sample_indices

    def crossover_batch(self, n_pop, n_parents, n_child, 
                           distance='cosine', order_by='low', keep_parents=False):
        parents = self.select_batch(n_parents)    
        offspring = self.crossover(parents, n_child, n_pop, distance, order_by, keep_parents)
        return offspring
