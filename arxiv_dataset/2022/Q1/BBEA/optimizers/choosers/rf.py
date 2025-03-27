import random
import numpy        as np

import sklearn.ensemble

from xoa.commons.logger import *

from .base import BaseChooser
from .acq import get_acq_func
from .transform import *


class RandomForestRegressorWithVariance(sklearn.ensemble.RandomForestRegressor):

    def predict(self,X):
        # Check data
        X = np.atleast_2d(X)

        all_y_hat = [ tree.predict(X) for tree in self.estimators_ ]

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators
        y_var = np.var(all_y_hat,axis=0,ddof=1)

        return y_hat, y_var


class RFChooser(BaseChooser):
    
    def __init__(self, space, 
                 n_trees=50,
                 max_depth=None,
                 min_samples_split=2, 
                 max_monkeys=7,
                 max_features="auto",
                 n_init_pop=2, max_obs=None,
                 n_jobs=1,
                 random_state=None,
                 shaping_func="no_shaping",
                 alpha=0.3):
        
        self.n_trees = float(n_trees)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_init_pop = int(n_init_pop)
        self.n_jobs = float(n_jobs)
        self.random_state = random_state
        self.alpha = float(alpha)
        self.model = RandomForestRegressorWithVariance(n_estimators=n_trees,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    max_features=max_features,
                                                    n_jobs=n_jobs,
                                                    random_state=random_state)

        acq_funcs = ['EI', 'PI', 'UCB']
        super(RFChooser, self).__init__(space, acq_funcs, shaping_func)

    def next(self, af, min_epoch=0):        

        if not af in self.acq_funcs:
            raise ValueError("Not supported acqusition function!")
        
        candidates = np.array(self.search_space.get_candidates()) 
        completions = np.array(self.search_space.get_completions(min_epoch))
        
        # Don't bother using fancy RF stuff at first.
        if len(completions) == 0:
            return int(candidates[0]), None # return the first candidate
        elif completions.shape[0] < self.n_init_pop:
            return int(random.choice(candidates)), None

        # Grab out the relevant sets.        
        cand_vec = self.search_space.get_param_vectors("candidates")
        comp_vec, errs = super(RFChooser, self).get_valid_obs(min_epoch)
        
        if len(errs) == 0:
            raise ValueError("No actual errors available")
        else:
            errs = self.output_transform(errs)

        if not self.is_modeled(candidates, completions):
            self.model_spec = { "outputs" : candidates, "inputs": completions }
            # Current best.
            self.best_err = np.min(errs)  
            self.func_m, self.func_v = self.build_model(comp_vec, cand_vec, errs)

        acq_func = get_acq_func(af)        
        af_values = acq_func(self.best_err, self.func_m, self.func_v)
        
        best_cand = np.argmax(af_values)
                
        est_values = {
            'candidates' : candidates.tolist(),
            'acq_funcs' : af_values.tolist(),
            'means': self.func_m.tolist(),
            'vars' : self.func_v.tolist()
        }

        return int(candidates[best_cand]), est_values


    def estimate(self, af, test_set_size=10, metric='top1'):
        if not af in self.acq_funcs:
            raise ValueError("Not supported acqusition function!")
        acq_func = get_acq_func(af)
        
        comp_vec, errs = self.get_valid_obs()

        if len(errs) <= test_set_size:
            return None 
        else:
            errs = self.output_transform(errs) 
            scores = self.cross_validate(comp_vec, errs, acq_func, test_set_size=test_set_size, metric=metric)
            mean_score = float(sum(scores) / len(scores))
            debug('[RF_{}-{}] {}-fold CV performance: {:.4f}'.format(self.shaping_func, af, len(scores), mean_score))
        return mean_score

    def build_model(self, comp_vec, cand_vec, errs):

        self.model.fit(comp_vec, errs)
        return self.model.predict(cand_vec)
                
