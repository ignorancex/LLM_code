from .base import BaseChooser
from .acq import get_acq_func
from .transform import *

from xoa.commons.logger import *

import numpy as np
import random
import logging
import time

# InputWarpedGPChooser is implemented using GPy module
from GPy.kern import Matern52, Matern32, Linear 
from GPy.util.input_warping_functions import KumarWarping 
from GPy.models import InputWarpedGP, GPRegression 
from GPy import priors

import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)


class InputWarpedGPChooser(BaseChooser):
    def __init__(self, space, 
                 covar="Matern52", mcmc_iters=10,
                 n_init_pop=2, max_obs=200,
                 shaping_func="no_shaping",
                 warp = True,
                 alpha=0.3):
        
        self.num_dims = space.get_params_dim()
        self.max_obs = int(max_obs)
        self.n_init_pop = int(n_init_pop)

        self.kernel_type = covar
        self.mcmc_iters = int(mcmc_iters)
        self.warp = bool(int(warp))
        self.model = None

        self.alpha = alpha

        acq_funcs = ['EI', 'PI', 'UCB']
        super(InputWarpedGPChooser, self).__init__(space, acq_funcs, shaping_func)        

    def get_valid_obs(self, min_epoch=0):
        comp, errs = super(InputWarpedGPChooser, self).get_valid_obs(min_epoch)
        size = len(comp)
        if size > self.max_obs:
            debug("Subsampling {} observations due to GP cubic complexity".format(self.max_obs))
            nc = []
            ne = []
            indices = np.random.choice(size, self.max_obs)
            for i in indices:
                nc.append(comp[i])
                ne.append(errs[i])
            return np.array(nc), np.array(ne)
        else:
            return np.array(comp), np.array(errs)

    def next(self, af, min_epoch=0):

        if not af in self.acq_funcs:
            raise ValueError("Not supported acqusition function!")
        
        s_t = time.time()

        comp_grid, errs = self.get_valid_obs(min_epoch)

        candidates = np.array(self.search_space.get_candidates()) 
        cand_grid = self.search_space.get_param_vectors("candidates")
        
        # Don't bother using fancy GP stuff at first.
        if len(comp_grid) == 0:
            return int(candidates[0]), None # return the first candidate 
        elif len(comp_grid) < self.n_init_pop:
            return int(random.choice(candidates)), None

        # transform errors for enhancing optimization performance
        errs = self.output_transform(errs)

        if not self.is_modeled(candidates, errs):
            self.model_spec = { "outputs" : candidates, "inputs": errs }                             
            self.func_m, self.func_v = self.build_model(comp_grid, cand_grid, errs)
            
            # Save previous calculation result!
            self.model_spec["ms"] = self.func_m
            self.model_spec["vs"] = self.func_v

        elif "ms" in self.model_spec and "vs" in self.model_spec:
            self.func_m = self.model_spec["ms"]
            self.func_v = self.model_spec["vs"]
        else:
            raise ValueError("Invalid previous model result!")
        
        # Current best.
        self.best_err = np.min(errs) 

        acq_func = get_acq_func(af)
        af_values = acq_func(self.best_err, self.func_m, self.func_v)
        best_cand = np.argmax(af_values)
        
        est_values = {
            'candidates' : candidates.tolist(),
            'acq_funcs' : af_values.tolist(),
            'means': self.func_m.tolist(),
            'vars' : self.func_v.tolist()
        }
        debug('[WGP-{}_{}] using {} MCMC takes {:.2f} secs'.format(self.kernel_type, self.shaping_func, self.mcmc_iters, time.time() - s_t))

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
            debug('[WGP-{}_{}-{}] {}-fold CV performance: {:.4f}'.format(self.kernel_type, self.shaping_func, af, 
                                                                        len(scores), mean_score))
        return mean_score

    def build_model(self, comp, cand, errs):
        comp = np.array(comp)
        cand = np.array(cand)  
        errs = np.array(errs)
        
        logging.disable(logging.WARNING) # XXX:disable all warnings 
        # initialize GP model
        kern = None
        if self.kernel_type == "Matern52":
            kern = Matern52(comp.shape[1], ARD=True)
        elif self.kernel_type == "Matern32":
            kern = Matern32(comp.shape[1], ARD=True)
        elif self.kernel_type == 'Linear+Matern32':
            k1  = Linear(comp.shape[1],   ARD = False)
            k2  = Matern32(comp.shape[1], ARD = True)
            k2.lengthscale = np.std(comp, axis = 0)
            k2.variance    = 0.5
            k2.variance.set_prior(priors.Gamma(0.5, 1), warning = False)
            kern = k1 + k2
        else:
            raise NotImplementedError('Not supported kernel type: {}'.format(self.kernel_type))
        
        xmin    = np.zeros(comp.shape[1])
        xmax    = np.ones(comp.shape[1])
        errs = errs.reshape(-1, 1)
        warp_f  = KumarWarping(comp, Xmin = xmin, Xmax = xmax)

        if self.model:
            del self.model # XXX:Force to remove previous model
            self.model = None
        
        if self.warp:
            self.model = InputWarpedGP(comp, errs, kern, warping_function=warp_f)
            if self.kernel_type == 'Linear+Matern32':
                self.model.likelihood.variance.set_prior(priors.LogGaussian(-4.63, 0.5), warning = False)
        else:
            self.model = GPRegression(comp, errs, kern)

        # Training GP model
        self.model.optimize_restarts(max_iters=200, verbose=False, num_restarts = self.mcmc_iters)

        # Predict the marginal means and variances at candidates.
        mu, var = self.model.predict(cand)
        mu = mu.reshape(1, -1)
        var = var.reshape(1, -1)
        logging.disable(logging.NOTSET) # FIX:enable all loggings
        
        return mu, var
        
