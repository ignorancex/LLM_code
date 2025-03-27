import jax.numpy as jnp
from jax import random, jit
from sgmcmcjax.diffusions import sgld, welling_teh_schedule,sgnht,badodab,baoab
import torch
from tqdm.auto import tqdm
import numpy as np
from .optimisers import build_adam_lfi_optimizer
from .kernels import build_sgnht_lfi_kernel, build_psgld_lfi_kernel
from .gradient_estimation import build_gradient_estimation_fn
from ..mamba.mamba import run_MAMBA
from ..mamba.ksd import imq_KSD
from ..utils import GradientNANException

torch.set_default_dtype(torch.float64) #necessary to prevent nan errors!

import jax

class SGMCMC():
    """
    Stochastic-Gradient MCMC
    """

    def __init__(self, model, 
                 observations, 
                 joint_log_prob, 
                 transformer, 
                 build_kernel_func=build_sgnht_lfi_kernel, 
                 build_grad_est_func=build_gradient_estimation_fn, 
                 n_samples=110000, 
                 seed=42, 
                 w=1):
        self.model = model
        self.observations = observations
        self.joint_log_prob = joint_log_prob
        self.transformer = transformer
        self.build_kernel_func = build_kernel_func
        self.build_grad_est_func = build_grad_est_func 
        self.n_samples = n_samples
        self.seed = seed
        self.weight = w

        self.output = {}
        self.output['config'] = {'obs': observations, 'chain length': n_samples, 'w':w, 'seed': self.seed , 'param_dim':self.model.param_dim, 'init_params': None, 'step_size_seq':[], 'scores':[], 'transformer':self.transformer}

    def _mcmc_sample(self, dt):
        key = random.PRNGKey(self.seed) #Ensure within this function, determinism is fixed to seed
        self.model.seed = self.seed #Reset model seed

        samples_uncon = torch.zeros(size=(self.n_samples,self.model.param_dim)) 
        samples_grad = torch.zeros(size=(self.n_samples,self.model.param_dim)) 

        init_fn, my_kernel, get_params = self.build_kernel_func(dt, data=self.observations, joint_log_prob=self.joint_log_prob, grad_est_func=self.build_grad_est_func)

        key, subkey = jax.random.split(key)
        state = init_fn(subkey, self.output['config']['init_params'])
        for i in tqdm(range(self.n_samples)):
            key, subkey = jax.random.split(key)
            state = my_kernel(i, subkey, state)
            samples_uncon[i] = torch.from_numpy(np.asarray(get_params(state)))
            samples_grad[i] = torch.from_numpy(np.asarray(state.param_grad))

            if torch.isnan(samples_grad[i]).any():
                raise GradientNANException("Gradients are NAN")

        self.output['samples_uncon'] = samples_uncon
        self.output['samples_grad'] = samples_grad

    def _optim(self, dt, init_params):
        key = random.PRNGKey(self.seed) #Ensure within this function, determinism is fixed to seed
        self.model.seed = self.seed #Reset model seed

        if init_params is None:
            init_params = jax.numpy.zeros(self.model.param_dim)

        # To find centering value
        run_adam = build_adam_lfi_optimizer(dt=dt, data=self.observations, joint_log_prob=self.joint_log_prob, grad_est_func=self.build_grad_est_func)
        params_IC = run_adam(key, 250, init_params) #250 steps of adam 
        self.output['config']['init_params'] = params_IC
        print(f"Initial params: {params_IC}")

    def _mamba(self, R, log_dt_range):
        key = random.PRNGKey(self.seed) #Ensure within this function, determinism is fixed to seed
        self.model.seed = self.seed #Reset model seed

        error_fn=lambda x,y: imq_KSD(x, y) 
        # Multi-armed bandit for optimal step size selection
        build_kernel = lambda dt: self.build_kernel_func(dt, data=self.observations, joint_log_prob=self.joint_log_prob, grad_est_func=self.build_grad_est_func)

        grid_params = {'log_dt': log_dt_range}

        best_arm = run_MAMBA(key, build_kernel, error_fn ,R , self.output['config']['init_params'], 
                            grid_params=grid_params)

        print(best_arm.hyperparameters, best_arm.metric, best_arm.samples.shape)

        optimal_dt = best_arm.hyperparameters['dt']
        self.output['config']['step_size'] = optimal_dt
        self.output['config']['step_size_seq'].append(optimal_dt)
        self.output['config']['mamba_log_step_size_range'] = log_dt_range
        print(optimal_dt)

        return optimal_dt

    def sample(self, 
               init_params=None, 
               use_mamba=True, 
               mamba_log_step_size_range = -jax.numpy.arange(1., 8., 0.5), 
               R=10, 
               step_size=0.01, 
               use_optim=True, 
               optim_step_size=0.01):
        if use_optim:
            self._optim(dt=optim_step_size, init_params=init_params)
        else:
            self.output['config']['init_params'] = init_params

        if use_mamba:
            optimal_dt = self._mamba(R, mamba_log_step_size_range)
        else:
            optimal_dt = step_size
            self.output['config']['step_size'] = optimal_dt
            self.output['config']['step_size_seq'].append(optimal_dt)

        # Main sampling section
        completed = False
        while not completed:
            try:
                self._mcmc_sample(optimal_dt)
                completed = True
            except GradientNANException:
                optimal_dt = optimal_dt * 0.5
                print(f"using dt: {optimal_dt}")
                self.output['config']['step_size_seq'].append(optimal_dt)
                self.output['config']['step_size'] = optimal_dt
            finally:
                self.output['config']['scores'].append(self.model.scores.copy())
                self.model.scores = []

        return self.output