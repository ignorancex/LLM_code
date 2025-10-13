# This module defines a class for the BO based on IBNN. It has only three hyper-parameters: number of layers,
# variance of the weights, variance of the bias terms. Following sources are used for below code:
# 1. Original source code: https://github.com/yucenli/bnn-bo/tree/main
# 2. BoTorch docs: https://botorch.org/docs/tutorials/ibnn_bo/#infinite-width-bayesian-neural-networks-for-bayesian-optimization

import gpytorch, torch, warnings
from torch.quasirandom import SobolEngine
import numpy as np
from time import time

from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel

from botorch.models import SingleTaskGP
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.generation import MaxPosteriorSampling
from botorch.sampling.stochastic_samplers import StochasticSampler

from utils import latin_hypercube

warnings.filterwarnings('ignore')

class IBNN:

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        num_layers=3, # from paper
        acq_func="logei",
        verbose=True,
        max_cholesky_size=2000,
        restarts=5, # from paper
        raw_samples=16, # from paper
        device="cuda",
        dtype="float64"
    ):
        """
            Class for setting up and running the BO-IBNN algorithm
            Acquisition function is either EI, LogEI, or Thompson Sampling

            Note: IBNN is equivalent to a GP with a specific kernel function,
            so the code looks very similar to a standard BO loop
            
            Parameters
            -----------
            f: objective function handle, func should accept input in [0,1]

            lb: lower bounds, np.ndarray, shape (d,)

            ub: upper bounds, np.ndarray, shape (d,)

            n_init: initialize number of points, int

            max_evals: total evaluation budget, int

            batch_size: number of points in each batch (q), int, default=1

            num_layers: number of layers or depth of the network, int, default=3

            acq_func: acquisition function to be used, default="logei"

            verbose: print information about iteration progress, bool, default=True

            max_cholesky_size: maximum number of training points to use Cholesky, int, default=2000
            
            device: device to use for GP fitting ("cpu" or "cuda"), str, default="cuda"
            
            dtype: dtype to use for GP fitting ("float32" or "float64"), str, default="float64"

            Note: This code assumes you are solving a maximization problem
        """
    
        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(num_layers, int) and num_layers > 0
        assert isinstance(acq_func, str) and acq_func.lower() in ["logei", "ei", "ts"]
        assert isinstance(verbose, bool)
        assert max_cholesky_size >= 0
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"

        # Save objective function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.acq_func = acq_func.lower()
        self.verbose = verbose
        self.max_cholesky_size = max_cholesky_size
        self.restarts = restarts
        self.raw_samples = raw_samples
        self.n_evals = 0

        # Initialize empty arrays for storing the history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.train_time = np.array([])
        self.acq_func_opt_time = np.array([])

        # Device and dtype for GPyTorch
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")

    def optimize(self):
        """
            Method to run the optimization loop
        """

        self.X = latin_hypercube(self.n_init, self.dim) # initial doe in [0,1]^d
        self.fX = self.f(self.X)

        # Update budget
        self.n_evals += self.n_init

        if self.verbose:
            fbest = self.fX.max()
            print(f"Starting from fbest = {fbest:.4}", flush=True)

        device, dtype = self.device, self.dtype

        while self.n_evals < self.max_evals:

            ibnn_kernel = InfiniteWidthBNNKernel(self.num_layers, device=device)
            ibnn_kernel.weight_var = 10.0 # from paper
            ibnn_kernel.bias_var = 1.6 # from paper
            ibnn_kernel = ScaleKernel(ibnn_kernel, device=device)

            # uses constant mean and IBNN kernel
            model = SingleTaskGP(
                torch.from_numpy(self.X),
                torch.from_numpy(self.fX),
                covar_module=ibnn_kernel,
                outcome_transform=Standardize(m=1),
                input_transform=None # inputs are already in [0,1]^d
            ).to(device=device, dtype=dtype)

            mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device=device, dtype=dtype)

            # Use CG + Lanczos for training if sample size increases beyond a limit
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):

                t1 = time()

                fit_gpytorch_mll(mll, max_attempts=10) # fits using L-BFGS-B and jacobian through backprop

                self.train_time = np.append(self.train_time, time()-t1)

                t1 = time()

                # Next infill points
                if self.acq_func in ["logei", "ei"]:

                    sampler = StochasticSampler(sample_shape=torch.Size([128])) # from original source code

                    if self.acq_func == "logei":
                        acq_func = qLogExpectedImprovement(model, self.fX.max(), sampler=sampler)
                    elif self.acq_func == "ei":
                        acq_func = qExpectedImprovement(model, self.fX.max(), sampler=sampler)

                    X_next, _ = optimize_acqf(
                        acq_func,
                        bounds=torch.stack(
                            [
                                torch.zeros(self.dim, dtype=dtype, device=device),
                                torch.ones(self.dim, dtype=dtype, device=device),
                            ]
                        ),
                        q=self.batch_size,
                        num_restarts=self.restarts,
                        raw_samples=self.raw_samples
                    )

                elif self.acq_func == "ts":

                    n_candidates = min(5000, max(2000, 200 * self.dim))

                    sobol = SobolEngine(self.dim, scramble=True)

                    X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)

                    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)

                    with torch.no_grad():  # no need for gradients when using TS
                        X_next = thompson_sampling(X_cand, num_samples=self.batch_size)

                self.acq_func_opt_time = np.append(self.acq_func_opt_time, time())

            del acq_func, model

            torch.cuda.empty_cache()

            X_next = X_next.numpy(force=True)

            fX_next = self.f(X_next)

            if self.verbose and fX_next.max() > self.fX.max():
                n_evals, fbest = self.n_evals, fX_next.max()
                print(f"{n_evals}) New best: {fbest:.4}", flush=True)

            # Append data to the global history
            self.n_evals += self.batch_size
            self.X = np.vstack((self.X, X_next))
            self.fX = np.vstack((self.fX, fX_next))
