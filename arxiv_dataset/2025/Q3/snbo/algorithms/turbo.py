# This module defines a class for the TuRBO algorithm and other
# helper functions used in the algorithm. Following code is adapted
# from the original source code available at https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_1.py

import gpytorch, torch, math
import numpy as np
from copy import deepcopy
from time import time
from utils import latin_hypercube

from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

class Turbo:

    def __init__(
        self, 
        f, 
        lb, 
        ub, 
        n_init, 
        max_evals,
        L_init=0.8,
        L_max=1.6,
        L_min=1/2**7,
        batch_size=1,
        verbose=True,
        max_cholesky_size=2000, 
        n_training_steps=100,
        device="cuda",
        dtype="float64"
    ):
        """
            Class for setting up and running the TuRBO-1 algorithm

            Parameters
            -----------
            f: objective function handle

            lb: lower bounds, np.ndarray, shape (d,)

            ub: upper bounds, np.ndarray, shape (d,)

            n_init: initialize number of points, int

            max_evals: total evaluation budget, int

            L_init: initial length of the trust region, float, default=0.8

            L_max: maximum length of the trust region, float, default=1.6

            L_min: minimum length of the trust region, float, default=1/2**7

            batch_size: number of points in each batch (q), int, default=1

            verbose: print information about iteration progress, bool, default=True

            max_cholesky_size: maximum number of training points to use Cholesky, int, default=2000

            n_training_steps: number of training steps for the GP model, int, default=100
            
            device: device to use for GP fitting ("cpu" or "cuda"), str, default="cpu"
            
            dtype: dtype to use for GP fitting ("float32" or "float64"), str, default="float64"

            Note: This code assumes you are solving a minimization problem
        """
        
        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert isinstance(L_init, float)
        assert isinstance(L_min, float)
        assert isinstance(L_max, float)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
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
        self.verbose = verbose
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Tolerances and counters
        self.n_evals = 0
        self.n_cand = min(5000, max(2000, 200 * self.dim))
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3

        # Trust region parameters
        self.length_min = L_min
        self.length_max = L_max
        self.length_init = L_init

        # Initialize empty arrays for storing the history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.train_time = np.array([])

        # Device and dtype for GPyTorch
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        
        # Initialize parameters
        self._restart()

    def _restart(self):
        """
            Method to reset the hyper-rectangle parameters
        """

        # (_X,_fX) is the training dataset for a TR
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        """
            Method to expand/shrink the length of the trust region

            Parameters
            ----------
            fX_next: numpy array consisting of y values for next set of infill points
        """

        # relative difference between the current best and next best should be atleast 1e-3
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """
            Method to generate candidate set to identify the next batch of samples

            Parameters
            -----------
            X: input values for training, np.ndarray, this should be normalized

            fx: input values for training, np.ndarray, this should NOT be standarized

            length: base side length which is later modified based on GP lengthscale, int

            n_training_steps: number of iterations for GP training, int

            hypers: dictionary containing hyperparameters for the GP models, dict
        """

        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values
        mu, sigma = np.mean(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        device, dtype = self.device, self.dtype

        # Use CG + Lanczos for training if sample size increases beyond a limit
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            
            X_torch = torch.tensor(X).to(device=self.device, dtype=self.dtype)
            y_torch = torch.tensor(fX).to(device=self.device, dtype=self.dtype)

            t1 = time()

            # train GP model
            gp = train(
                xtrain=X_torch, 
                ytrain=y_torch,
                use_ard=True, 
                max_iters=n_training_steps, 
                hypers=hypers
            )

            self.train_time = np.append(self.train_time, time()-t1)

        # center of the TR
        x_center = X[fX.argmin().item(), :][None, :]

        # weight for each dimension based on lengthscale
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0/self.dim))  # We now have weights.prod() = 1

        # lower and upper bound of the TR
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobol sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = torch.quasirandom.SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0) # If dim is less than 20, then perturb prob will be 1
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb # 2D bool array to identify which dimensions to perturb for each candidate point
        ind = np.where(np.sum(mask, axis=1) == 0)[0] # find candidate points which are not perturbed
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1 # perturb atleast 1 dimension

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # unstandardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """
            Select next evaluation points from the given candidate set

            Parameters
            ----------
            X_cand: x values of the points in the candidate set, np.ndarray

            y_cand: y values of the points in the candidate set, np.ndarray 
        """

        # Initialize array to store candidate points
        X_next = np.ones((self.batch_size, self.dim))

        # Pick the best points and make sure we never pick it again
        for i in range(self.batch_size):
            
            idxdbest = np.argmin(y_cand[:, i])

            X_next[i, :] = deepcopy(X_cand[idxdbest, :])

            y_cand[idxdbest, :] = np.inf # to ensure this x is never picked again

        return X_next

    def optimize(self):
        """
            Method to run the optimization loop
        """

        while self.n_evals < self.max_evals:

            if len(self._fX) > 0 and self.verbose:
                print(f"{self.n_evals}) Restarting with fbest = {self._fX.min():.4}", flush=True)

            # Initialize parameters
            self._restart()

            # Generate and evalute initial design points
            X_init = latin_hypercube(self.n_init, self.dim)
            fX_init = self.f(X_init)
    
            # Update budget 
            self.n_evals += self.n_init

            # Set initial data for this TR - (self._X,self._fX) is training dataset for this TR
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history - self.X, self.fX are global history variables
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                print(f"Starting from fbest = {self._fX.min():.4}", flush=True)

            # Thompson sample to add new batch of infill points
            while self.n_evals < self.max_evals and self.length >= self.length_min:

                X = deepcopy(self._X)

                # compute 
                fX = deepcopy(self._fX).ravel()

                # create candidate set
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )

                # select best candidates to be the next batch
                X_next = self._select_candidates(X_cand, y_cand)

                # evaluate batch
                fX_next = self.f(X_next)

                # update trust region
                self._adjust_length(fX_next)

                # update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next.min() < self.fX.min():
                    print(f"{self.n_evals}) New best: {fX_next.min():.4}", flush=True)

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))

class GP(ExactGP):

    def __init__(self, xtrain, ytrain, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims=None):
        """
            Class for defining a GP model

            Parameters
            ----------
            xtrain: torch tensor representing input values

            ytrain: torch tensor representing output values

            likelihood: likelihood model used while computing the output

            lengthscale_constraint: constraint on the lengthscale of the GP model

            outputscale_constraint: constraint on the outputscale of the GP model

            ard_dims: parameter to set separate lengthscale for each dimension, default=None
        """
        
        super(GP, self).__init__(xtrain, ytrain, likelihood)

        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        
        mean = self.mean_module(x) # computes the mean vector for given x
        covar = self.covar_module(x) # computes the covariance matrix for given x

        return MultivariateNormal(mean, covar)

def train(xtrain, ytrain, max_iters, use_ard=True, hypers={}):
    """
        Method for training the GP model

        Note: xtrain should be normalized to [0, 1]^d and ytrain should be standardized to zero mean and unit variance
    """

    assert xtrain.ndim == 2 and ytrain.ndim == 1 and xtrain.shape[0] == ytrain.shape[0]

    # Create hyperparameter bounds
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, torch.sqrt(xtrain.shape[1]).item())

    outputscale_constraint = Interval(0.05, 20.0)

    likelihood = GaussianLikelihood(noise_constraint=Interval(5e-4, 0.2)).to(xtrain)

    # Create GP model
    model = GP(
        xtrain=xtrain,
        ytrain=ytrain,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=xtrain.shape[1] if use_ard else None,
    ).to(xtrain)

    model.train()
    likelihood.train()

    # Loss
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    for _ in range(max_iters):
        optimizer.zero_grad()
        output = model(xtrain)
        loss = -mll(output, ytrain)
        loss.backward()
        optimizer.step()

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model
