# This module defines class and function for the SNBO algorithm

import torch
import numpy as np
from time import time
from utils import latin_hypercube

class SNBO:

    def __init__(
        self, 
        f, 
        lb, 
        ub, 
        n_init, 
        max_evals,
        neurons,
        act_funcs,
        r_init=1.6,
        r_max=1.6,
        r_min=0.025,
        batch_size=1,
        lr=1e-3,
        max_epochs=3000,
        early_stop_tol=1e-3,
        verbose=True,
        device="cuda",
        dtype="float32",
        initial_x=None,
        initial_y=None
    ):
        """
            Class for setting up and running SNBO algorithm

            Parameters
            ----------
            f: objective function handle

            lb: lower bounds, np.ndarray, shape (d,)

            ub: upper bounds, np.ndarray, shape (d,)

            n_init: initialize number of points, int

            max_evals: total evaluation budget, int

            neurons: number of neurons in each layer, list

            act_funcs: activation function in each layer, list

            r_init: initial perturbation range, float, default=1.6

            r_max: maximum perturbation range, float, default=1.6

            r_min: minimum perturbation range, float, default=0.025

            batch_size: number of infills to add (q), int, default=1
            
            lr: learning rate for training the NN model, float, default=1e-3

            max_epochs: maximum number of epochs for training the NN model, int, default=3000

            early_stop_tol: early stopping tolerance for NN training, float, default=1e-3

            verbose: print information about iteration progress, bool, default=True
            
            device: device to use for NN training and inference ("cpu" or "cuda"), str, default="cuda"
            
            dtype: dtype to use for NN training and inference ("float32" or "float64"), str, default="float32"

            initial_x: 2D numpy array of shape (samples,dim) containing starting doe, np.ndarry, default=None

            initial_y: 2D numpy array of shape (samples,1) containing y values for corresponding initial doe, np.ndarry, default=None

            Note: This code assumes you are solving a minimization problem
        """

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert isinstance(r_init, float)
        assert isinstance(r_min, float)
        assert isinstance(r_max, float)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(verbose, bool)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if initial_x is not None or initial_y is not None:
            assert isinstance(initial_x,np.ndarray) and initial_x.ndim == 2
            assert isinstance(initial_y,np.ndarray) and initial_y.ndim == 2
            assert initial_x.shape[0] == initial_y.shape[0]
            assert initial_y.shape[1] == 1

        # Save objective function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub
        self.initial_x = initial_x
        self.initial_y = initial_y

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose

        # Tolerances and counters
        self.n_evals = 0
        self.n_explore = self.batch_size * self.dim
        self.failtol = np.ceil(self.dim/self.batch_size)
        self.succtol = 3

        # Perturbation range parameters
        self.r_min = r_min
        self.r_max = r_max
        self.r_init = r_init

        # NN settings
        self.neurons = neurons
        self.act_funcs = act_funcs
        self.lr = lr
        self.early_stop_tol = early_stop_tol
        self.max_epochs = max_epochs
        self.neurons = neurons
        self.act_funcs = act_funcs

        # Initialize empty arrays for storing the history
        self.xhistory = np.zeros((0, self.dim))
        self.yhistory = np.zeros((0, 1))
        self.rhistory = np.zeros((0, 1))
        self.train_time = np.array([])

        # Device and dtype for NN model
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")

    def optimize(self):
        """
            Method to run the optimization loop
        """

        args = {
            "device": self.device,
            "dtype": self.dtype
        }

        while self.n_evals < self.max_evals:

            if len(self.yhistory) > 0 and self.verbose:
                    print(f"{self.n_evals}) Restarting with fbest = {self.yhistory.min():.4}", flush=True)

            r = self.r_init
            fail = 0
            succ = 0

            # Generate initial samples
            if self.initial_x is None and self.initial_y is None:
                x = latin_hypercube(self.n_init, self.dim)
                y = self.f(x)
            else:
                x = self.initial_x
                y = self.initial_y

            self.n_evals += self.n_init

            # Append history
            self.xhistory = np.vstack((self.xhistory, x))
            self.yhistory = np.vstack((self.yhistory, y))

            # Training dataset
            xtrain = torch.from_numpy(x).to(**args)
            ytrain = torch.from_numpy(y).to(**args)

            # Instantiate the pred model
            model = Network(self.dim, 1, self.neurons, self.act_funcs).to(**args)
            model.apply(init_weights)

            del x, y

            if self.verbose:
                print(f"Starting from fbest = {ytrain.min().item():.4}", flush=True)

            while self.n_evals < self.max_evals and r >= self.r_min:

                # Minimum in the current dataset
                index = torch.argmin(ytrain)
                ybest = ytrain[index,0].item()
                xbest = xtrain[index,:]

                ############## Standardize the data
                xtransform = Standardize(xtrain)
                ytransform = Standardize(ytrain)

                xtrain = xtransform.transform(xtrain)
                ytrain = ytransform.transform(ytrain)

                ############## Train the model
                t1 = time()
                train(model, xtrain, ytrain, self.max_epochs, self.lr, tol=self.early_stop_tol)
                self.train_time = np.append(self.train_time, time() - t1)

                # Inverse transform
                xtrain = xtransform.inverse_transform(xtrain)
                ytrain = ytransform.inverse_transform(ytrain)

                ############## Generate candidates
                self.rhistory = np.vstack((self.rhistory,r))
                x_explore = generate_exploration_points(self.n_explore,r,xbest,self.device).to(**args)

                # Predict
                x_explore = xtransform.transform(x_explore)
                y_explore_pred = ytransform.inverse_transform(model(x_explore))
                x_explore = xtransform.inverse_transform( x_explore )

                # Find best q exploration points
                idx_exploit = torch.topk(y_explore_pred, k=self.batch_size, dim=0, largest=False, sorted=False)[1].reshape(-1,)
                x_infill = x_explore[idx_exploit].numpy(force=True)
                y_infill = self.f(x_infill)

                self.n_evals += self.batch_size

                ############## Update
                
                if np.min(y_infill) < ybest - 1e-3 * np.fabs(ybest):
                    succ += 1
                    fail = 0
                else:
                    fail += 1
                    succ = 0

                if succ == self.succtol:
                    r = min([2.0 * r, self.r_max])
                    succ = 0
                    fail = 0
                elif fail == self.failtol:
                    r = r/2.0
                    fail = 0
                    succ = 0

                if self.verbose and y_infill.min() < self.yhistory.min():
                    print(f"{self.n_evals}) New best: {y_infill.min():.4}", flush=True)

                # Appending infill points
                xtrain = torch.vstack(( xtrain, torch.from_numpy(x_infill).to(**args) ))
                ytrain = torch.vstack(( ytrain, torch.from_numpy(y_infill).to(**args) ))

                # Appending history
                self.xhistory = np.vstack(( self.xhistory, x_infill.reshape(-1,self.dim) ))
                self.yhistory = np.vstack(( self.yhistory, y_infill.reshape(-1,1) ))

            del model, xtrain, ytrain

class Network(torch.nn.Module):

    def __init__(self, in_dim, out_dim, neurons, act):
        """
            Class for defining a simple fully connected network

            in_dim: input dimension, int

            out_dim: output dimension, int

            neurons: list representing number of neurons in each layer

            act: list representing activation function to be used in each layer
        """

        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.num_layers = len(neurons)

        for i in range(self.num_layers):

            # Layer
            if len(self.layers) == 0:
                self.layers.append(torch.nn.Linear(in_dim, neurons[i]))
            else:
                self.layers.append(torch.nn.Linear(neurons[i-1], neurons[i]))

            # Activation
            act_func = getattr(torch.nn, act[i])
            self.acts.append(act_func())

        self.output = torch.nn.Linear(neurons[i], out_dim)

    def forward(self, x):

        for i in range(self.num_layers):
            x = self.acts[i](self.layers[i](x))

        output = self.output(x)

        return output

class Standardize():

    def __init__(self, x):
        """
            Class for standardizing the tensor data
        """

        self.x = torch.std(x, dim=0, keepdim=True)
        self.mean = torch.mean(x, dim=0, keepdim=True)

    def transform(self, x):

        return (x - self.mean)/self.x

    def inverse_transform(self, x):

        return x*self.x + self.mean

def init_weights(m):
    """
        Function for initializing the weights using He initialization
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)

def ptp(t, axis):
    """
        Method for computing the range of a tensor
    """

    return t.max(axis).values - t.min(axis).values

def train(model, xtrain, ytrain, epochs, lr, tol=1e-2):
    """
        Function for training a neural network model
        
        Parameters
        ----------
        model: neural network to be trained

        xtrain: torch.Tensor, training data for x, shape (n_samples, n_features)

        ytrain: torch.Tensor, training data for y, shape (n_samples, n_outputs)

        epochs: int, number of epochs for training

        lr: float, learning rate for the optimizer

        tol: float, tolerance for the training loss, default=1e-2
    """

    # Define the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):

        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        ypred = model(xtrain)

        # Compute loss
        loss = loss_fn(ypred, ytrain)

        if torch.sqrt(loss) / ptp(ytrain, 0) < tol:
            break

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    return epoch

def generate_exploration_points(n_explore, r, xbest, device):
    """
        Method to generate exploration points using FSSF-fr method

        Parameters
        ----------
        dim: dimension of the problem, int

        n_explore: number of exploration to be generated, int

        r: perturbation range, float

        xbest: 1d tensor array representing best point around which perturbation will be centered

        device: device on which tensor will created

        Returns
        -------
        x_explore: tensor array containing the generated points
    """

    assert isinstance(xbest, torch.Tensor)

    args = {
        "dtype": torch.float64, # required for accuracy
        "device": device
    }

    dim = xbest.shape[0]
    n_cand = 1000*dim + 2*n_explore # number of candidates
    pert_prob = 1/dim**0.5

    mask = torch.rand((n_cand,dim)).to(**args) <= pert_prob
    idx = torch.where(torch.sum(mask,dim=1) == 0)[0] # find points where there is no pertubation
    idx_dim = torch.randint(0,dim,size=(len(idx),),device=device)
    mask[idx,idx_dim] = 1 # perturb atleast one dimension

    perturbation = torch.distributions.Uniform(-r/2,r/2).sample(torch.Size([n_cand,dim])).to(**args)

    # create candidate set
    x_cand = xbest.to(**args) * torch.ones((n_cand, dim)).to(**args)
    x_cand[mask] = x_cand[mask] + perturbation[mask]

    # succesive reflection to handle bounds
    while torch.any(x_cand>1):
        x_cand[x_cand>1] = x_cand[x_cand>1] - 2*(x_cand[x_cand>1] - 1) # mirror about x=1
        x_cand = torch.abs(x_cand) # mirror about x=0

    # compute distance metric for each candidate point
    D = torch.hstack(( x_cand, 1 - x_cand ))
    D = 2*(2*dim)**0.5*torch.min(D, dim=1)[0].reshape(-1,1)

    x_explore = torch.tensor([]).reshape(-1,dim).to(**args)

    # select x_explore from candidate set
    for i in range(n_explore):
        
        new_point_index = torch.argmax(D)

        # Append new point
        x_explore = torch.vstack(( x_explore, x_cand[new_point_index,:] ))

        # New distance
        Dnew = torch.cdist(x_cand, x_explore[-1,:].reshape(1,-1))

        D = torch.minimum(D, Dnew)

    return x_explore
