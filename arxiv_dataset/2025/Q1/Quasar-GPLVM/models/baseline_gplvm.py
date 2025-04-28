#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model class Base GPLVM - single set of hyperparameters across all dimensions 

"""

from models.latent_variable import PointLatentVariable, MAPLatentVariable
import torch
import numpy as np
from tqdm import trange
from gpytorch.mlls import VariationalELBO
from prettytable import PrettyTable
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.priors import NormalPrior
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel, RQKernel
from gpytorch.distributions import MultivariateNormal

class QuasarModel(ApproximateGP):
     def __init__(self, n, data_dim, latent_dim, n_inducing, latent_config='point', kernel_config='standard', spectra_dims=7):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations \tilde{Z_{d}} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
    
        # Define prior for Z
        Z_prior_mean = torch.zeros(self.n, latent_dim)  # shape: N x Q
    
        # Initialise Z
     
        Z_init = torch.nn.Parameter(torch.zeros(n, latent_dim))
          
        # LatentVariable configuration
        if latent_config == 'map':
            
            prior_z = NormalPrior(Z_prior_mean, torch.ones_like(Z_prior_mean))
            Z = MAPLatentVariable(n, latent_dim, Z_init, prior_z)
            #X = VariationalLatentVariable(self.n, data_dim, latent_dim, X_init, prior_x)
        
        elif latent_config == 'point':
        
            Z = PointLatentVariable(Z_init)
            
        super(QuasarModel, self).__init__(q_f)
        
        self.Z = Z
        
        # Kernel 
        
        self.mean_module = ConstantMean(batch_shape=self.batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
            
     def forward(self, Z):
        mean_z = self.mean_module(Z)
        covar_z = self.covar_module(Z)
        dist = MultivariateNormal(mean_z, covar_z)
        return dist
    
     def _get_batch_idx(self, batch_size):
            
         valid_indices = np.arange(self.n)
         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
         return np.sort(batch_indices)
     
     def get_trainable_param_names(self):
        
        ''' Prints a list of parameters (model + variational) which will be 
        learnt in the process of optimising the objective '''
        
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        
     def initialise_model_test(self, n_test, latent_dim):
        
        ## Initialise test model by resetting n and Z
        
        self.n = n_test
        
        Z_init_test = torch.nn.Parameter(torch.randn(self.n, latent_dim))

        self.Z.reset(Z_init_test)
        
        return self
    
     def reconstruct(self, Z):
           # Just decode from the Z that is passed
           # Returns a batch of multivariate-normals 
           y_pred = self(Z)
           return y_pred, y_pred.loc, y_pred.covariance_matrix

def predict_latent(test_model, Y_test, likelihood, lr=0.001, prior_z = None, steps = 2000, batch_size = 100):
     
         # Train for test Z variational params
         
         # Initialise fresh test optimizer 
         optimizer = torch.optim.Adam(test_model.Z.parameters(), lr=lr)
         elbo = VariationalELBO(likelihood, test_model, num_data=len(Y_test))
         
         print('---------------Learning variational parameters for test ------------------')
         for name, param in test_model.Z.named_parameters():
             print(name)
             
         loss_list = []
         iterator = trange(steps, leave=True)
         batch_size = len(Y_test) if len(Y_test) < 100 else 100
         for i in iterator: 
             batch_index = test_model._get_batch_idx(batch_size)
             optimizer.zero_grad()
             sample_batch = test_model.Z.Z[batch_index]  # a full sample returns latent Z across all N
             sample_batch.requires_grad_(True)
             
             output_batch = test_model(sample_batch)
             loss = -elbo(output_batch, Y_test[batch_index].T).sum()
             loss.requires_grad_(True)
             loss_list.append(loss.item())
             iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
             loss.backward()
             optimizer.step()
             
         return loss_list, test_model, test_model.Z
     
