#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared GPLVM Model classes 

"""
from models.latent_variable import PointLatentVariable, MAPLatentVariable, GaussianLatentVariable
import torch
import numpy as np
from tqdm import trange
from prettytable import PrettyTable
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

class QuasarModel(ApproximateGP):
     def __init__(self, Z, n, data_dim, latent_dim, n_inducing, inducing_inputs):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
    
        # Sparse Variational Formulation
        
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, inducing_inputs, q_u, learn_inducing_locations=False)
    
        super(QuasarModel, self).__init__(q_f)
        
        # Kernel 
        self.shared_inducing_inputs = inducing_inputs

        self.mean_module = ConstantMean(batch_shape=self.batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
                   
     def forward(self, Z):
         
        self.variational_strategy.inducing_points = self.shared_inducing_inputs
        
        #self.variational_strategy = self.variational_strategy.to(device)
        
        mean_z = self.mean_module(Z)
        covar_z = self.covar_module(Z)
        dist = MultivariateNormal(mean_z, covar_z)
        return dist
    
     def reconstruct(self, Z):
            # Just decode from the Z that is passed
            # Returns a batch of multivariate-normals 
            y_pred = self(Z)
            return y_pred, y_pred.loc, y_pred.covariance_matrix

class SharedGPLVM(gpytorch.Module):
    
     def __init__(self, n, spectra_dim, label_dim, latent_dim, n_inducing, latent_config='point', kernel_config='standard'):
         
        super(SharedGPLVM, self).__init__()

        self.n = n
        self.latent_dim = latent_dim
        self.data_dim = spectra_dim + label_dim
        
        # Define prior for Z

        Z_prior_mean = torch.zeros(self.n, latent_dim)  # shape: N x Q
        Z_init = torch.nn.Parameter(torch.zeros(n, latent_dim))
          
        # LatentVariable configuration
        
        if latent_config == 'map':
            
            prior_z = NormalPrior(Z_prior_mean, torch.ones_like(Z_prior_mean))
            Z = MAPLatentVariable(n, latent_dim, Z_init, prior_z)
        
        elif latent_config == 'point':
        
            Z = PointLatentVariable(Z_init)
            
        self.Z = Z
        self.inducing_inputs = torch.nn.Parameter(torch.randn(n_inducing, latent_dim))#.to(device)
        self.model_spectra = QuasarModel(self.Z, n, spectra_dim, latent_dim, n_inducing, self.inducing_inputs)
        self.model_labels = QuasarModel(self.Z, n, label_dim, latent_dim, n_inducing, self.inducing_inputs)
        
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
        
        ## Initialise test models for both the internal models: spectra and labels
        self.n = n_test
        
        self.model_spectra.n = n_test
        self.model_labels.n = n_test
        
        Z_init_test = torch.nn.Parameter(torch.randn(self.n, latent_dim))

        #self.model_spectra.Z.reset(Z_init_test)
        #self.model_labels.Z.reset(Z_init_test)
        self.Z.reset(Z_init_test)
        
        return self
    
def predict_joint_latent(test_model, X_test, Y_test, likelihood_spectra, likelihood_labels, lr=0.001, prior_z = None, steps = 2000, batch_size = 100):
   
    # Initialise a new test optimizer with just the test model latents
    
    test_optimizer = torch.optim.Adam(test_model.Z.parameters(), lr=lr)
    
    mll_spectra = VariationalELBO(likelihood_spectra, test_model.model_spectra, num_data=test_model.n)
    mll_labels = VariationalELBO(likelihood_labels, test_model.model_labels, num_data=test_model.n)
    
    print('---------------Learning variational parameters for test ------------------')
    
    for name, param in test_model.Z.named_parameters():
        print(name)
        
    loss_list = []
    iterator = trange(steps, leave=True)
    batch_size = batch_size
    
    for i in iterator: 
        
           joint_loss = 0.0
           batch_index = test_model._get_batch_idx(batch_size)
           test_optimizer.zero_grad()
           sample_batch = test_model.Z.Z[batch_index]  # a full sample returns latent Z across all N
           #sample_batch = sample[batch_index]
           
           ### Getting the output of the two groups of GPs
           
           output_spectra = test_model.model_spectra(sample_batch)
           output_labels = test_model.model_labels(sample_batch)
           
           ### Adding together the ELBO losses 
           
           if X_test is not None:
               joint_loss += -mll_spectra(output_spectra, X_test[batch_index].T).sum() 
               
           if Y_test is not None:
               joint_loss += -mll_labels(output_labels, Y_test[batch_index].T).sum()
               
           loss_list.append(joint_loss.item())
           iterator.set_description('Loss: ' + str(float(np.round(joint_loss.item(),2))) + ", iter no: " + str(i))
           joint_loss.backward()
           test_optimizer.step()
        
    return loss_list, test_model, test_model.Z
