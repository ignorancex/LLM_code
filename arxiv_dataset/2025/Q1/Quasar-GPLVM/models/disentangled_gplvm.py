#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disentangled GPLVM - each scientific label with individual hyperparameters.

"""
from models.latent_variable import PointLatentVariable, MAPLatentVariable
import torch
import numpy as np
from tqdm import trange
from prettytable import PrettyTable
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

class QuasarModel(ApproximateGP):
     def __init__(self, n: int, data_dim: int, latent_dim: int, n_inducing: int, inducing_inputs):
         
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
        
        mean_z = self.mean_module(Z)
        covar_z = self.covar_module(Z)
        dist = MultivariateNormal(mean_z, covar_z)
        return dist
    
     def reconstruct(self, Z):
           # Just decode from the Z that is passed
           # Returns a batch of multivariate-normals 
           y_pred = self(Z)
           return y_pred, y_pred.loc, y_pred.covariance_matrix

class DisentangledGPLVM(gpytorch.Module):
    
     def __init__(self, n, spectra_dim, joint_latent_dim, n_inducing, latent_config='point'):
         
        super(DisentangledGPLVM, self).__init__()

        self.n = n
        self.joint_latent_dim = joint_latent_dim
        
        # Define prior for Z

        Z_prior_mean = torch.zeros(self.n, joint_latent_dim)  # shape: N x Q
        Z_init = torch.nn.Parameter(torch.zeros(n, joint_latent_dim))
          
        # Latent Variable configuration
        
        if latent_config == 'map':
            prior_z = NormalPrior(Z_prior_mean, torch.ones_like(Z_prior_mean))
            Z = MAPLatentVariable(n, joint_latent_dim, Z_init, prior_z)
        
        elif latent_config == 'point':
        
            Z = PointLatentVariable(Z_init)
            
        self.Z = Z
        self.inducing_inputs = torch.nn.Parameter(torch.randn(n_inducing, joint_latent_dim))
        
        self.model_spectra = QuasarModel(n, spectra_dim, joint_latent_dim, n_inducing, self.inducing_inputs)
        self.model_lumin = QuasarModel(n, 1, joint_latent_dim, n_inducing, self.inducing_inputs)
        self.model_bhm = QuasarModel(n, 1, joint_latent_dim, n_inducing, self.inducing_inputs)
        self.model_edd = QuasarModel(n, 1, joint_latent_dim, n_inducing, self.inducing_inputs)
        
        #for key in component_gplvms.keys():
        #    model_name = 'model_' + key
        #    setattr(self, model_name) = BaseGPLVM(self.Z, n, component_gplvm[key], latent_dim, n_inducing, self.inducing_inputs)
        
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
        
        ## Initialise test models for all the internal models
        
        self.n = n_test
        
        self.model_spectra.n = n_test
        self.model_lumin.n = n_test
        self.model_bhm.n = n_test
        self.model_edd.n = n_test

        Z_init_test = torch.nn.Parameter(torch.randn(self.n, latent_dim))

        self.Z.reset(Z_init_test)

        return self
    
def predict_joint_latent(test_model, X_test, Y_test, likelihood_spectra, likelihood_lumin, likelihood_bhm, likelihood_edd, lr=0.001, prior_z = None, steps = 2000, batch_size = 100):
   
    # Initialise a new test optimizer with just the test model latents
    
    test_optimizer = torch.optim.Adam(test_model.Z.parameters(), lr=lr)
    
    mll_spectra = VariationalELBO(likelihood_spectra, test_model.model_spectra, num_data=test_model.n)
    mll_lumin = VariationalELBO(likelihood_lumin, test_model.model_lumin, num_data=test_model.n)
    mll_bhm = VariationalELBO(likelihood_bhm, test_model.model_bhm, num_data=test_model.n)
    mll_edd = VariationalELBO(likelihood_edd, test_model.model_edd, num_data=test_model.n)

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
           sample = test_model.Z.Z  # a full sample returns latent Z across all N
           sample_batch = sample[batch_index]
           
           ### Getting the output of the two groups of GPs
           
           output_spectra = test_model.model_spectra(sample_batch)
           output_lumin = test_model.model_lumin(sample_batch)
           output_bhm = test_model.model_bhm(sample_batch)
           output_edd = test_model.model_edd(sample_batch)

           ### Adding together the ELBO losses 
           
           if X_test is not None:
               joint_loss += -mll_spectra(output_spectra, X_test[batch_index].T).sum() 
           
           if Y_test is not None:
               
               joint_loss += -mll_lumin(output_lumin, Y_test[batch_index].T[0]).sum()
               joint_loss += -mll_bhm(output_bhm, Y_test[batch_index].T[1]).sum()
               joint_loss += -mll_edd(output_edd, Y_test[batch_index].T[2]).sum()
               
           loss_list.append(joint_loss.item())
           iterator.set_description('Loss: ' + str(float(np.round(joint_loss.item(),2))) + ", iter no: " + str(i))
           joint_loss.backward()
           test_optimizer.step()
        
    return loss_list, test_model, test_model.Z