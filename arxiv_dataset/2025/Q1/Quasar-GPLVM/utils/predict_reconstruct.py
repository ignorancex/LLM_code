#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for getting predictions and reconstructions

"""

import torch
import numpy as np


def predict_from_latents_baseline(Z_train, XY_train, model, likelihood, std_X, std_Y):
    
     batch_size = 100
     num_batches = Z_train.size(0) // batch_size + int(Z_train.size(0) % batch_size != 0)
    
     # List to store the outputs
     XY_train_recon_list = []
     Y_train_pred_sigma_list = []
     X_train_pred_sigma_list = []
     
     # Iterate over the Z_train in batches
     for i in range(num_batches):
         
         # Determine start and end indices of the batch
         start_idx = i * batch_size
         end_idx = min((i + 1) * batch_size, Z_train.size(0))
         
         # Get the current batch
         Z_train_batch = Z_train[start_idx:end_idx]
      
         # Reconstruct X and Y for the current batch
         XY_pred, XY_train_recon, XY_train_pred_covar = model.reconstruct(Z_train_batch)
                  
         diags = torch.Tensor(np.array([m.diag().sqrt().cpu().detach().numpy() for m in XY_train_pred_covar]).T) ## extracting diagonals per dimensions
         
         sigma_X = diags[::,0:-4]*std_X.cpu()
         sigma_Y = diags[:,-4::]*std_Y.cpu()
         
         del XY_train_pred_covar

         # Append the results to the list
         
         XY_train_recon_list.append(XY_train_recon.cpu().detach().T)
         
         X_train_pred_sigma_list.append(sigma_X)
         Y_train_pred_sigma_list.append(sigma_Y) 

     # Concatenate the results after the loop
     XY_train_recon_full = torch.cat(XY_train_recon_list, dim=0)
     
     X_train_recon_full = XY_train_recon_full[::,0:-4]
     Y_train_recon_full = XY_train_recon_full[:,-4::]
     
     Y_train_recon_sigma = torch.cat(Y_train_pred_sigma_list, dim=0)
     X_train_recon_sigma = torch.cat(X_train_pred_sigma_list, dim=0)

     return X_train_recon_full, Y_train_recon_full, X_train_recon_sigma, Y_train_recon_sigma, XY_pred


def predict_from_latents_shared(Z_train, X_train, Y_train, model, likelihood_spectra, likelihood_labels, std_X, std_Y):
    
     batch_size = 100
     num_batches = Z_train.size(0) // batch_size + int(Z_train.size(0) % batch_size != 0)
    
     # List to store the outputs
     Y_train_recon_list = []
     X_train_recon_list = []
     Y_train_pred_sigma_list = []
     X_train_pred_sigma_list = []
     
     # Iterate over the Z_train in batches
     for i in range(num_batches):
         
         # Determine start and end indices of the batch
         start_idx = i * batch_size
         end_idx = min((i + 1) * batch_size, Z_train.size(0))
         
         # Get the current batch
         Z_train_batch = Z_train[start_idx:end_idx]
      
         # Reconstruct X and Y for the current batch
         X_pred, X_train_recon, X_train_pred_covar = model.model_spectra.reconstruct(Z_train_batch)
         Y_pred, Y_train_recon, Y_train_pred_covar = model.model_labels.reconstruct(Z_train_batch)
         
         vars_X_noiseless = np.array([(m.diag()).cpu().detach().numpy() for m in X_train_pred_covar]).T ## extracting diagonals per dimensions
         vars_X_noisy = torch.Tensor(np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless]))
         sigma_X = torch.sqrt(vars_X_noisy)*std_X.cpu()
         
         vars_Y_noiseless_list = [m.diag().cpu().detach() for m in Y_train_pred_covar]
         vars_Y_noiseless = torch.cat(vars_Y_noiseless_list).reshape(len(Y_train_recon.T),4)
         vars_Y_noisy =  [m + likelihood_labels.noise_covar.noise.flatten().cpu().detach() for m in vars_Y_noiseless]
         sigma_Y = torch.sqrt(torch.Tensor(np.array([t.numpy() for t in vars_Y_noisy])))*std_Y.cpu()
         
         del X_train_pred_covar
         del Y_train_pred_covar

         # Append the results to the list
         
         Y_train_recon_list.append(Y_train_recon.cpu().detach().T)
         X_train_recon_list.append(X_train_recon.cpu().detach().T)
         
         Y_train_pred_sigma_list.append(sigma_X) 
         X_train_pred_sigma_list.append(sigma_Y)
    
     # Concatenate the results after the loop
     Y_train_recon_full = torch.cat(Y_train_recon_list, dim=0)
     X_train_recon_full = torch.cat(X_train_recon_list, dim=0)
     
     Y_train_recon_sigma = torch.cat(Y_train_pred_sigma_list, dim=0)
     X_train_recon_sigma = torch.cat(X_train_pred_sigma_list, dim=0)

     return X_train_recon_full, Y_train_recon_full, X_train_recon_sigma, Y_train_recon_sigma, X_pred, Y_pred
