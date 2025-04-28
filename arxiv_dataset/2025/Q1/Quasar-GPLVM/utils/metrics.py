#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Negative test log-likelihood and sq. root mean squared error

"""

import torch
import numpy as np
from torch.distributions import Normal

def mean_absolute_error_spectra(X_train_orig, X_train_recon_orig):
    
    return torch.nanmean(torch.abs(X_train_orig - X_train_recon_orig))

def mean_absolute_error_labels(Y_train_orig, Y_train_recon_orig):
    
    return torch.nanmean(torch.abs(Y_train_orig - Y_train_recon_orig), dim=0)

def nll_lum_bhm_edd(Y_test_pred, Y_test, std_Y, likelihood_labels):
    
      mask = ~torch.isnan(Y_test).any(dim=1)
      Y_test = Y_test[mask]
      skip_ids = torch.nonzero(~mask, as_tuple=False).squeeze()
      lpd_n = []
      
      for i in range(len(Y_test)):
          
          if i not in skip_ids:
              
              # Extract the mean and covariance for the marginal distribution
              marginal_mean = Y_test_pred.loc.T[i] 
              marginal_var = Y_test_pred.covariance_matrix[:,i,i]
    
              # Create the marginal distribution
              Y_pred_marginal = Normal(marginal_mean, torch.sqrt(marginal_var + likelihood_labels.noise_covar.noise.flatten()))
              lpd = Y_pred_marginal.log_prob(Y_test[i]).cpu().detach()
              lpd_n.append(lpd)
            
          else:
                continue;
        
      avg_lpd_rescaled = lpd.mean(dim=0) - torch.log(std_Y)
      
      return -avg_lpd_rescaled
  
def rmse(X_test_orig, X_test_recon_orig):
    
    return torch.sqrt(torch.mean(torch.Tensor([np.nanmean(np.square(X_test_orig - X_test_recon_orig))])))

def rmse_lum_bhm_edd(Y_test_orig, Y_test_recon_orig):
    
    rmse_labels = []
    for i in [1,2,3]:
        rmse_labels.append(torch.sqrt(torch.mean(torch.Tensor([np.nanmean(np.square(Y_test_orig[:,i] - Y_test_recon_orig[:,i]))]))))
    return rmse_labels