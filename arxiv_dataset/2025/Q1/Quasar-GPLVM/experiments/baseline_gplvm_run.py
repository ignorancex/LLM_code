# -*- coding: utf-8 -*-
"""
Experiment script for Base GPLVM with 5 random splits and metric computation

"""
### Packages

import torch
import gc
import numpy as np
import json
import argparse
from astropy.io import fits
from tqdm import trange
from sklearn.model_selection import train_test_split
from gpytorch.mlls import VariationalELBO

### Internal functions and modules

from models.baseline_gplvm import QuasarModel, predict_latent
from utils.load_data import load_spectra_labels
from utils.predict_reconstruct import predict_from_latents_baseline
from models.likelihood import GaussianLikelihoodWithMissingObs
from utils.metrics import mean_absolute_error_spectra, mean_absolute_error_labels, nll_lum_bhm_edd

def run_experiment(seed, hdu, size, iterations, num_inducing, latent_dim):
    
    # Setting torch and numpy seed for reproducibility
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load joint spectra and label data 
    
    print('-----------Loading data----------------')

    X, Y, means_X, std_X, means_Y, std_Y, X_ivar, Y_ivar, snr, wave  = load_spectra_labels(hdu)

    data = np.hstack((X,Y))
    
    XY_train, XY_test, train_idx, test_idx = train_test_split(data, np.arange(len(data)), test_size=0.10, random_state=seed)
    
    XY_train = torch.Tensor(XY_train).to(device)
    XY_test = torch.Tensor(XY_test).to(device)
    std_X = torch.Tensor(std_X).to(device)
    std_Y = torch.Tensor(std_Y).to(device)
    means_X = torch.Tensor(means_X).to(device)
    means_Y = torch.Tensor(means_Y).to(device)
    
    # Experiment config
      
    N = len(XY_train)
    
    data_dim = XY_train.shape[1]
      
    # Model 
    
    baseline_model = QuasarModel(N, data_dim, latent_dim, num_inducing, latent_config='point', kernel_config='standard', spectra_dims=7).to(device)
    
    # Missing data Likelihood
    
    likelihood = GaussianLikelihoodWithMissingObs(batch_shape = baseline_model.batch_shape).to(device)
 
    # Declaring objective to be optimised along with optimiser
    
    mll = VariationalELBO(likelihood, baseline_model, num_data=len(XY_train)).to(device)

    optimizer = torch.optim.Adam([
        dict(params=baseline_model.parameters(), lr=0.001),
        dict(params=likelihood.parameters(), lr=0.001),
    ])
    
    ############## Training loop - optimises the objective wrt kernel hypers, ######
    ################  variational params and inducing inputs using the optimizer provided. ########
    
    print('-----------Training loop----------------')
    
    loss_list = []
    iterator = trange(iterations, leave=True)
    batch_size = 128

    for i in iterator: 
        
        batch_index = baseline_model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample_batch = baseline_model.Z.Z[batch_index]
        
        ### Getting the output of the two groups of GPs
        
        output_batch = baseline_model(sample_batch)

        ### Adding together the ELBO losses 
        
        loss = -mll(output_batch, XY_train[batch_index].T).sum()
        loss_list.append(loss.item())
        iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()
        #baseline_model.inducing_inputs.grad = baseline_model.inducing_inputs.grad.to(device)
    
    #Z_train = shared_model.Z.Z
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    
    XY_train = XY_train.cpu()
      
    ####### Reconstruct training data ##########
    
    # print('-----------Reconstruction----------------')

    # X_train = XY_train[::,0:-4]
    # Y_train = XY_train[:,-4::]
    
    # X_train_recon_cpu, Y_train_recon_cpu, X_train_recon_sigma, Y_train_recon_sigma = predict_spectra_labels_from_latents(Z_train, X_train, Y_train, shared_model, likelihood_spectra, likelihood_labels)
    
    # X_train_recon_orig_cpu = X_train_recon_cpu*std_X.cpu() + means_X.cpu()
    # Y_train_recon_orig_cpu = Y_train_recon_cpu*std_Y.cpu() + means_Y.cpu()
    
    #########  Testing framework on unseen quasars ##########
    
    print('-----------Testing on unseen quasars----------------')
    
    ####### Initialise test model at training params
   
    X_test = XY_test[::,0:-4]
    Y_test = XY_test[:,-4::]
    
    X_test_orig = X_test*std_X + means_X
    Y_test_orig = Y_test*std_Y + means_Y
    
    X_test_orig_cpu = X_test_orig.cpu().detach()
    Y_test_orig_cpu = Y_test_orig.cpu().detach()

    test_model = baseline_model.initialise_model_test(len(Y_test), latent_dim).to(device)

    test_loss, test_model, Z_test = predict_latent(test_model, XY_test, 
                                                         likelihood, lr=0.001, prior_z = None, steps = iterations)

    X_test_recon_cpu, Y_test_recon_cpu, X_train_recon_sigma, Y_train_recon_sigma, X_pred, Y_pred = predict_from_latents_baseline(Z_test.Z, XY_test, test_model, likelihood, std_X, std_Y)
    
    X_test_recon_orig_cpu = X_test_recon_cpu*std_X.cpu() + means_X.cpu()
    Y_test_recon_orig_cpu = Y_test_recon_cpu*std_Y.cpu() + means_Y.cpu()
 
    ############ Collect training metrics RMSE and NLL for spectra, bhm, lumin and edd ##############
    
    print('-----------Saving metrics----------------')

    #X_train_orig = X_train*std_X + means_X
    #Y_train_orig = Y_train*std_Y + means_Y
    
    #X_train_cpu = X_train.cpu().detach()
    #X_train_orig_cpu = X_train_orig.cpu().detach()
    
    #Y_train_cpu = Y_train.cpu().detach()
    #Y_train_orig_cpu = Y_train_orig.cpu().detach()
    
    #mae_train_spectra = mean_absolute_error_spectra(X_train_orig_cpu, X_train_recon_orig_cpu)
    #mae_train_labels = mean_absolute_error_labels(Y_train_orig_cpu, Y_train_recon_orig_cpu)
    
    mae_test_spectra = mean_absolute_error_spectra(X_test_orig_cpu, X_test_recon_orig_cpu)
    mae_test_labels = mean_absolute_error_labels(Y_test_orig_cpu, Y_test_recon_orig_cpu)
    nlpd_labels = nll_lum_bhm_edd(Y_pred, Y_test, std_Y)
    
    ####### Save metrics to json file ##########
    
    file_path = 'results/metrics_baseline' + str(seed) + '_' + str(size) + '.json'
    
    metrics = {
    #    'train_mae_spectra': [mae_train_spectra.item()],
    #    'train_mae_labels': mae_train_labels.numpy().tolist(),
        'test_mae_spectra': [mae_test_spectra.item()],
        'test_mae_labels': mae_test_labels.numpy().tolist(),
        'test_nlpd_labels': nlpd_labels.tolist()
        }
    
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
        

def main():
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='-------------')
    
    # Define arguments
    
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--data_size', type=str, default='1k')
    parser.add_argument('--iterations', type=int, help='Number of iterations', default=15000)
    parser.add_argument('--num_inducing', type=int, help='Number of inducing points', default=120)
    parser.add_argument('--latent_dim', type=int, help='Number of latent dimensions', default=10)

    # Parse arguments
    args = parser.parse_args()
   
    # Access arguments
    
    num_seeds = args.num_seeds
    size = args.data_size
    iterations = args.iterations
    num_inducing = args.num_inducing
    latent_dim = args.latent_dim
    
    if size == '20k':
        
        hdu = fits.open('data/data_norm_sdss16_SNR10_all.fits')
    
    elif size == '1k':
        
        hdu = fits.open('data/data_norm_sdss16_SNR10_random_1.fits')
        
    
    print(f"Run an experiment with {num_seeds} seeds, data size {size}, iterations {iterations} num_inducing {num_inducing} latent_dim {latent_dim}")
    
    random_seeds = [24, 42,33,60,6]
    
    for i in range(num_seeds):
        
        # Run the experiment with the parsed arguments
        seed = random_seeds[i] 
    
        run_experiment(seed, hdu, size, iterations, num_inducing, latent_dim)
        
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
        
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main()
     
 
