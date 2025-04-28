#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation code for quasar spectra reconstruction plots, label reproduction and test predictions

"""

import matplotlib.pylab as plt
import numpy as np

def plot_spectra_reconstructions(wave, X_test_recon_orig, X_test_orig, X_pred_var, obj_id):
    
    plt.figure(figsize=(9,5))
    plt.plot(wave, X_test_recon_orig[obj_id], c='b', alpha=0.6, label='predicted spectrum')
    plt.plot(wave, X_test_orig[obj_id], c='r', alpha=0.7, label='data')
    
    lower = X_test_recon_orig[obj_id] - 1.96*X_pred_var[obj_id]
    upper = X_test_recon_orig[obj_id] + 1.96*X_pred_var[obj_id]
   
    plt.fill_between(wave, lower, upper, color='b', alpha=0.5, label=r'$\pm2\sigma$')
    plt.title('Spectrum for obj id ' + str(obj_id))
    plt.legend(fontsize='small')
    plt.ylabel('normalised flux')
    plt.xlabel(r'rest-frame wavelength [$\AA$]')
    plt.tight_layout()
    

def plot_partial_spectra_reconstruction_report(wave, X_partial_recon_orig, X_test_orig, X_partial_pred_sigma, bhm_value):
    
    plt.figure(figsize=(9,15))
    
    obs_regions = [(0,280), (280,585), (140,350), (310,400)]
    
    for i in np.arange(4):
    
        plt.subplot(4,1,i+1)
        plt.plot(wave, X_partial_recon_orig[i], c='b', alpha=0.5, label='predicted spectrum')
        plt.plot(wave, X_test_orig, c='r', linestyle='dotted', alpha=0.8, label=r'data: $\log_{10}(M_{\bullet}/M_{\odot}) = $ ' + str(np.round(bhm_value.item(),4)))
        lower = X_partial_recon_orig[i] - 1.96*X_partial_pred_sigma[i]
        upper = X_partial_recon_orig[i] + 1.96*X_partial_pred_sigma[i]
        plt.axvspan(xmin=wave[obs_regions[i][0]], xmax=wave[obs_regions[i][1]], color='orange', alpha=0.3, label='Observed Region')
        plt.fill_between(wave, lower, upper, color='b', alpha=0.3, label=r'$\pm2\sigma$')
        plt.tight_layout()
        plt.ylabel('normalised flux')
        plt.ylim(ymin=0)
        if i == 0:
            plt.legend(fontsize='medium')
        if i == 3:
            plt.xlabel(r'rest-frame wavelength [$\AA$]')
        plt.suptitle('Quasar Spectra Reconstruction [from partial observations]')

def plot_y_label_comparison(Y_test_recon_orig, Y_test_orig, Y_test_pred_sigma, Y_test_sigma, snr_test, col_id, title):
    
    plt.figure()
    plt.scatter(Y_test_orig[:,col_id], Y_test_recon_orig[:,col_id], c=snr_test,cmap='jet')
    plt.errorbar(Y_test_orig[:,col_id], Y_test_recon_orig[:,col_id], yerr=1.96*Y_test_pred_sigma[:,col_id], fmt='None', color='k', alpha=0.2, errorevery=1)
    plt.errorbar(Y_test_orig[:,col_id], Y_test_recon_orig[:,col_id], xerr=Y_test_sigma[:,col_id], fmt='None', color='k', alpha=0.4, errorevery=1)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k',alpha=0.7, scalex=False, scaley=False)
    plt.ylabel(r'Predicted ', fontsize='large')
    plt.xlabel(r'Measured ', fontsize='large')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('SNR')

def spectra_reconstruction_report(wave, X_test_recon_orig, X_test_orig, X_pred_sigma, bhms):
    
    plt.figure(figsize=(7,15))
    
    obj_ids = [42,4,13,78]
    
    for i in np.arange(4):
    
        obj_id = obj_ids[i]
        plt.subplot(4,1,i+1)
        plt.plot(wave, X_test_recon_orig[obj_id], c='b', alpha=0.5, label='predicted spectrum')
        plt.plot(wave, X_test_orig[obj_id], c='r', linestyle='--', alpha=0.8, label=r'data; $\log_{10}(M_{\bullet}/M_{\odot}) = $ ' + str(np.round(bhms[obj_id].item(),4)))
        lower = X_test_recon_orig[obj_id] - 1.96*X_pred_sigma[obj_id]
        upper = X_test_recon_orig[obj_id] + 1.96*X_pred_sigma[obj_id]
        plt.fill_between(wave, lower, upper, color='b', alpha=0.3, label=r'$\pm2\sigma$')
        plt.legend(fontsize='medium')
        plt.ylabel('normalised flux')
        if i == 3:
            plt.xlabel(r'rest-frame wavelength [$\AA$]')
        plt.suptitle('Spectra reconstruction')

def plot_y_label_report(Y_test_recon_orig, Y_test_orig, Y_test_pred_sigma, Y_test_sigma, snr_test):
    
    labels = [r'$\log_{10}(L_{\rm bol}/\rm erg\,s^{-1})$', r'$\log_{10}(M_\bullet/M_\odot)$', r'$\log_{10}\lambda_{\rm Edd}$']
    title = ['Bolometric luminosity', 'Black hole mass', 'Eddington Ratio']
    
    plt.figure(figsize=(8,3))
    
    for col_id in [1,2,3]: ## 3 columns 
    
       plt.subplot(1,3,col_id)
       plt.errorbar(Y_test_orig[:,col_id], Y_test_recon_orig[:,col_id], yerr=1.96*Y_test_pred_sigma[:,col_id], fmt='None', color='orange', alpha=0.4, errorevery=1, zorder=1)
       plt.errorbar(Y_test_orig[:,col_id], Y_test_recon_orig[:,col_id], xerr=1.96*Y_test_sigma[:,col_id], fmt='None', color='orange', alpha=0.4, errorevery=1, zorder=2)
       plt.scatter(Y_test_orig[:,col_id], Y_test_recon_orig[:,col_id], s=4, c=snr_test,cmap='jet', zorder=3)
       xpoints = ypoints = plt.xlim()
       plt.plot(xpoints, ypoints, linestyle='--', color='k',alpha=0.7, scalex=False, scaley=False)
       plt.ylabel(r'Predicted ' + labels[col_id-1], fontsize='small')
       plt.xlabel(r'Measured ' + labels[col_id-1], fontsize='small')
       plt.title(title[col_id - 1], fontsize='small')
       plt.xlim(xpoints[0], xpoints[-1])
       plt.ylim(xpoints[0], xpoints[-1])
       plt.tight_layout()
       plt.tick_params(axis='both', which='major', labelsize=8)
    cbar = plt.colorbar()
    cbar.set_label('SNR')
    cbar.ax.tick_params(labelsize=9)
    
    


