# A bunch of plotting routines

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
sys.path.append("../src/training/")
import NoInterNet_model as NIN
import NoInterNet_fraction_model as NINf

import torch

from torch.utils.data import DataLoader
from torch import nn
from sklearn.decomposition import PCA, TruncatedSVD

from matplotlib.pyplot import rc
import matplotlib.font_manager
rc('font',**{'size':'22','family':'serif','serif':['CMU serif']})
rc('mathtext', **{'fontset':'cm'})
rc('text', usetex=True)
rc('legend',**{'fontsize':'18'})
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['legend.fontsize'] = 15
#matplotlib.rcParams['legend.title_fontsize'] = 25
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['xtick.minor.size'] = 2.5
matplotlib.rcParams['ytick.minor.size'] = 2.5
matplotlib.rcParams['xtick.major.width'] = 1.5
matplotlib.rcParams['ytick.major.width'] = 1.5
matplotlib.rcParams['xtick.minor.width'] = 1.5
matplotlib.rcParams['ytick.minor.width'] = 1.5
matplotlib.rcParams['axes.titlesize'] = 30
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_history(folder, save=None, log=False):
    history = np.loadtxt(folder + 'history.dat')

    plt.plot(history[:,0], label='Training')
    plt.plot(history[:,1], label='Validation')

    if log:
        plt.yscale('log')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save is not None:
        plt.savefig(save)

    plt.show()

def load_multipoles(file_name, Pk_name, k_max=0.8):
    import pandas as pd

    Pk_paths = pd.read_csv(file_name)
    Pk_paths = Pk_paths.reset_index()

    Pks_0 = []
    Pks_2 = []
    Pks_4 = []

    for i, row in Pk_paths.iterrows():
        Pk  = np.loadtxt(row[Pk_name])

        selk = Pk[:,0]  < k_max

        Pks_0.append(Pk[:,1][selk])
        Pks_2.append(Pk[:,2][selk])
        Pks_4.append(Pk[:,3][selk])
            

    return np.array(Pks_0), np.array(Pks_2), np.array(Pks_4), np.array(Pk[selk,0])

def plot_mean_P0(file_name, test, model, k_max, device, lenk, test_start=-100, save=None, title=None, norm_fs=False):
    P0_int, P2_int, P4_int, ks_int = load_multipoles(file_name, 'Pk_int')
    P0_tru, P2_tru, P4_tru, ks_tru = load_multipoles(file_name, 'Pk_true')

    sel = ks_int < k_max
    k_plot = ks_int[sel]

    P0_int = P0_int[test_start:,sel]
    P0_tru = P0_tru[test_start:,sel]

    P0_int_mean = np.mean(P0_int, axis=0)
    P0_tru_mean = np.mean(P0_tru, axis=0)

    test_dataloader_all = DataLoader(test, batch_size=len(test), shuffle=False)

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        P0_corr = P0_int * pred.detach().numpy()[:,:lenk] #taking only the correction part
        P0_corr_error = P0_int * pred.detach().numpy()[:,lenk+1:2*lenk+1] #error in the correction
    
        f_true = y.detach().numpy()[:,lenk]
        f_pred = pred.detach().numpy()[:,lenk]
        f_error = pred.detach().numpy()[:,-1]

        if norm_fs:
            f_true = NINf.inv_minmax(f_true)
            f_pred = NINf.inv_minmax(f_pred)

        p_mean = np.mean(pred.detach().numpy()[:,:lenk], axis=0)
        y_mean = np.mean(y.detach().numpy()[:,:lenk], axis=0)
        P0_corr_mean = np.mean(P0_corr, axis=0)

        

    fig, ax = plt.subplots(1, 2, figsize=(6*2, 4.5))
    
    ax[0].loglog(k_plot, P0_tru_mean, label='Mean P0 true', color='tab:green')
    ax[0].loglog(k_plot, P0_int_mean, label='Mean P0 interloper', color='tab:blue')
    ax[0].loglog(k_plot, P0_corr_mean, label='Mean P0 corrected', color='tab:orange', linestyle='dashed')
    ax[0].legend()
    ax[0].set_xlabel('$k$')
    ax[0].set_ylabel('P(k) monopole')
    

    for i in range(len(test)):
        ax[1].semilogx(k_plot, 1 - P0_corr[i]/P0_tru[i], color='k', alpha=.05)
    ax[1].semilogx(k_plot, 1 - P0_corr_mean/P0_tru_mean)
    ax[1].fill_between(k_plot, -0.01, 0.01, color='tab:blue', alpha=.1)
    ax[1].axhline(0, color='k', linestyle='dashed', zorder=0)

    ax[1].set_xlabel('$k$')
    ax[1].set_ylabel('Relative error of mean correction')

    if title is not None:
        plt.suptitle(title)
    
    plt.tight_layout()
    if save is not None:
        plt.savefig(save + '_correction.png', dpi=300)
    plt.show()

    plt.scatter(f_true, f_pred)
    plt.plot([.01, .11], [.01, .11], linestyle='dashed', color='k')
    plt.xlabel('True fraction')
    plt.ylabel('Measured fraction')

    if save is not None:
        plt.savefig(save + '__fraction.png', dpi=300)
    plt.show()

def plot_residual_error(test_dataloader, model, k_max, ks, device, lenk, batch_test=10, N_batches=5, norm_fs=False, max_corr=None, min_corr=None):
    assert batch_test == 10, 'for now only 10 is aviable'

    sel = ks < k_max
    k_plot = ks[sel]

    count = 1

    for X, y, n in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        fig, ax = plt.subplots(2,5, figsize=(4*5, 3*2))

        if max_corr is not None:
            pred.detach()[:,:lenk] = NINf.inv_maxmin_corr(pred.detach()[:,:lenk], max_corr, min_corr)
            y.detach()[:,:lenk] = NINf.inv_maxmin_corr(y.detach()[:,:lenk], max_corr, min_corr)
            pred.detach().numpy()[:,lenk+1:2*lenk+1] *= (max_corr - min_corr)
    
        for i in range(batch_test):
            k = i//5
            j = i if i < 5 else i -5
            ax[k,j].errorbar(k_plot, 1 - pred.detach()[i,:lenk]/y.detach()[i,:lenk], yerr=np.sqrt((pred.detach().numpy()[i,lenk+1:2*lenk+1]/y.detach()[i,:lenk])**2),
                              label='After correction')
            ax[k,j].semilogx(k_plot, 1 - 1/y.detach()[i,:-1], label='Before correction')
            ax[k,j].axhline(0, color='k', linestyle='dashed')
            ax[k,j].fill_between(k_plot, -0.01, 0.01, color='k', alpha=.1)
            ax[k,j].set_xlabel('$k$', fontsize=9)
            ax[k,j].set_ylabel('Relative error', fontsize=9)
            ax[k,j].legend(fontsize=9)
            f_true = y.detach()[i,lenk]
            f_pred = pred.detach()[i,lenk]
            f_erro = pred.detach()[i,-1]
            if norm_fs:
                f_true = NINf.inv_minmax(f_true)
                f_pred = NINf.inv_minmax(f_pred)
                f_erro *= 0.1
            
            ax[k,j].set_title(f'f_true = {f_true:.4f} \n f_meas = {f_pred:.4f} +- {f_erro:.4f}', size=11)
            #plt.semilogx(Pk_int[sel,0], y.detach()[i,:])
        plt.tight_layout()
        plt.show()

        if count == N_batches: return
        else: count += 1

    return
    
    
def plot_mean_error(test, model, k_max, ks, device, lenk, norm_fs=False, max_corr=None, min_corr=None, pole=False, lim=[None,None]):

    test_dataloader_all = DataLoader(test, batch_size=len(test), shuffle=False)
    j = 2 if pole else 1

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        f_y = y.detach().numpy()[:,lenk*j]
        if norm_fs:
            f_y = NINf.inv_minmax(f_y)
        f_u = np.unique(f_y)
            
        p_mean = np.zeros((len(f_u), lenk * j))
        y_mean = np.zeros((len(f_u), lenk * j))
        std_mean = np.zeros((len(f_u), lenk * j))

        pred_n = pred.detach().numpy()[:,:lenk * j]
        y_n = y.detach().numpy()[:,:lenk * j]
        pred_std = pred.detach().numpy()[:,lenk*j+1:lenk*j*2+1]

        if max_corr is not None:
            pred_n[:,:lenk*j] = NINf.inv_maxmin_corr(pred_n[:,:lenk*j], max_corr, min_corr)
            y_n[:,:lenk*j] = NINf.inv_maxmin_corr(y_n[:,:lenk*j], max_corr, min_corr)
            pred_std *= (max_corr - min_corr)

        for i, f in enumerate(f_u):
           sel = np.where(f_y == f)
           p_mean[i] = np.mean(pred_n[sel], axis=0)
           y_mean[i] = np.mean(y_n[sel], axis=0)
           std_mean[i] = np.sqrt(np.mean(pred_std[sel]**2, axis=0))
    
    
    sel = ks < k_max
    k_plot = ks[sel]

    for i, f in enumerate(f_u):

        if pole:
            fig, ax = plt.subplots(2,2, figsize=(6*2, 4.5*2))
            ax = ax.flatten()
        else:
            fig, ax = plt.subplots(1, 2, figsize=(6*2, 4.5))

        ax[0].errorbar(k_plot, p_mean[i][:lenk], yerr=std_mean[i][:lenk], label='Predicition mean')
        ax[0].semilogx(k_plot, y_mean[i][:lenk], label='Label mean')
        ax[0].legend()
        ax[0].set_xlabel('$k$')
        ax[0].set_ylabel('Mean correction monopole')
        
        ax[1].errorbar(k_plot, 1 - p_mean[i][:lenk]/y_mean[i][:lenk], yerr=std_mean[i][:lenk]/np.abs(y_mean[i][:lenk]))
        ax[1].axhline(0, color='k', linestyle='dashed')
        ax[1].set_xscale('log')
        ax[1].set_xlabel('$k$')
        ax[1].set_ylim(lim[0],lim[1])
        ax[1].set_ylabel('Relative error of mean correction monopole')

        if pole:
            ax[2].errorbar(k_plot, p_mean[i][lenk:], yerr=std_mean[i][lenk:], label='Predicition mean')
            ax[2].semilogx(k_plot, y_mean[i][lenk:], label='Label mean')
            ax[2].legend()
            ax[2].set_xlabel('$k$')
            ax[2].set_ylabel('Mean correction quadrupole')
            
            ax[3].errorbar(k_plot, 1 - p_mean[i][lenk:]/y_mean[i][lenk:], yerr=std_mean[i][lenk:]/np.abs(y_mean[i][lenk:]))
            ax[3].axhline(0, color='k', linestyle='dashed')
            ax[3].set_xscale('log')
            ax[3].set_xlabel('$k$')
            ax[3].set_ylim(-1,1)
            ax[3].set_ylabel('Relative error of mean correction quadrupole')

        plt.suptitle(f'f_true = {f:.4f}')
        
        plt.tight_layout()
        plt.show()

    return p_mean, y_mean

def plot_fraction(test, model, lenk, device, norm_fs=False, pole=False, save=None, equal=False, limits=[0.0, 0.12], title=None, step=None):

    test_dataloader_all = DataLoader(test, batch_size=len(test), shuffle=True)

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
    
        j = 2 if pole else 1
        f_true = y.detach().numpy()[:,lenk * j]
        f_pred = pred.detach().numpy()[:,lenk * j]
        f_erro = pred.detach().numpy()[:,-1]

        if step is not None:
            f_true = f_true[::step]
            f_pred = f_pred[::step]
            f_erro = f_erro[::step]

        if norm_fs:
            f_true = NINf.inv_minmax(f_true)
            f_pred = NINf.inv_minmax(f_pred)
            f_erro *= 0.1
    
    plt.errorbar(f_true, f_pred, yerr=np.sqrt(f_erro**2), ecolor='tab:orange', capsize=2, elinewidth=.5, marker='.', linestyle='', alpha=.7)
    plt.plot([.01, .11], [.01, .11], linestyle='dashed', color='k')
    plt.xlabel('True fraction')
    plt.ylabel('Measured fraction')
    #plt.xlim(limits[0], limits[1])
    plt.ylim(limits[0], limits[1])

    if title is not None: plt.title(title, size=20)

    if equal: plt.axis('equal')
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300)

    plt.show()

    print('MSE error:', np.mean((f_true - f_pred)**2))

def plot_fractionN(test, model, device, norm_fs=False):

    test_dataloader_all = DataLoader(test, batch_size=len(test), shuffle=True)

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
    
        f_true = y.detach().numpy()[:,-1]
        f_pred = pred.detach().numpy()[:,-1]

        if norm_fs:
            f_true = NINf.inv_minmax(f_true)
            f_pred = NINf.inv_minmax(f_pred)
    
    plt.scatter(f_true, f_pred)
    plt.plot([.01, .11], [.01, .11], linestyle='dashed', color='k')
    plt.xlabel('True fraction')
    plt.ylabel('Measured fraction')

    plt.show()

    print('MSE error:', np.mean((f_true - f_pred)**2))

def plot_residual_error_ratio(test_dataloader, model, k_max, ks, device, lenk, batch_test=10, N_batches=5):
    assert batch_test == 10, 'for now only 10 is aviable'

    sel = ks < k_max
    k_plot = ks[sel]

    count = 1

    colors = ['b', 'r', 'g', 'c', 'y']

    for X, y, n in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
    
        for i in range(batch_test):
            #k = i//5
            j = i if i < 5 else i -5
            plt.semilogx(k_plot, (1 - pred.detach()[i,:-1]/y.detach()[i,:lenk]) / (1 - 1/y.detach()[i,:lenk]), color=colors[j], alpha=.5)
            #plt.semilogx(k_plot, 1 - 1/y.detach()[i,:], label='Before correction')
            #plt.axhline(0, color='k', linestyle='dashed')
            plt.xlabel('$k$')
            plt.ylim(-10,10)
            #plt.semilogx(Pk_int[sel,0], y.detach()[i,:])
        plt.tight_layout()

        if count == N_batches: return
        else: count += 1

    plt.show()

def plot_mean_error_all(test, model, k_max, ks, device, lenk):

    test_dataloader_all = DataLoader(test, batch_size=len(test), shuffle=True)

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
    
        p_mean = np.mean(pred.detach().numpy()[:lenk], axis=0)
        y_mean = np.mean(y.detach().numpy()[:lenk], axis=0)
    
    
    sel = ks < k_max
    k_plot = ks[sel]

    fig, ax = plt.subplots(1, 2, figsize=(6*2, 4.5))
    
    ax[0].semilogx(k_plot, p_mean, label='Predicition mean')
    ax[0].semilogx(k_plot, y_mean, label='Label mean')
    ax[0].legend()
    ax[0].set_xlabel('$k$')
    ax[0].set_ylabel('Mean correction')
    
    ax[1].semilogx(k_plot, 1 - p_mean/y_mean)
    ax[1].axhline(0, color='k', linestyle='dashed')
    ax[1].set_xlabel('$k$')
    ax[1].set_ylabel('Relative error of mean correction')

    p = pred.detach().numpy()[:lenk]
    y = y.detach().numpy()[:lenk]

    for i in range(len(test)):
        ax[0].semilogx(k_plot, p[i,:], color='b', alpha=.3, zorder=0)
        ax[0].semilogx(k_plot, y[i,:], color='r', alpha=.3, zorder=0)
        
        ax[1].semilogx(k_plot, 1 - p[i,:]/y[i,:], color='r', alpha=.2, zorder=0)
        
    
    plt.tight_layout()
    plt.show()

    return p_mean, y_mean

def compare_residual_error(test_dataloader, model1, model2, k_max, ks, device, lenk, batch_test=10, N_batches=5, norm_fs=False, max_corr=None, min_corr=None):
    assert batch_test == 10, 'for now only 10 is aviable'

    sel = ks < k_max
    k_plot = ks[sel]

    count = 1

    for X, y, n in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        pred1 = model1(X)
        pred2 = model2(X)
        fig, ax = plt.subplots(2,5, figsize=(4*5, 3*2))
    
        if max_corr is not None:
            pred1.detach()[:,:lenk] = NINf.inv_maxmin_corr(pred1.detach()[:,:lenk], max_corr, min_corr)
            y.detach()[:,:lenk] = NINf.inv_maxmin_corr(y.detach()[:,:lenk], max_corr, min_corr)
            pred1.detach().numpy()[:,lenk+1:2*lenk+1] *= (max_corr - min_corr)
            pred2.detach()[:,:lenk] = NINf.inv_maxmin_corr(pred2.detach()[:,:lenk], max_corr, min_corr)

        for i in range(batch_test):
            k = i//5
            j = i if i < 5 else i -5
            ax[k,j].errorbar(k_plot, 1 - pred1.detach()[i,:lenk]/y.detach()[i,:lenk], yerr=np.sqrt((pred1.detach().numpy()[i,lenk+1:2*lenk+1]/y.detach()[i,:lenk])**2),
                              label='Corrected moments')
            ax[k,j].semilogx(k_plot, 1 - pred2.detach()[i,:-1]/y.detach()[i,:-1], label='Corrected')
            ax[k,j].semilogx(k_plot, 1 - 1/y.detach()[i,:-1], label='Before correction')
            ax[k,j].axhline(0, color='k', linestyle='dashed')
            ax[k,j].fill_between(k_plot, -0.01, 0.01, color='k', alpha=.1)
            ax[k,j].set_xlabel('$k$')
            ax[k,j].set_ylabel('Relative error')
            ax[k,j].legend()
            f_true = y.detach()[i,lenk]
            f_pred1 = pred1.detach()[i,lenk]
            f_erro1 = pred1.detach()[i,-1]
            f_pred2 = pred2.detach()[i,-1]
            if norm_fs:
                f_true = NINf.inv_minmax(f_true)
                f_pred1 = NINf.inv_minmax(f_pred1)
                f_erro1 *= 0.1
                f_pred2 = NINf.inv_minmax(f_pred2)
            ax[k,j].set_title(f'f_true = {f_true:.4f} \n f_meas = {f_pred1:.4f} +- {f_erro1:.4f} \n f_meas = {f_pred2:.4f}')
            #plt.semilogx(Pk_int[sel,0], y.detach()[i,:])
        plt.tight_layout()
        plt.show()

        if count == N_batches: return
        else: count += 1

    return

def compare_mean_error(test, model1, model2, k_max, ks, device, lenk, norm_fs=False, max_corr=None, min_corr=None): 

    test_dataloader_all = DataLoader(test, batch_size=len(test), shuffle=False)

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred1 = model1(X)
        pred2 = model2(X)

        f_y = y.detach().numpy()[:,lenk]
        if norm_fs:
            f_y = NINf.inv_minmax(f_y)
        f_u = np.unique(f_y)
            
        p_mean1 = np.zeros((len(f_u), lenk))
        y_mean = np.zeros((len(f_u), lenk))
        std_mean1 = np.zeros((len(f_u), lenk))
        p_mean2 = np.zeros((len(f_u), lenk))

        pred_n1 = pred1.detach().numpy()[:,:lenk]
        y_n = y.detach().numpy()[:,:lenk]
        pred_std1 = pred1.detach().numpy()[:,lenk+1:lenk*2+1]
        pred_n2 = pred2.detach().numpy()[:,:-1]

        if max_corr is not None:
            pred_n1[:,:lenk] = NINf.inv_maxmin_corr(pred_n1[:,:lenk], max_corr, min_corr)
            y_n[:,:lenk] = NINf.inv_maxmin_corr(y_n[:,:lenk], max_corr, min_corr)
            pred_std1 *= (max_corr - min_corr)
            pred_n2[:,:lenk] = NINf.inv_maxmin_corr(pred_n2[:,:lenk], max_corr, min_corr)

        for i, f in enumerate(f_u):
           sel = np.where(f_y == f)
           p_mean1[i] = np.mean(pred_n1[sel], axis=0)
           y_mean[i] = np.mean(y_n[sel], axis=0)
           std_mean1[i] = np.sqrt(np.mean(pred_std1[sel]**2, axis=0))
           p_mean2[i] = np.mean(pred_n2[sel], axis=0)
    
    
    sel = ks < k_max
    k_plot = ks[sel]

    for i, f in enumerate(f_u):

        fig, ax = plt.subplots(1, 2, figsize=(6*2, 4.5))
    
        ax[0].errorbar(k_plot, p_mean1[i], yerr=std_mean1[i], label='Predicition mean mom')
        ax[0].semilogx(k_plot, p_mean2[i], label='Predicition mean')
        ax[0].semilogx(k_plot, y_mean[i], label='Label mean')
        ax[0].legend()
        ax[0].set_xlabel('$k$')
        ax[0].set_ylabel('Mean correction')
        
        ax[1].errorbar(k_plot, 1 - p_mean1[i]/y_mean[i], yerr=std_mean1[i]/y_mean[i], label='Moments')
        ax[1].semilogx(k_plot, 1 - p_mean2[i]/y_mean[i], label='Standard')
        ax[1].axhline(0, color='k', linestyle='dashed')
        ax[1].legend()
        ax[1].set_xlabel('$k$')
        ax[1].set_ylabel('Relative error of mean correction')

        plt.suptitle(f'f_true = {f:.4f}')
        
        plt.tight_layout()
        plt.show()

    return p_mean1, y_mean

def plot_residual_error_knowf(test_dataloader, model, k_max, ks, device, lenk, batch_test=10, N_batches=5, norm_fs=False, max_corr=None, min_corr=None):
    assert batch_test == 10, 'for now only 10 is aviable'

    sel = ks < k_max
    k_plot = ks[sel]

    count = 1

    for X, y, n in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        fig, ax = plt.subplots(2,5, figsize=(4*5, 3*2))

        if max_corr is not None:
            pred.detach()[:,:lenk] = NINf.inv_maxmin_corr(pred.detach()[:,:lenk], max_corr, min_corr)
            y.detach()[:,:lenk] = NINf.inv_maxmin_corr(y.detach()[:,:lenk], max_corr, min_corr)
    
    
        for i in range(batch_test):
            k = i//5
            j = i if i < 5 else i -5
            ax[k,j].semilogx(k_plot, 1 - pred.detach()[i,:]/y.detach()[i,:], label='After correction')
            ax[k,j].semilogx(k_plot, 1 - 1/y.detach()[i,:], label='Before correction')
            ax[k,j].axhline(0, color='k', linestyle='dashed')
            ax[k,j].fill_between(k_plot, -0.01, 0.01, color='k', alpha=.1)
            ax[k,j].set_xlabel('$k$')
            ax[k,j].set_ylabel('Relative error')
            ax[k,j].legend()
            f_true = X.detach()[i,-1]
            f_pred = pred.detach()[i,-1]
            if norm_fs:
                f_true = NINf.inv_minmax(f_true)
            ax[k,j].set_title(f'f_true = {f_true:.4f}')
            #plt.semilogx(Pk_int[sel,0], y.detach()[i,:])
        plt.tight_layout()
        plt.show()

        if count == N_batches: return
        else: count += 1

    return
    
    
def plot_mean_error_knowf(test, model, k_max, ks, device, norm_fs=False, max_corr=None, min_corr=None): 

    test_dataloader_all = DataLoader(test, batch_size=len(test), shuffle=False)

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        lenk = len(pred.detach().numpy()[0,:])
        f_y = X.detach().numpy()[:,-1]
        if norm_fs:
            f_y = NINf.inv_minmax(f_y)
        f_u = np.unique(f_y)
            
        p_mean = np.zeros((len(f_u), lenk))
        y_mean = np.zeros((len(f_u), lenk))

        pred_n = pred.detach().numpy()
        y_n = y.detach().numpy()

        if max_corr is not None:
            pred_n[:,:lenk] = NINf.inv_maxmin_corr(pred_n[:,:lenk], max_corr, min_corr)
            y_n[:,:lenk] = NINf.inv_maxmin_corr(y_n[:,:lenk], max_corr, min_corr)

        for i, f in enumerate(f_u):
           sel = np.where(f_y == f)
           p_mean[i] = np.mean(pred_n[sel], axis=0)
           y_mean[i] = np.mean(y_n[sel], axis=0)
    
    
    sel = ks < k_max
    k_plot = ks[sel]

    for i, f in enumerate(f_u):

        fig, ax = plt.subplots(1, 2, figsize=(6*2, 4.5))
    
        ax[0].semilogx(k_plot, p_mean[i], label='Predicition mean')
        ax[0].semilogx(k_plot, y_mean[i], label='Label mean')
        ax[0].legend()
        ax[0].set_xlabel('$k$')
        ax[0].set_ylabel('Mean correction')
        
        ax[1].semilogx(k_plot, 1 - p_mean[i]/y_mean[i])
        ax[1].axhline(0, color='k', linestyle='dashed')
        ax[1].set_xlabel('$k$')
        ax[1].set_ylabel('Relative error of mean correction')

        plt.suptitle(f'f_true = {f:.4f}')
        
        plt.tight_layout()
        plt.show()

    return p_mean, y_mean

def plot_residual_error_knowf_inference(test_dataloader, model, k_max, ks, device, lenk, batch_test=10, N_batches=5, norm_fs=False, max_corr=None, min_corr=None):
    assert batch_test == 10, 'for now only 10 is aviable'

    sel = ks < k_max
    k_plot = ks[sel]

    count = 1

    for X, y, n in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        fig, ax = plt.subplots(2,5, figsize=(4*5, 3*2))

        if max_corr is not None:
            pred.detach()[:,:lenk] = NINf.inv_maxmin_corr(pred.detach()[:,:lenk], max_corr, min_corr)
            y.detach()[:,:lenk] = NINf.inv_maxmin_corr(y.detach()[:,:lenk], max_corr, min_corr)
            pred.detach().numpy()[:,lenk:] *= (max_corr - min_corr)
    
    
        for i in range(batch_test):
            k = i//5
            j = i if i < 5 else i -5
            ax[k,j].errorbar(k_plot, 1 - pred.detach()[i,:lenk]/y.detach()[i,:lenk], yerr=np.sqrt((pred.detach().numpy()[i,lenk:]/y.detach()[i,:lenk])**2),
                              label='After correction')
            ax[k,j].semilogx(k_plot, 1 - 1/y.detach()[i,:], label='Before correction')
            ax[k,j].axhline(0, color='k', linestyle='dashed')
            ax[k,j].fill_between(k_plot, -0.01, 0.01, color='k', alpha=.1)
            ax[k,j].set_xlabel('$k$')
            ax[k,j].set_ylabel('Relative error')
            ax[k,j].legend()
            f_true = X.detach()[i,-1]
            f_pred = pred.detach()[i,-1]
            if norm_fs:
                f_true = NINf.inv_minmax(f_true)
            ax[k,j].set_title(f'f_true = {f_true:.4f}')
            #plt.semilogx(Pk_int[sel,0], y.detach()[i,:])
        plt.tight_layout()
        plt.show()

        if count == N_batches: return
        else: count += 1

    return
    
    
def plot_mean_error_knowf_inference(test, model, k_max, ks, device, lenk, norm_fs=False, max_corr=None, min_corr=None): 
    test_dataloader_all = DataLoader(test, batch_size=len(test), shuffle=False)

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        f_y = X.detach().numpy()[:,-1]
        if norm_fs:
            f_y = NINf.inv_minmax(f_y)
        f_u = np.unique(f_y)
            
        p_mean = np.zeros((len(f_u), lenk))
        y_mean = np.zeros((len(f_u), lenk))
        std_mean = np.zeros((len(f_u), lenk))

        pred_n = pred.detach().numpy()
        y_n = y.detach().numpy()
        pred_std = pred.detach().numpy()[:,lenk:]

        if max_corr is not None:
            pred_n[:,:lenk] = NINf.inv_maxmin_corr(pred_n[:,:lenk], max_corr, min_corr)
            y_n[:,:lenk] = NINf.inv_maxmin_corr(y_n[:,:lenk], max_corr, min_corr)
            pred_std *= (max_corr - min_corr)

        for i, f in enumerate(f_u):
           sel = np.where(f_y == f)
           p_mean[i] = np.mean(pred_n[sel], axis=0)
           y_mean[i] = np.mean(y_n[sel], axis=0)
           std_mean[i] = np.sqrt(np.mean(pred_std[sel]**2, axis=0))
    
    
    sel = ks < k_max
    k_plot = ks[sel]

    for i, f in enumerate(f_u):

        fig, ax = plt.subplots(1, 2, figsize=(6*2, 4.5))
    
        ax[0].errorbar(k_plot, p_mean[i][:lenk], yerr=std_mean[i][:lenk], label='Predicition mean')
        ax[0].semilogx(k_plot, y_mean[i], label='Label mean')
        ax[0].legend()
        ax[0].set_xlabel('$k$')
        ax[0].set_ylabel('Mean correction')
        
        ax[1].errorbar(k_plot, 1 - p_mean[i][:lenk]/y_mean[i][:lenk], yerr=std_mean[i][:lenk]/np.abs(y_mean[i][:lenk]))
        ax[1].axhline(0, color='k', linestyle='dashed')
        ax[1].set_xlabel('$k$')
        ax[1].set_ylabel('Relative error of mean correction')

        plt.suptitle(f'f_true = {f:.4f}')
        
        plt.tight_layout()
        plt.show()

    return p_mean, y_mean


def plot_residual_error_pca(test_dataloader, model, pca, k_max, ks, device, lenk, batch_test=10, N_batches=5, norm_fs=False, 
                            max_corr=None, min_corr=None, max_corr_pca=None, min_corr_pca=None):
    assert batch_test == 10, 'for now only 10 is aviable'

    sel = ks < k_max
    k_plot = ks[sel]

    count = 1

    for X, y, n in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        if max_corr_pca is not None:
            pred.detach()[:,:lenk] = NINf.inv_maxmin_corr(pred.detach()[:,:lenk], max_corr_pca, min_corr_pca)
            y.detach()[:,:lenk] = NINf.inv_maxmin_corr(y.detach()[:,:lenk], max_corr_pca, min_corr_pca)
            pred.detach().numpy()[:,lenk+1:2*lenk+1] *= (max_corr_pca - min_corr_pca)

        #decompress P
        P_corr_p = pca.inverse_transform(pred.detach()[:,:lenk])
        P_corr_l = pca.inverse_transform(y.detach()[:,:lenk])
        P_corr_s = pca.inverse_transform(pred.detach().numpy()[:,lenk+1:2*lenk+1])
        
        if max_corr is not None:
            P_corr_p = NINf.inv_maxmin_corr(P_corr_p, max_corr, min_corr)
            P_corr_l = NINf.inv_maxmin_corr(P_corr_l, max_corr, min_corr)
            P_corr_s *= (max_corr - min_corr)

        fig, ax = plt.subplots(2,5, figsize=(4*5, 3*2))
    
        for i in range(batch_test):
            k = i//5
            j = i if i < 5 else i -5
            ax[k,j].errorbar(k_plot, 1 - P_corr_p[i,:]/P_corr_l[i,:], yerr=np.sqrt((P_corr_s[i,:]/P_corr_l[i,:])**2),
                              label='After correction')
            ax[k,j].semilogx(k_plot, 1 - 1/P_corr_l[i,:], label='Before correction')
            ax[k,j].axhline(0, color='k', linestyle='dashed')
            ax[k,j].fill_between(k_plot, -0.01, 0.01, color='k', alpha=.1)
            ax[k,j].set_xlabel('$k$')
            ax[k,j].set_ylabel('Relative error')
            ax[k,j].legend()
            f_true = y.detach()[i,lenk]
            f_pred = pred.detach()[i,lenk]
            f_erro = pred.detach()[i,-1]
            if norm_fs:
                f_true = NINf.inv_minmax(f_true)
                f_pred = NINf.inv_minmax(f_pred)
                f_erro *= 0.1
            
            ax[k,j].set_title(f'f_true = {f_true:.4f} \n f_meas = {f_pred:.4f} +- {f_erro:.4f}')
            #plt.semilogx(Pk_int[sel,0], y.detach()[i,:])
        plt.tight_layout()
        plt.show()

        if count == N_batches: return
        else: count += 1

    return
    
def compare_residual_error_pca(test_dataloader1, test_dataloader2, model1, model2, pca, k_max, ks, device, lenk1, lenk2, batch_test=10, N_batches=5, norm_fs=False, 
                               max_corr=None, min_corr=None, max_corr_pca=None, min_corr_pca=None):
    assert batch_test == 10, 'for now only 10 is aviable'

    sel = ks < k_max
    k_plot = ks[sel]

    count = 1

    for d1, d2 in zip(test_dataloader1, test_dataloader2):
        X1, y1, n1 = d1
        X2, y2, n2 = d2
    
        X1 = X1.to(device)
        y1 = y1.to(device)
        X2 = X2.to(device)
        y2 = y2.to(device)

        pred1 = model1(X1)
        pred2 = model2(X2)

        if max_corr_pca is not None:
            pred2.detach()[:,:lenk2] = NINf.inv_maxmin_corr(pred2.detach()[:,:lenk2], max_corr_pca, min_corr_pca)
            y2.detach()[:,:lenk2] = NINf.inv_maxmin_corr(y2.detach()[:,:lenk2], max_corr_pca, min_corr_pca)
            pred2.detach().numpy()[:,lenk2+1:2*lenk2+1] *= (max_corr_pca - min_corr_pca)

        #decompress P
        P_corr_p = pca.inverse_transform(pred2.detach()[:,:lenk2])
        P_corr_l = pca.inverse_transform(y2.detach()[:,:lenk2])
        P_corr_s = pca.inverse_transform(pred2.detach().numpy()[:,lenk2+1:2*lenk2+1])
        
        if max_corr is not None:
            P_corr_p = NINf.inv_maxmin_corr(P_corr_p, max_corr, min_corr)
            P_corr_l = NINf.inv_maxmin_corr(P_corr_l, max_corr, min_corr)
            P_corr_s *= (max_corr - min_corr)
            pred1.detach()[:,:lenk1] = NINf.inv_maxmin_corr(pred1.detach()[:,:lenk1], max_corr, min_corr)
            y1.detach()[:,:lenk1] = NINf.inv_maxmin_corr(y1.detach()[:,:lenk1], max_corr, min_corr)
            pred1.detach().numpy()[:,lenk1+1:2*lenk1+1] *= (max_corr - min_corr)

        fig, ax = plt.subplots(2,5, figsize=(4*5, 3*2))
    
        for i in range(batch_test):
            k = i//5
            j = i if i < 5 else i -5
            ax[k,j].errorbar(k_plot, 1 - pred1.detach()[i,:lenk1]/y1.detach()[i,:lenk1], yerr=np.sqrt((pred1.detach().numpy()[i,lenk1+1:2*lenk1+1]/y1.detach()[i,:lenk1])**2),
                              label='Corrected P0', alpha=.5)
            ax[k,j].errorbar(k_plot, 1 - P_corr_p[i,:]/P_corr_l[i,:], yerr=np.sqrt((P_corr_s[i,:]/P_corr_l[i,:])**2),
                              label='Corrected PCA', alpha=.5)
            ax[k,j].semilogx(k_plot, 1 - 1/y1.detach()[i,:-1], label='Before correction P0')
            ax[k,j].semilogx(k_plot, 1 - 1/P_corr_l[i,:], label='Before correction PCA')
            ax[k,j].axhline(0, color='k', linestyle='dashed')
            ax[k,j].fill_between(k_plot, -0.01, 0.01, color='k', alpha=.1)
            ax[k,j].set_xlabel('$k$')
            ax[k,j].set_ylabel('Relative error')
            ax[0,0].legend()
            f_true1 = y1.detach()[i,lenk1]
            f_true2 = y2.detach()[i,lenk2]
            f_pred1 = pred1.detach()[i,lenk1]
            f_erro1 = pred1.detach()[i,-1]
            f_pred2 = pred2.detach()[i,lenk2]
            f_erro2 = pred2.detach()[i,-1]
            if norm_fs:
                f_true1 = NINf.inv_minmax(f_true1)
                f_true2 = NINf.inv_minmax(f_true2)
                f_pred1 = NINf.inv_minmax(f_pred1)
                f_erro1 *= 0.1
                f_pred2 = NINf.inv_minmax(f_pred2)
                f_erro2 *= 0.1
            ax[k,j].set_title(f'f_true = {f_true1:.4f} {f_true2:.4f} \n f_meas = {f_pred1:.4f} +- {f_erro1:.4f} \n f_meas = {f_pred2:.4f} +- {f_erro2:.4f}')
            #plt.semilogx(Pk_int[sel,0], y.detach()[i,:])
        plt.tight_layout()
        plt.show()

        if count == N_batches: return
        else: count += 1

    return

def plot_mean_error_f(test, model, k_max, ks, device, lenk, f_min, f_max=0.11, k_min=0, norm_fs=False, max_corr=None, min_corr=None, lim=[None,None], plot=True, save=None):

    f_thresh_low = NINf.minmax(f_min)
    f_thresh_top = NINf.minmax(f_max)
    self = (test.labels[:,-1] > f_thresh_low) & (test.labels[:,-1] < f_thresh_top)
    new_test = NIN.PkDataset(test.Pks_int[self], test.labels[self], test.norm[self])

    test_dataloader_all = DataLoader(new_test, batch_size=len(new_test), shuffle=False)
    j = 1

    for X, y, n in test_dataloader_all:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        f_y = y.detach().numpy()[:,lenk]
        if norm_fs:
            f_y = NINf.inv_minmax(f_y)

        pred_n = pred.detach().numpy()[:,:lenk * j]
        y_n = y.detach().numpy()[:,:lenk * j]
        pred_std = pred.detach().numpy()[:,lenk*j+1:lenk*j*2+1]

        if max_corr is not None:
            pred_n[:,:lenk*j] = NINf.inv_maxmin_corr(pred_n[:,:lenk*j], max_corr, min_corr)
            y_n[:,:lenk*j] = NINf.inv_maxmin_corr(y_n[:,:lenk*j], max_corr, min_corr)
            pred_std *= (max_corr - min_corr)

        p_mean = np.mean(pred_n, axis=0)
        y_mean = np.mean(y_n, axis=0)
        m = 1 - np.mean(pred_n/y_n, axis=0)
        std_mean = np.sqrt(np.mean(pred_std**2, axis=0))
        sm = np.mean(np.sqrt(pred_std**2)/np.abs(y_n), axis=0)/np.sqrt(np.sum(self))
    
    if plot == True:
        sel = (ks> k_min) & (ks < k_max)
        k_plot = ks[sel]
    
               
        plt.errorbar(k_plot, 1 - p_mean[:lenk]/y_mean[:lenk], yerr=std_mean[:lenk]/np.abs(y_mean[:lenk])/np.sqrt(np.sum(self)))
        plt.axhline(0, color='k', linestyle='dashed')
        plt.xscale('log')
        plt.xlabel('$k$')
        plt.ylim(lim[0],lim[1])
        plt.ylabel('''Relative error \n of mean correction''')
        plt.title(f'${f_min} < f < {f_max}$')
    
        plt.fill_between([k_min, k_max], -0.01, 0.01, alpha=.2, color='k')
        
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        plt.show()

    return p_mean, y_mean, std_mean, m, sm, np.sum(self)