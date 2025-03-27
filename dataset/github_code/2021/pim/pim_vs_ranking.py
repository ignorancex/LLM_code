import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from pim import PredictionInterval
from scipy.stats import norm
import matplotlib.pyplot as plt
from functools import partial

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

reset_seeds(3)

interpolation = ['linear', 'lower', 'higher', 'midpoint', 'nearest']
def pi_numpy(err, alpha, interp):
    q_lo = np.quantile(err, alpha/2, interpolation=interp)
    q_hi = np.quantile(err, 1-alpha/2, interpolation=interp)
    return q_hi-q_lo

def pi_exact_norm(alpha):
    return norm.ppf(1-alpha/2) - norm.ppf(alpha/2)

def pi_pim(err, alpha, err_type='sym'):
    umodel = PredictionInterval(1-alpha, beta=pim_beta, err_type=err_type)
    umodel.compile(loss=umodel.pim.loss, optimizer=Adam(lr=pim_lr))
    _ = umodel.fit(err, err, 
                    epochs=1000, 
                    batch_size=err.shape[0], 
                    callbacks=[umodel.early_stop], 
                    verbose=0)
    rad = umodel.pim.radius
    return rad[0] + rad[1]

def get_quantile_funcs(err, dist='norm', err_type='sym', b=None):
    alphas = np.arange(0.1, 1, 0.05)
    pi_numpy_arr = np.empty((alphas.shape[0],len(interpolation)))
    pi_pim_arr = np.empty((alphas.shape[0],))
    pi_exact_arr = np.empty((alphas.shape[0],))
    for i, alpha in enumerate(alphas):
        print(f'Running alpha={alpha}')
        pi_numpy_arr[i,:] = [pi_numpy(err, alpha, interp) for interp in interpolation]
        pi_pim_arr[i] = pi_pim(err, alpha, err_type=err_type)
        pi_exact_arr[i] = pi_exact_norm(alpha)

    min_pi_numpy = pi_numpy_arr.min(axis=1).reshape((-1,1))
    max_pi_numpy = pi_numpy_arr.max(axis=1).reshape((-1,1))

    pi_numpy_arr = np.concatenate((min_pi_numpy, max_pi_numpy), axis=1)
    results = np.concatenate((pi_numpy_arr, pi_pim_arr[:,None], pi_exact_arr[:,None]), axis=1)
    np.save(f'figs/{dist}/nobs{n_obs}_beta{pim_beta}_lr{pim_lr}.npy', results)

    plt.plot(alphas, pi_exact_arr, label='exact')
    plt.plot(alphas, pi_pim_arr, label='pim')
    plt.plot(alphas, pi_numpy_arr[:,0], 'k-', label='np_min')
    plt.plot(alphas, pi_numpy_arr[:,1], 'k-.', label='np_max')
    plt.legend(loc='best')
    plt.title(f'Number of observations: {n_obs}')
    plt.savefig(f'figs/{dist}/nobs{n_obs}_beta{pim_beta}_lr{pim_lr}.svg')
    plt.show()

def plot_rmse(dist='norm'):
    other = [200,300,400,500,1000]
    list_nobs = np.concatenate((np.arange(10,110,10), np.asarray(other)))
    out = np.empty((len(list_nobs), 3))
    for i, n_obs in enumerate(list_nobs):
        arrs = np.load(f'figs/{dist}/nobs{n_obs}_beta{pim_beta}_lr{pim_lr}.npy')
        pi_exact_arr = arrs[:other[-1],-1]
        pi_pim_arr = arrs[:other[-1],-2]
        pi_numpy_arr = arrs[:other[-1],:-2]
        # Distance from numpy to exact
        rmse_numpy_min = np.sqrt(((pi_numpy_arr[:,0] - pi_exact_arr)**2).mean())
        rmse_numpy_max = np.sqrt(((pi_numpy_arr[:,1] - pi_exact_arr)**2).mean())
        # Distance from pim to exact
        rmse_pim = np.sqrt(((pi_pim_arr - pi_exact_arr)**2).mean())
        out[i,:] = [rmse_numpy_min, rmse_numpy_max, rmse_pim]

    x = np.log10(list_nobs)
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(1, figsize=(10,4))
    p1 = ax.plot(x, out[:,0], color=color[1], alpha=0.3, lw=1)
    p2 = ax.plot(x, out[:,1], color=color[1], alpha=0.3, lw=1)
    ax.fill_between(x, out[:,0], out[:,1], color=color[1], alpha=0.3)
    p3 = ax.plot(x, out[:,2], lw=2)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend([(p2[0], p1[0]), p3[0]], ['Ranking', 'PIM'], loc='best', prop={'size':16})
    ax.set_xlabel(r'$log(m)$', labelpad=5, fontsize=18)
    ax.set_ylabel('RMSE', labelpad=10, fontsize=18)
    plt.tight_layout()
    plt.savefig(f'paper/rmse.svg')
    plt.show()

n_obs = 1000
pim_beta = 1e1
pim_lr = 0.01

# Gaussian error
err = np.random.normal(size=(n_obs,))
get_quantile_funcs(err)
# plot_rmse()



