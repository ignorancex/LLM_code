import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import median_abs_deviation as mad
import tensorflow as tf
from time import time
from pim import PredictionInterval
from sqr import *
from scipy.stats import norm, beta
import argparse
import random

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# QD hyperparameters
lambda_ = 0.01 # lambda in loss fn
alpha_ = 0.05  # capturing (1-alpha)% of samples
soften_ = 160.
n_ = 100 # batch size

def qd_objective(y_true, y_pred):
    '''Loss_QD-soft, from algorithm 1'''
    y_true = y_true[:,0]
    y_u = y_pred[:,0]
    y_l = y_pred[:,1]
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_true))
    K_HL = tf.maximum(0.,tf.sign(y_true - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    K_SU = tf.sigmoid(soften_ * (y_u - y_true))
    K_SL = tf.sigmoid(soften_ * (y_true - y_l))
    K_S = tf.multiply(K_SU, K_SL)
    
    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/tf.reduce_sum(K_H)
    PICP_S = tf.reduce_mean(K_S)
    
    Loss_S = MPIW_c + lambda_ * n_ / (alpha_*(1-alpha_)) * (tf.maximum(0.,(1-alpha_) - PICP_S)**2)
    
    return Loss_S

def qd_two_point_model():
    model = Sequential()
    model.add(Dense(100, input_dim=1, activation='relu',
                    kernel_initializer = RandomNormal(mean=0.0, stddev=0.2)))                   
    model.add(Dense(2, activation='linear',
                    kernel_initializer = RandomNormal(mean=0.0, stddev=0.3), 
                    bias_initializer = Constant(value=[3.,-3.])))
    model.compile(loss=qd_objective, optimizer=Adam(lr=0.02, decay=0.01))
    return model

def sqr_two_point_model():
    model = Sequential()
    model.add(Dense(100, input_dim=1, activation='relu',
                    kernel_initializer = RandomNormal(mean=0.0, stddev=0.2)))                   
    model.add(SQ_Dense(units = 1, n_quant = 2, activation='linear',
                    kernel_initializer = RandomNormal(mean=0.0, stddev=0.3), 
                    bias_initializer = Constant(value=[3.,-3.])))
    model.compile(loss=SQ_Loss(tau=[1-alpha_/2, alpha_/2]), optimizer=Adam(lr=0.02, decay=0.01))
    return model

def one_point_model():
    model = Sequential()
    model.add(Dense(100, input_dim=1, activation='relu',
                    kernel_initializer = RandomNormal(mean=0.0, stddev=0.2)))
    model.add(Dense(1, activation='linear',
                    kernel_initializer = RandomNormal(mean=0.0, stddev=0.3), 
                    bias_initializer = Constant(value=[3.]))) 
    model.compile(loss='mse', optimizer=Adam(lr=0.02, decay=0.01))
    return model

class GenData():
    def __init__(self, distrib='normal'):
        self.x_sampler = lambda n: np.random.uniform(xmin, xmax, size=(n,1))
        self.loc = lambda x: 0.3*np.sin(np.pi*x)
        self.scale = lambda x: 0.2*x**2
        if distrib == 'normal':
            self.noise = lambda x: norm.rvs(0., self.scale(x))
            self.quantile = lambda x, p: norm.ppf(p, self.loc(x), self.scale(x))
        elif distrib == 'beta':
            a, b = 0.2, 0.3
            self.noise = lambda x: beta.rvs(a, b, 0., self.scale(x))
            self.quantile = lambda x, p: beta.ppf(p, a, b, self.loc(x), self.scale(x))

    def generate(self, n_samples, x='fixed', seed=3):
        reset_seeds(seed)
        x = x_grid if x == 'fixed' else self.x_sampler(n_samples)
        y = self.loc(x) + self.noise(x)  
        return x, y

def ensemble_true_one_pred(trial, x_train, y_train, data, n_samples, n_ens):
    y_true = np.empty((n_ens, len(x_grid)))
    for k in range(n_ens):
        _, y_true[k,:] = data.generate(n_samples, x='fixed', seed=trial+k)
    es = EarlyStopping(monitor='loss', mode='min', patience=100)
    # Generate one prediction    
    model = one_point_model()
    t_ini = time()
    model.fit(x_train, y_train, 
              epochs=args.n_epochs, 
              batch_size=n_, 
              verbose=0, 
              callbacks=[es])
    y_pred = np.squeeze(model.predict(x_grid, verbose=0))
    t_end = time()
    print(f'one-point model took: {t_end - t_ini} sec')
    return y_true, y_pred, t_end - t_ini
    
def pim_ens(trial, p, x_train, y_train, data, n_samples, n_ens, err_type='sym'):
    y_true, y_pred, time_1p = ensemble_true_one_pred(trial, x_train, y_train, data, n_samples, n_ens)
    err = y_true - y_pred[np.newaxis,:]
    umodel = PredictionInterval(p, beta=1e3, err_type=err_type)
    umodel.compile(loss=umodel.pim.loss, optimizer=Adam(lr=0.005))
    t_ini = time()
    hist = umodel.fit(err, err, 
                   epochs=1000, 
                   batch_size=n_ens, 
                   callbacks=[umodel.early_stop], 
                   verbose=0)
    t_end = time()
    print(f'PIM took: {t_end - t_ini} sec')
    rad = umodel.pim.radius
    loss = hist.history['loss']
    time_pim = time_1p + t_end - t_ini
    return y_pred, rad, loss, umodel.pim.picp(err.astype('float32')), time_pim

def run_toy(trial, distrib, n_samples, n_ens, n_epochs, objective_2point='QD'):
    data = GenData(distrib=distrib)
    x_train, y_train = data.generate(n_samples, x='rand', seed=trial+n_ens+1)
    # Enlarge train set of QD by the n_ens used by PIM
    x_train_e, y_train_e = data.generate(n_ens, x='rand', seed=trial+n_ens+1)
    x_train_e = np.vstack((x_train, x_train_e))
    y_train_e = np.vstack((y_train, y_train_e))
    ## Train two point model
    print(f'Training {objective_2point} for {distrib} error distribution...')
    if objective_2point == 'QD':
        model = qd_two_point_model()
    elif objective_2point == 'SQR':
        model = sqr_two_point_model()
    es = EarlyStopping(monitor='loss', mode='min', patience=100)
    t_ini = time()
    hist_2p = model.fit(x_train_e, y_train_e, 
                        epochs=n_epochs, 
                        batch_size=n_, 
                        verbose=0, 
                        validation_split=0., 
                        callbacks=[es])
    y_pred = model.predict(x_grid, verbose=0)
    t_end = time()
    print(f'{objective_2point} took: {t_end - t_ini} sec')
    y_u_pred = np.squeeze(y_pred[:,0])
    y_l_pred = np.squeeze(y_pred[:,1])
    loss_2p = hist_2p.history['loss']
    time_2p = t_end - t_ini
    ## Train one point model -> PIM
    print(f'Training PIM for {distrib} error distribution...')
    err_type = 'sym' if distrib == 'normal' else 'asym'
    y_m_pred, rad, loss_pim, picp_pim, time_pim = pim_ens(trial, 0.95, x_train, y_train,
                                                          data, n_samples, n_ens, 
                                                          err_type=err_type)
    y_u_pim, y_l_pim = y_m_pred + rad[1], y_m_pred - rad[0]
    return x_train, y_train, y_u_pred, y_l_pred, y_u_pim, y_l_pim, data, picp_pim, time_pim, time_2p

def normal_vs_beta_noise(trial, n_samples, n_ens, n_epochs, obj_2pt):
    print('PIM vs QD')
    (x_train_n, y_train_n, y_u_pred_n, y_l_pred_n, 
    y_u_pim_n, y_l_pim_n, data_n, _, _, _) = run_toy(trial, 'normal', 
                                                    n_samples, 
                                                    n_ens, 
                                                    n_epochs,
                                                    objective_2point='QD')
    (x_train_b, y_train_b, y_u_pred_b, y_l_pred_b, 
    y_u_pim_b, y_l_pim_b, data_b, _, _, _) = run_toy(trial, 'beta', 
                                                    n_samples, 
                                                    n_ens, 
                                                    n_epochs,
                                                    objective_2point='QD')
    print('PIM vs SQR')
    _, _, y_u_pred_n_sqr, y_l_pred_n_sqr, _, _, _, _, _, _ = run_toy(trial, 'normal', 
                                                                     n_samples, 
                                                                     n_ens, 
                                                                     n_epochs,
                                                                     objective_2point='SQR')

    _, _, y_u_pred_b_sqr, y_l_pred_b_sqr, _, _, _, _, _, _ = run_toy(trial, 'beta', 
                                                                     n_samples, 
                                                                     n_ens, 
                                                                     n_epochs,
                                                                     objective_2point='SQR')


    # fig, axs = plt.subplots(1, 2)
    # # Plot normal error
    # axs[0].scatter(x_train_n, y_train_n, c='k', s=10)
    # axs[0].plot(x_grid, data_n.quantile(x_grid, 0.5*(1+0.95)), 'k')
    # axs[0].plot(x_grid, data_n.quantile(x_grid, 0.5*(1-0.95)), 'k')
    # axs[0].plot(x_grid, data_n.quantile(x_grid, 0.5), 'k', lw=0.5)
    # axs[0].fill_between(x_grid, y_l_pim_n, y_u_pim_n, color='r', alpha=0.2)
    # axs[0].fill_between(x_grid, y_l_pred_n, y_u_pred_n, color='gray', alpha=0.4)
    # axs[0].fill_between(x_grid, y_l_pred_n_sqr, y_u_pred_n_sqr, color='g', alpha=0.2)
    # axs[0].tick_params(axis='both', labelsize=13)
    # axs[0].set_xlabel('x', fontsize=14)
    # axs[0].set_ylabel('y', fontsize=14)
    # axs[0].text(-0.9, 1.3, 'Symmetric', fontsize=14)
    # axs[0].text(-0.9, 1.0, 'Unimodal', fontsize=14)
    # # Plot beta error
    # axs[1].scatter(x_train_b, y_train_b, c='k', s=10)
    # axs[1].plot(x_grid, data_b.quantile(x_grid, 0.5*(1+0.95)), 'k')
    # axs[1].plot(x_grid, data_b.quantile(x_grid, 0.5*(1-0.95)), 'k')
    # axs[1].plot(x_grid, data_b.quantile(x_grid, 0.5), 'k', lw=0.5)
    # axs[1].fill_between(x_grid, y_l_pim_b, y_u_pim_b, color='r', alpha=0.2)
    # axs[1].fill_between(x_grid, y_l_pred_b, y_u_pred_b, color='gray', alpha=0.4)
    # axs[1].fill_between(x_grid, y_l_pred_b_sqr, y_u_pred_b_sqr, color='g', alpha=0.2)
    # axs[1].tick_params(axis='both', labelsize=13)
    # # for ax in axs: ax.set(xlabel='x', ylabel='y')
    # axs[1].set_xlabel('x', fontsize=14)
    # axs[1].set_ylabel('y', fontsize=14)
    # axs[1].text(-0.7, 1.02, 'Skewed', fontsize=14)
    # axs[1].text(-0.7, 0.87, 'Bimodal', fontsize=14)
    # plt.subplots_adjust(wspace=0.4)
    # plt.tight_layout()
    # plt.savefig("paper/toy.pdf")
    # plt.show()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(1, 2)
    # Plot normal error
    axs[0].scatter(x_train_n, y_train_n, c='k', s=10)
    axs[0].plot(x_grid, data_n.quantile(x_grid, 0.5*(1+0.95)), 'k')
    axs[0].plot(x_grid, data_n.quantile(x_grid, 0.5*(1-0.95)), 'k')
    axs[0].plot(x_grid, data_n.quantile(x_grid, 0.5), 'k', lw=0.5)
    pl_pim = axs[0].plot(x_grid, y_l_pim_n, color=colors[0])
    pu_pim = axs[0].plot(x_grid, y_u_pim_n, color=colors[0])
    pl_qd = axs[0].plot(x_grid, y_l_pred_n, color=colors[1])
    pu_qd = axs[0].plot(x_grid, y_u_pred_n, color=colors[1])
    pl_sqr = axs[0].plot(x_grid, y_l_pred_n_sqr, color=colors[2])
    pu_sqr = axs[0].plot(x_grid, y_u_pred_n_sqr, color=colors[2])
    axs[0].legend([(pl_pim[0], pu_pim[0]), 
                   (pl_qd[0], pu_qd[0]),
                   (pl_sqr[0], pu_sqr[0])], 
                   ['PIM', 'QD', 'SQR'], 
                   loc='lower center', prop={'size':16})
    axs[0].tick_params(axis='both', labelsize=13)
    axs[0].set_xlabel('$x$', fontsize=14)
    axs[0].set_ylabel('$y$', fontsize=14)
    axs[0].text(-0.9, 1.3, 'Symmetric', fontsize=14)
    axs[0].text(-0.9, 1.0, 'Unimodal', fontsize=14)
    # Plot beta error
    axs[1].scatter(x_train_b, y_train_b, c='k', s=10)
    axs[1].plot(x_grid, data_b.quantile(x_grid, 0.5*(1+0.95)), 'k')
    axs[1].plot(x_grid, data_b.quantile(x_grid, 0.5*(1-0.95)), 'k')
    axs[1].plot(x_grid, data_b.quantile(x_grid, 0.5), 'k', lw=0.5)
    pl_pim = axs[1].plot(x_grid, y_l_pim_b, color=colors[0])
    pu_pim = axs[1].plot(x_grid, y_u_pim_b, color=colors[0])
    pl_qd = axs[1].plot(x_grid, y_l_pred_b, color=colors[1])
    pu_qd = axs[1].plot(x_grid, y_u_pred_b, color=colors[1])
    pl_sqr = axs[1].plot(x_grid, y_l_pred_b_sqr, color=colors[2])
    pu_sqr = axs[1].plot(x_grid, y_u_pred_b_sqr, color=colors[2])
    axs[1].tick_params(axis='both', labelsize=13)
    # for ax in axs: ax.set(xlabel='x', ylabel='y')
    axs[1].set_xlabel('$x$', fontsize=14)
    axs[1].set_ylabel('$y$', fontsize=14)
    axs[1].text(-0.7, 1.02, 'Skewed', fontsize=14)
    axs[1].text(-0.7, 0.87, 'Bimodal', fontsize=14)
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig("paper/toy.pdf")
    plt.show()

def run_trials(obj_2pt, distrib):
    times_pim = []
    times_2pt = []
    rmse_pim = []
    rmse_2pt = []

    t_ini = time()
    for trial in range(args.n_trials):
        print(f'Trial {trial+1}/{args.n_trials}')
        (x_train, y_train, y_u_pred, y_l_pred, 
        y_u_pim, y_l_pim, data, picp_pim, time_pim, time_2pt) = run_toy(trial,
                                                                        distrib, 
                                                                        args.n_samples, 
                                                                        args.n_ens, 
                                                                        args.n_epochs,
                                                                        objective_2point=obj_2pt)
        pi_2pt = y_u_pred - y_l_pred
        pi_pim = y_u_pim - y_l_pim
        pi_exact = data.quantile(x_grid, 0.5*(1+0.95)) - data.quantile(x_grid, 0.5*(1-0.95))
        rmse_pim.append(np.sqrt(((pi_pim - pi_exact)**2).mean()))
        rmse_2pt.append(np.sqrt(((pi_2pt - pi_exact)**2).mean()))
        times_pim.append(time_pim)
        times_2pt.append(time_2pt)
    
    total_time = time() - t_ini
    time_2pt = np.asarray(times_2pt) / total_time
    time_pim = np.asarray(times_pim) / total_time
    rmse_2pt = np.asarray(rmse_2pt)
    rmse_pim = np.asarray(rmse_pim)

    print('------------------------------------')
    print(f'rmse_{obj_2pt} = {np.median(rmse_2pt)} +/- {mad(rmse_2pt)}')
    print(f'rmse_pim = {np.median(rmse_pim)} +/- {mad(rmse_pim)}')
    print(f'time_{obj_2pt} = {np.median(time_2pt)} +/- {mad(time_2pt)}')
    print(f'time_pim = {np.median(time_pim)} +/- {mad(time_pim)}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distrib', type=str, default='normal')
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_ens', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--plot', type=bool, default=False)
    args = parser.parse_args()

    xmin, xmax = -2., 2.
    x_grid = np.linspace(xmin, xmax, args.n_samples) 
    obj_2pt = 'SQR'
    distrib = 'normal'

    # run_trials(obj_2pt, distrib)
    normal_vs_beta_noise(0, args.n_samples, args.n_ens, args.n_epochs, obj_2pt)

    


    
    