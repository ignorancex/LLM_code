# Copyright 2022 SciANN -- Ehsan Haghighat. 
# All Rights Reserved.
#
# Licensed under the MIT License.
# 
# vonMises_isotropic_dimensionless.py
# 
#   Main code characterizing vonMises model with isotropic hardening
# 
# Requirements:
# - data/vonMises.csv
# 

import numpy as np 
import matplotlib.pyplot as plt 
from Tensor import SnTensor as Tensor
import sciann as sn 
import pandas as pd
from sciann.utils.math import sign, relu, abs, diff, step, tanh, sigmoid, exp


import sys    
import os    
file_name =  os.path.basename(sys.argv[0]).split('.')[0]
if not os.path.exists(file_name):
    os.mkdir(file_name)
path = os.path.join(file_name, 'res_')


# true values.
E_true = 200e9 
nu_true = 0.2
mu_true = E_true/2/(1+nu_true)
lmbd_true = E_true*nu_true/(1+nu_true)/(1-2*nu_true)
bulk_true = lmbd_true + 2/3*mu_true
Sy_true = 200e6
Hp_true = 10e9


inverted_parameters = {
    "delta": [0.], 
    "G": [mu_true], 
    "K": [bulk_true], 
    "Sy": [Sy_true], 
    "Hp":[Hp_true]
}


# data
data = pd.read_csv('data/vonMises.csv')

class Normalizer:
    def __init__(self, x):
        self.mu = x.mean()
        self.sig = x.std()
        
    def transform(self, x):
        return (x - self.mu) / self.sig
      
    def inv_transform(self, x):
        return x * self.sig + self.mu


input_data = {}
scalar = {}

t = data['t'].values
dt = np.gradient(t)

sig = data['S1'].values
dsig = np.gradient(sig)/dt

eps = data['E1'].values
deps = np.gradient(eps)/dt

SIG_STAR = np.abs(sig).max()
EPS_STAR = np.abs(eps).max()
ELS_STAR = SIG_STAR/EPS_STAR


mu_true /= ELS_STAR
bulk_true /= ELS_STAR
lmbd_true /= ELS_STAR
Hp_true /= ELS_STAR
Sy_true /= SIG_STAR


input_data['t'] = data['t'].values.reshape(-1,1)
input_data['dt'] = np.gradient(data['t'].values).reshape(-1,1)

scalar["t"] = Normalizer(input_data['t'])

for v in ['E1', 'E2', 'E3', 'Eq', 'Ep_q']:
    x = data[v].values / EPS_STAR
    input_data[v] = x.reshape(-1,1)
    dx = (np.gradient(x).reshape(-1,1) / input_data['dt'])
    input_data["d"+v] = dx

for v in ['S1', 'S2', 'S3', 'Sq']:
    x = data[v].values / SIG_STAR
    input_data[v] = x.reshape(-1,1)
    dx = (np.gradient(x).reshape(-1,1) / input_data['dt'])
    input_data["d"+v] = dx


fig, ax = plt.subplots(1, 3, figsize=(15, 4))


delta_vals = [10, 50, 100, 200, 500]
for idelta, delta in enumerate(delta_vals):

    def sigmoidf(x):
        return sigmoid(delta*x)
    
    sn.reset_session()
    sn.set_random_seed(12345)

    DTYPE = 'float32'

    # Variables
    t =  sn.Variable('t', dtype=DTYPE)

    # Strain components 
    E1 = sn.Variable('E1', dtype=DTYPE)
    E2 = sn.Variable('E2', dtype=DTYPE)
    E3 = sn.Variable('E3', dtype=DTYPE)
    E = Tensor([E1, E2, E3])
    # Deviatoric strain components
    e1, e2, e3 = E.d()
    Ev, Eeq = E.v(), E.eq()

    # Incremental strain components 
    dE1 = sn.Variable('dE1', dtype=DTYPE)
    dE2 = sn.Variable('dE2', dtype=DTYPE)
    dE3 = sn.Variable('dE3', dtype=DTYPE)
    dE = Tensor([dE1, dE2, dE3])
    # Deviatoric incremental strain components 
    de1, de2, de3 = dE.d()
    dEv, dEeq = dE.v(), dE.eq()

    # Stress components 
    S1 = sn.Variable('S1', dtype=DTYPE)
    S2 = sn.Variable('S2', dtype=DTYPE)
    S3 = sn.Variable('S3', dtype=DTYPE)
    S = Tensor([S1, S2, S3])
    # Deviatoric stress components 
    s1, s2, s3 = S.d()
    p, q = S.p(), S.q()

    # Incremental stress components 
    dS1 = sn.Variable('dS1', dtype=DTYPE)
    dS2 = sn.Variable('dS2', dtype=DTYPE)
    dS3 = sn.Variable('dS3', dtype=DTYPE)
    dS = Tensor([dS1, dS2, dS3])
    # Deviatoric stress components 
    ds1, ds2, ds3 = dS.d()
    dp, dq = dS.p(), dS.q()


    t_s = scalar['t'].transform(t)
    g = sn.Functional('g', [t_s], 8*[20], 'tanh')
    dg = diff(g, t)


    lmbd_par = sn.Parameter(np.random.rand(), inputs=[t_s], name='lmbd')
    lmbd = lmbd_par*lmbd_true
    mu_par = sn.Parameter(np.random.rand(), inputs=[t_s], name='mu')
    mu = mu_par*mu_true
    bulk = lmbd + 2*mu/3

    SY0_par = sn.Parameter(np.random.rand(), inputs=[t_s], name='SY0')
    SY0 = SY0_par*Sy_true
    Hp_par = sn.Parameter(np.random.rand(), inputs=[t_s], name='Hp')
    Hp = Hp_par*Hp_true
    Sy = SY0 + Hp*g


    params = [lmbd_par, mu_par, SY0_par, Hp_par]
    params_mult = [lmbd_true, mu_true, Sy_true, Hp_true]


    F = q - Sy
    dF = dq - Hp*dg

    Ce = np.array([[lmbd + 2*mu, lmbd, lmbd], 
                   [lmbd, lmbd + 2*mu, lmbd],
                   [lmbd, lmbd, lmbd + 2*mu]])
    
    n = np.array(S.dq())
    dEt = np.array(dE._t)
    He = np.dot(np.dot(n, Ce), n)

    dL = np.dot(np.dot(n, Ce), dEt) #/ (He + Hp)
    dL = np.dot(n, dS()) #/ (He + Hp)

    dGamma = np.dot(np.dot(n, Ce), dEt) / (He + Hp)

    e1_e = s1/(2.*mu)
    e2_e = s2/(2.*mu)
    e3_e = s3/(2.*mu)

    e1_p = g * n[0]
    e2_p = g * n[1]
    e3_p = g * n[2]

    Eeq_p = g

    de1_e = ds1/(2*mu)
    de2_e = ds2/(2*mu)
    de3_e = ds3/(2*mu)

    de1_p = dg * n[0]
    de2_p = dg * n[1]
    de3_p = dg * n[2]

    de1_tr = de1_e + de1_p
    de2_tr = de2_e + de2_p
    de3_tr = de3_e + de3_p

    dEeq_p = dg

    inputs = [t, 
            E1, E2, E3, dE1, dE2, dE3, 
            S1, S2, S3, dS1, dS2, dS3]

    dL_scaler = np.abs(input_data['dS1']).max()
    
    targets = [
        sigmoidf(-g) * abs(g),
        sigmoidf(-dg) * abs(dg),
        sigmoidf(F) * abs(F),
        (dg) * (F),
        sigmoidf(-F) * (de1 - de1_e),
        sigmoidf(-F) * (de2 - de2_e),
        sigmoidf(-F) * (de3 - de3_e),
        sigmoidf(-dL/dL_scaler) * (de1 - de1_e),
        sigmoidf(-dL/dL_scaler) * (de2 - de2_e),
        sigmoidf(-dL/dL_scaler) * (de3 - de3_e),
        sigmoidf(F)*(dg - dGamma),
        sigmoidf(dL/dL_scaler) * sigmoidf(F) * (de1 - de1_tr),
        sigmoidf(dL/dL_scaler) * sigmoidf(F) * (de2 - de2_tr),
        sigmoidf(dL/dL_scaler) * sigmoidf(F) * (de3 - de3_tr),
        (Ev + p/bulk),
    ]

    training_inputs = [
        input_data['t'], 
        input_data['E1'], 
        input_data['E2'],
        input_data['E3'],
        input_data['dE1'], 
        input_data['dE2'],
        input_data['dE3'],
        input_data['S1'], 
        input_data['S2'],
        input_data['S3'],
        input_data['dS1'], 
        input_data['dS2'],
        input_data['dS3'],
    ]

    training_data = len(targets)*['zeros']

    ls_scheduler = {'scheduler': 'exponentialdecay', 
                    'initial_learning_rate':0.001, 
                    'final_learning_rate': 0.00005,
                    'decay_epochs': 40_000,
                    'delay_epochs': 10_000}

    # Data driven fit.
    m1 = sn.SciModel(inputs, [g] + targets, loss_func='mse', optimizer='adam')
    h= m1.train(training_inputs, 
                [(np.array([0]), 0.0)] + training_data,
                learning_rate=ls_scheduler,
                batch_size=1000,
                epochs=50_000,
                log_parameters=dict(parameters=params, freq=1),
                # reduce_lr_after=1000,
                stop_after=5000,
                shuffle=False,
                verbose=-1)


    m1.save_weights(path + f"d-{delta}.hdf5")

    print('Training for $\\delta$={} is completed! '.format(delta))

    # print('Lmbd = {:12.4e}  vs  {:12.4e}'.format(lmbd.eval(m1, training_inputs).mean(), lmbd_true))
    # print('Mu = {:12.4e}  vs  {:12.4e}'.format(mu.eval(m1, training_inputs).mean(), mu_true))
    # print('Sy = {:12.4e}  vs  {:12.4e}'.format(SY0.eval(m1, training_inputs).mean(), Sy_true))
    # print('Hp = {:12.4e}  vs  {:12.4e}'.format(Hp.eval(m1, training_inputs).mean(), Hp_true))
    lmbd_pred = lmbd.eval(m1, training_inputs).mean()
    mu_pred = mu.eval(m1, training_inputs).mean()
    bulk_pred = bulk.eval(m1, training_inputs).mean()
    Sy_pred = SY0.eval(m1, training_inputs).mean()
    Hp_pred = Hp.eval(m1, training_inputs).mean()


    to_write = 'Training for $\\delta$={} is completed!  \n'.format(delta)
    to_write += f'{bulk_pred = } v.s. {bulk_true} \n'
    to_write += f'{mu_pred = } v.s. {mu_true} \n'
    to_write += f'{Sy_pred = } v.s. {Sy_true} \n'
    to_write += f'{Hp_pred = } v.s. {Hp_true} \n \n'
    print(to_write)


    inverted_parameters["delta"] += [delta]
    inverted_parameters["G"] += [mu_pred*ELS_STAR]
    inverted_parameters["K"] += [bulk_pred*ELS_STAR]
    inverted_parameters["Sy"] += [Sy_pred*SIG_STAR]
    inverted_parameters["Hp"] += [Hp_pred*ELS_STAR]


    mode = 'a' if idelta>0 else 'w'
    with open(path + "params.txt", mode) as f:
        f.write(to_write)

    g_pred = (g.eval(m1, training_inputs)*EPS_STAR)
    ax[0].semilogy(h.history['loss'], label='$\delta$={}'.format(delta))
    ax[1].plot(training_inputs[0], g_pred, label='$\delta$={}'.format(delta))
    ax[2].loglog(4*[delta], [lmbd_pred, mu_pred, Sy_pred, Hp_pred], '*')

    pd.DataFrame({'epochs': np.arange(len(h.history['loss'])),
                  'loss': h.history['loss']}).to_csv(path + f"loss-d{delta}.csv")
    pd.DataFrame({'t': input_data['t'].flatten(), 'gamma': g_pred.flatten()}).to_csv(path + f"gamma-d{delta}.csv")

    for v, v_mult in zip(params, params_mult):
        v_name = v.name 
        v_vals = np.array(h.history[v_name]).flatten()*v_mult
        pd.DataFrame({'epochs': np.arange(len(h.history['loss'])),
                      v_name: v_vals}).to_csv(path + f"{v_name}-d{delta}.csv")


df = pd.DataFrame.from_dict(inverted_parameters, orient='index')
df.to_csv(path + "params.csv")


ax[0].set_xlabel('epochs')
ax[0].set_ylabel('$\mathcal{L}$')
ax[0].legend()

ax[1].set_xlabel('t')
ax[1].set_ylabel('$\gamma$')
ax[1].legend()

ax[1].plot(training_inputs[0], data['Ep_q'].values, 'k')

lw=3.0
for var, comp in zip([Sy_true, Hp_true, lmbd_true, mu_true], 
                     ['$\\sigma_Y$', '$H_p$', '$\\lambda$', '$\\mu$']):
    ax[2].loglog(delta_vals, np.ones_like(delta_vals)*var, '--k', lw=lw, label=comp)
    lw -= 0.5

ax[2].set_xlabel('$\\delta$')
ax[2].set_ylabel('$\\sigma_Y, ~H_p, ~\\lambda, ~\\mu$')
ax[2].legend()

plt.legend()

fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig(path + "results.pdf", dpi=300)
plt.savefig(path + "results.png", dpi=300)
