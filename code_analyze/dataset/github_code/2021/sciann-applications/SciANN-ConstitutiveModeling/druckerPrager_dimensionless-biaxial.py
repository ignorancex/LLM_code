# Copyright 2022 SciANN -- Ehsan Haghighat. 
# All Rights Reserved.
#
# Licensed under the MIT License.
# 
# druckerPrager_dimensionless-biaxial.py
# 
#   Main code for characterizing Drucker-Prager model for biaxial loading
# 
# Requirements:
# - data/druckerPrager-biaxial.csv
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

EPOCH_MAX = 50_000

# true values.
NU = 0.25
G_TRUE = 100e6
K_TRUE = 2/3*(1+NU)/(1-2*NU)*G_TRUE
SY_TRUE = 100.e3
BETA_TRUE = np.radians(30.0)
ETA_TRUE = np.tan(BETA_TRUE)
SHAPE = 1.
VOID0 = 0.5


inverted_parameters = {"delta": [0.], "G": [G_TRUE], "K": [K_TRUE], "Sy": [SY_TRUE], "eta":[ETA_TRUE]}


DTYPE = 'float32'


# data
data = pd.read_csv('data/druckerPrager-biaxial.csv')


class Normalizer:
    def __init__(self, x):
        self.mu = x.mean()
        self.sig = x.std()
        
    def transform(self, x):
        return (x - self.mu) / self.sig
      
    def inv_transform(self, x):
        return x * self.sig + self.mu
      
def softstep(x):
    return (tanh(100*x) + 1.0)/2.0

def sigmoidf(x, a=100):
    return sigmoid(a*x)

def gaussian(x, a=100):
    return exp(-(x*a)**2)
  
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


G_TRUE /= ELS_STAR
K_TRUE /= ELS_STAR
SY_TRUE /= SIG_STAR


input_data['t'] = data['t'].values.reshape(-1,1)
input_data['dt'] = np.gradient(data['t'].values).reshape(-1,1)


scalar["t"] = Normalizer(input_data['t'])


for v in ['E1', 'E2', 'E3', 'gamma', 'Ep_q', 'Eq']:
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
    p, q, r = S.p(), S.q(), S.r()

    # Incremental stress components 
    dS1 = sn.Variable('dS1', dtype=DTYPE)
    dS2 = sn.Variable('dS2', dtype=DTYPE)
    dS3 = sn.Variable('dS3', dtype=DTYPE)
    dS = Tensor([dS1, dS2, dS3])

    # Deviatoric stress components 
    ds1, ds2, ds3 = dS.d()
    dp, dq, dr = dS.p(), dS.q(), dS.r()

    t_s = scalar['t'].transform(t)
    g = sn.Functional('g', [t_s], 8*[20], 'tanh')
    dg = diff(g, t)


    bulk_par = sn.Parameter(np.random.rand(), inputs=[t], min_max=[0.1, 10], name='bulk')
    bulk = bulk_par*K_TRUE
    
    shear_par = sn.Parameter(np.random.rand(), inputs=[t], min_max=[0.1, 10], name='shear')
    shear = shear_par*G_TRUE

    lame = bulk - shear*(2./3.) 

    eta_par = sn.Parameter(np.random.rand(), inputs=[t], min_max=[0.1, 10], name='eta')
    eta = eta_par*ETA_TRUE

    sy_par = sn.Parameter(1., inputs=[t], min_max=[0.5, 2], name='SY0')
    sy = sy_par*SY_TRUE
    
    params = [bulk_par, shear_par, eta_par, sy_par]
    params_mult = [K_TRUE, G_TRUE, ETA_TRUE, SY_TRUE]


    R = 0.5*(1 + 1/SHAPE) # - (1-SHAPE)*(r/q)**3)

    G = R*q - p*eta
    dG_dp = -eta
    dG_dq = R

    Eeq_q = g * dG_dq
    dEeq_q = dg * dG_dq

    py = sy
    dpy = 0.

    F = G - py
    dF_dp = dG_dp
    dF_dq = dG_dq
    dF_dg = -dpy*dF_dp

    dF = S.dp()*dF_dp + S.dq()*dF_dq

    Ce = np.array([[lame + 2*shear, lame, lame], 
                   [lame, lame + 2*shear, lame],
                   [lame, lame, lame + 2*shear]])

    ce_jac = (3.*lame + 2.*shear)*shear
    Ce_inv = np.array([[lame+shear, -lame/2., -lame/2.],
                       [-lame/2., lame+shear, -lame/2.],
                       [-lame/2., -lame/2., lame+shear]]) / ce_jac

    dFN = np.matmul(Ce, dF)
    
    He = np.dot(dF, dFN)
    Hp = dF_dg

    dE_e = np.matmul(Ce_inv, dS())
    dE_e_v = sum(dE_e)

    dE_p = dF*dg
    dE_tr = dE_e + dE_p
    dE_tr_v = sum(dE_tr)

    dGamma = np.dot(dFN, dE())/(He - Hp)
    dLoad = np.dot(dF, dS())
    # dLoad = np.dot(dFN, dE())

    inputs = [t, 
        E1, E2, E3, dE1, dE2, dE3, 
        S1, S2, S3, dS1, dS2, dS3]

    dL_scaler = np.abs(input_data['dS1']).max()

    targets = [
        sigmoidf(-g) * abs(g),
        sigmoidf(-dg) * abs(dg),
        sigmoidf(F) * abs(F),
        (dg) * (F),
        sigmoidf(F) * (dg - dGamma),
        sigmoidf(-F)*(dE1 - dE_e[0]),
        sigmoidf(-F)*(dE2 - dE_e[1]),
        sigmoidf(-F)*(dE3 - dE_e[2]),
        sigmoidf(-F)*(dEv - dE_e_v),
        (dLoad < 0.) * (dE1 - dE_e[0]),
        (dLoad < 0.) * (dE2 - dE_e[1]),
        (dLoad < 0.) * (dE3 - dE_e[2]),
        (dLoad < 0.) * (dEv - dE_e_v),
        (dLoad > 0.) * sigmoidf(F) * (dE1 - dE_tr[0]),
        (dLoad > 0.) * sigmoidf(F) * (dE2 - dE_tr[1]),
        (dLoad > 0.) * sigmoidf(F) * (dE3 - dE_tr[2]),
        (dLoad > 0.) * sigmoidf(F) * (dEv - dE_tr_v),
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

    # Data driven fit.
    m1 = sn.SciModel(inputs, [g] + targets, optimizer='adam')

    ls_scheduler = {'scheduler': 'exponentialdecay', 
                    'initial_learning_rate':0.002, 
                    'final_learning_rate': 0.0002, 
                    'decay_epochs': 40_000,
                    'delay_epochs': 10_000}

    h= m1.train(training_inputs, 
                [(np.array([0]), 0.0)] + training_data,
                learning_rate=ls_scheduler,
                batch_size=10000, 
                epochs=EPOCH_MAX,
                log_parameters=dict(parameters=params, freq=1),
                reduce_lr_after=1000,
                stop_after=2000,
                shuffle=False,
                verbose=1)

    m1.save_weights(path + f"d-{delta}.hdf5")
    

    bulk_pred = bulk.eval(m1, training_inputs).mean()
    shear_pred = shear.eval(m1, training_inputs).mean()
    eta_pred = eta.eval(m1, training_inputs).mean()
    sy_pred = sy.eval(m1, training_inputs).mean()

    to_write = 'Training for $\\delta$={} is completed!  \n'.format(delta)
    to_write += f'{bulk_pred = } v.s. {K_TRUE} \n'
    to_write += f'{shear_pred = } v.s. {G_TRUE} \n'
    to_write += f'{eta_pred = } v.s. {ETA_TRUE} \n'
    to_write += f'{sy_pred = } v.s. {SY_TRUE} \n \n'
    print(to_write)


    inverted_parameters["delta"] += [delta]
    inverted_parameters["G"] += [np.abs(shear_pred/G_TRUE - 1.)]
    inverted_parameters["K"] += [np.abs(bulk_pred/K_TRUE - 1.)]
    inverted_parameters["Sy"] += [np.abs(sy_pred/SY_TRUE - 1.)]
    inverted_parameters["eta"] += [np.abs(eta_pred/ETA_TRUE - 1.)]


    mode = 'a' if idelta>0 else 'w'
    with open(path + "params.txt", mode) as f:
        f.write(to_write)

    g_pred = (g.eval(m1, training_inputs)*EPS_STAR)
    ax[0].semilogy(h.history['loss'], label='$\delta$={}'.format(delta))
    ax[1].plot(training_inputs[0], g_pred, label='$\delta$={}'.format(delta))
    ax[2].loglog(4*[delta], [eta_pred, sy_pred, shear_pred, bulk_pred], '*')

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
ax[0].set_ylabel('MSE')
ax[0].legend()

ax[1].set_xlabel('t')
ax[1].set_ylabel('$\gamma$')
ax[1].legend()
ax[1].plot(training_inputs[0], data['gamma'].values, 'k')

lw=3.0
for var, comp in zip([ETA_TRUE, SY_TRUE, G_TRUE, K_TRUE], 
                     ['$tan(\\beta)$', '$C$', '$\\mu$', '$\\kappa$']):
    ax[2].loglog(delta_vals, np.ones_like(delta_vals)*var, '--k', lw=lw, label=comp)
    lw -= 0.5

ax[2].set_xlabel('$\\delta$')
ax[2].set_ylabel('$\\eta, ~C, ~\\mu, ~\\kappa$')
ax[2].legend()




fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.2, hspace=0.3)
plt.savefig(path + "results.pdf", dpi=300)
plt.savefig(path + "results.png", dpi=300)



