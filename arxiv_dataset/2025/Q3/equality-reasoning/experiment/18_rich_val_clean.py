"""Validating rich-regime behavior"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from common import *
from train import *
from model.mlp import MlpConfig
from task.same_different import * 


# <codecell>
### RICH REGIME MARKOV MODEL SIMULATION

def is_match(w1, w2, z1, z2):
    if w1[z1] + w2[z2] > 0:
        return True
    elif w1[z1] + w2[z2] == 0:
        if np.random.random() >= 0.5:
            return True
    
    return False

def run_sim(a, n_rep=100):
    meas = []

    for _ in tqdm(range(n_rep)):
        n_iters = 100
        n_vocab = 16
        batch_size = 512

        w1 = np.zeros(n_vocab)
        w2 = np.zeros(n_vocab)

        mu_count_0 = 0
        mu_total_0 = 0

        mu_count_pos = 0
        mu_total_pos = 0

        mu_count_neg = 0
        mu_total_neg = 0

        for i in range(n_iters):
            bu1 = np.zeros(n_vocab)
            bu2 = np.zeros(n_vocab)

            for _ in range(batch_size):
                z1, z2 = np.random.choice(n_vocab, size=2, replace=False)
                if np.random.random() >= 0.5:
                    z1 = z2

                
                s1 = w1[z1] + w1[z2]
                s2 = w2[z1] + w2[z2]

                if s1 > 0 and s2 > 0:
                    mu_total_pos += 1

                if s1 < 0 and s2 < 0:
                    mu_total_neg += 1

                if s1 * s2 < 0:
                    mu_total_0 += 1

                if is_match(w1, w2, z1, z2):
                    if z1 == z2:
                        bu1[z1] += a
                        bu2[z2] += a
                    else:
                        bu1[z1] -= a
                        bu2[z2] -= a

                    if s1 > 0 and s2 > 0:
                        mu_count_pos += 1

                    if s1 < 0 and s2 < 0:
                        mu_count_neg += 1

                    if s1 * s2 < 0:
                        mu_count_0 += 1

            w1 += bu1
            w2 += bu2

            
        if mu_total_pos == 0 or mu_total_neg == 0 or mu_total_0 == 0:
            continue


        acc = 0

        for _ in range(batch_size):
            z1, z2 = np.random.choice(n_vocab, size=2, replace=False)
            if np.random.random() >= 0.5:
                z1 = z2
            
            acc += jax.nn.relu(w1[z1] + w2[z2])

        meas.append({
            'pos': mu_count_pos / mu_total_pos,
            'neg': mu_count_neg / mu_total_neg,
            'zer': mu_count_0 / mu_total_0,
            'magn': acc / batch_size,
            'abs': np.mean(np.abs(w1))
        })
    
    return meas


# %%
meas = run_sim(a=1)
df = pd.DataFrame(meas)
df

# <codecell>
mean = np.mean(df['zer'])
std = np.std(df['zer'])
inc = 2 * std
print('mean', mean)
print('+/-', inc)

# %%
meas = run_sim(a=-1)
df = pd.DataFrame(meas)
df

# <codecell>
mean = np.mean(df['zer'])
std = np.std(df['zer'])
inc = 2 * std
print('mean', mean)
print('+/-', inc)

# <codecell>
n_points = 2
n_dims = 512
n_hidden = 1024

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0**2 * 10

n_patches = 2

n_rep = 10

train_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128)
test_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

config = MlpConfig(mup_scale=True,
                   as_rf_model=False,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu')

res = []
for _ in tqdm(range(n_rep)):
    state, hist = train(config,
                        data_iter=iter(train_task), 
                        test_iter=iter(test_task), 
                        loss='bce',
                        test_every=1000,
                        train_iters=5_000, 
                        optim=optax.sgd,
                        lr=lr,
                        gamma=gamma,
                        seed=None)

    a = state.params['Dense_1']['kernel'].flatten()
    avg = np.mean(a[a < 0]) / np.mean(a[a > 0])
    res.append(avg)

# <codecell>
mean = np.mean(res)
inc = np.std(res) * 2
print('mean', mean)
print('+/-', inc)

