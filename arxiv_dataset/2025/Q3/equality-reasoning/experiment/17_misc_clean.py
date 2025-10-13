"""Miscellaneous illustrative figures"""

# <codecell>
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from common import *
from train import *
from model.mlp import MlpConfig
from task.same_different import * 

set_theme()

# <codecell>
n_points = 16
n_dims = 128
n_hidden = 1024

gamma0 = 1e-5
gamma = gamma0
lr = gamma0**2 * 0.1

n_patches = 2

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

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=15_000, 
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)

# %%
jax.tree.map(np.shape, state.params)

W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
w1, w2 = W_sort[:n_dims], W_sort[n_dims:]
dots = w1.T @ w2 / (np.linalg.norm(w1, axis=0) * np.linalg.norm(w2, axis=0))
cos_dists = np.diag(dots)

# <codecell>
plt.gcf().set_size_inches(3.5, 2.5)

plt.scatter(a[sort_idxs], cos_dists, alpha=0.5)
plt.xlabel('$a_i$')
plt.ylabel(r'$(\mathbf{v}_i^1 \cdot \mathbf{v}_i^2)\, / \, \ell_i$')
plt.ylim((-1.1, 1.1))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig/l2_weight_struct.svg')
plt.show()

# <codecell>
### PCA and Same Different recovers parallel / anti-parallel features
n_dims = 1024
n_points = 64
batch_size = 3000

train_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=batch_size)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

train_task.symbols.shape

xs, ys = next(train_task)
xs = xs @ train_task.symbols.T
xs = xs.reshape(xs.shape[0], -1)

pos_idx = ys.astype(bool)

xs_pos = xs[pos_idx]
xs_neg = xs[~pos_idx]

plt.gcf().set_size_inches(2, 2)
diff = (xs_pos.T @ xs_pos - xs_neg.T @ xs_neg) / np.sqrt(batch_size)
im = plt.imshow(diff, vmin=-1, vmax=1, cmap='plasma')
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.gca().set_axis_off()
plt.title('Empirical')
plt.tight_layout()

plt.savefig('fig/xx_emp.svg')
plt.show()

# <codecell>
plt.gcf().set_size_inches(2, 2)
diff = np.zeros((128, 128))
diff[0:64,64:128] = np.eye(64)
diff[64:128,0:64] = np.eye(64)
im = plt.imshow(diff, vmin=-1, vmax=1, cmap='plasma')
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.gca().set_axis_off()
plt.title('Ideal')
plt.tight_layout()

plt.savefig('fig/xx_ideal.svg')
plt.show()

# <codecell>
### PSVRT EXAMPLES
task = SameDifferentPsvrt(patch_size=5, n_patches=3, batch_size=10, seed=3)
xs, ys = next(task)

xs = np.clip(xs + 0.15, a_max=1, a_min=0)
plt.imshow(xs[3], cmap='binary')
plt.axis('off')
plt.savefig('fig/psvrt_same.svg')
plt.show()

# <codecell>
plt.imshow(xs[5], cmap='binary')
plt.axis('off')
plt.savefig('fig/psvrt_diff.svg')
plt.show()

# <codecell>
### PENTOMINO EXAMPLES
task = SameDifferentPentomino(width=2, batch_size=32)
xs, ys = next(task)

xs = np.clip(xs + 0.15, a_max=1, a_min=0)
plt.imshow(xs[0], cmap='binary', vmin=0)
plt.axis('off')
plt.savefig('fig/pentomino_same.svg')
plt.show()

# <codecell>
plt.imshow(xs[31], cmap='binary', vmin=0)
plt.axis('off')
plt.savefig('fig/pentomino_diff.svg')
plt.show()

# <codecell>
### CIFAR-100 examples
task = SameDifferentCifar100(np.arange(100), batch_size=50, preprocess_cnn=False, normalize=False)
xs, ys = next(task)

xs = xs.reshape(xs.shape[0], 2, 32, 32, 3, order='F')
xs = np.swapaxes(xs, 2, 3)
xs.shape

xs_sel = xs[0]
print(ys[0])

xs_sel = np.concat(xs_sel, axis=1)

plt.imshow(xs_sel)
plt.axis('off')
plt.savefig('fig/cifar100_same.svg')
plt.show()

# %%
idx = 19
xs_sel = xs[idx]
print(ys[idx])

xs_sel = np.concat(xs_sel, axis=1)

plt.imshow(xs_sel)
plt.axis('off')
plt.savefig('fig/cifar100_diff.svg')
plt.show()

# <codecell>
n_points = 256
n_hidden = 512

gamma0 = 1e-5
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0**2 * 1

inc_set = gen_patches(patch_size=5, n_examples=n_points)
train_task = SameDifferentPsvrt(patch_size=5, n_patches=2, inc_set=inc_set)
test_task = SameDifferentPsvrt(patch_size=5, n_patches=2)

config = MlpConfig(mup_scale=True,
                   as_rf_model=False,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu')

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=15_000, 
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)


# <codecell>
W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
idx = 0

plt.imshow(W_sort[:,idx].reshape(10, 10), vmin=-0.1, vmax=0.1,  cmap='bwr')
plt.axhline(y=4.5, color='white')
plt.axvline(x=4.5, color='white')
plt.colorbar()

plt.title(f'$a = {a[sort_idxs[idx]]:.2f}$', fontsize=20)
plt.axis('off')

plt.savefig('fig/concept/psvrt_lazy_neg.svg')
plt.show()

# <codecell>
idx = -1

plt.imshow(W_sort[:,idx].reshape(10, 10), vmin=-0.1, vmax=0.1, cmap='viridis')
plt.axhline(y=4.5, color='white')
plt.axvline(x=4.5, color='white')
plt.colorbar()

plt.title(f'$a = {a[sort_idxs[idx]]:.2f}$')
plt.axis('off')

plt.savefig('fig/concept/psvrt_lazy_pos.svg')
plt.show()

# <codecell>
### PENTOMINOS
n_points = 16
n_hidden = 512

gamma0 = 1
# gamma0 = 1e-5
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0**2 * 10

train_task = SameDifferentPentomino(ps=np.arange(n_points), blur=0.5, random_blur=True)
test_task = SameDifferentPentomino(ps=[16, 17], blur=0, random_blur=False)

config = MlpConfig(mup_scale=True,
                   as_rf_model=False,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu')

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=15_000, 
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)


# %%
W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
idx = 6

plt.imshow(W_sort[:,idx].reshape(14, 14), vmin=-4, vmax=4, cmap='bwr')
plt.axhline(y=6.5, color='white')
plt.axvline(x=6.5, color='white')
plt.colorbar()

plt.title(f'$a = {a[sort_idxs[idx]]:.2f}$', fontsize=20)
plt.axis('off')

plt.savefig('fig/concept/pentomino_lazy_neg.svg')
plt.show()

# <codecell>
idx = -9

plt.imshow(W_sort[:,idx].reshape(14, 14), vmin=-1, vmax=1, cmap='bwr')
plt.axhline(y=6.5, color='white')
plt.axvline(x=6.5, color='white')
plt.colorbar()

plt.title(f'$a = {a[sort_idxs[idx]]:.2f}$', fontsize=20)
plt.axis('off')

plt.savefig('fig/concept/pentomino_lazy_pos.svg')
plt.show()

# <codecell>
### CIFAR-100 CONCEPT
df = collate_dfs('remote/16_vision_clean/cifar100_concept', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    params = jax.tree.map(np.array, row['info']['params'])

    return pd.Series([
        row['name'],
        row['info']['n_classes'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['preprocess'],
        row['info']['actv'],
        row['info']['acc_best'],
        params
    ], index=['name', 'n_classes', 'gamma0', 'acc_seen', 'acc_unseen', 'preprocess', 'actv', 'acc_best', 'params'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df[plot_df['actv'] == 'relu4_3']
mdf.loc[10]

# %%
state = mdf.loc[0]

W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
n_dims = W_sort.shape[0] // 2

w1, w2 = W_sort[:n_dims], W_sort[n_dims:]
dots = w1.T @ w2 / (np.linalg.norm(w1, axis=0) * np.linalg.norm(w2, axis=0))
cos_dists = np.diag(dots)

plt.gcf().set_size_inches(3.5, 2.5)

plt.scatter(a[sort_idxs], cos_dists, alpha=0.3)
plt.xlabel('$a_i$')
plt.ylabel(r'$(\mathbf{v}_i^1 \cdot \mathbf{v}_i^2)\, / \, \ell_i$')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(-1, 1)

plt.tight_layout()
plt.savefig('fig/concept/cifar100_rich.svg')
plt.show()

# <codecell>
### GAMMA SWEEP
df = collate_dfs('remote/15_sd_clean/gamma_sweep', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_dims,
        row['info']['norm_change'] / 0.015,  # normalized by w_0
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        10**row['info']['log10_gamma0'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['root_d']
    ], index=['name', 'n_dims', 'norm_change', 'gamma0', 'gamma', 'acc_seen', 'acc_unseen', 'root_d'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['root_d'] == False]

g = sns.lineplot(mdf, x='gamma0', y='norm_change', hue='n_dims', marker='o')
g.set_yscale('log')
g.set_ylim((1e-6, 10))

g.legend().set_title('$d$')
labs = g.get_xticklabels()
for text in labs:
    t = text.get_text()
    text.set_text('$10^{%s}$' % t)

g.set_xticklabels(labs)

g.set_xlabel(r'$\gamma$')
g.set_ylabel(r'$|\mathbf{\tilde{w}}(t) \cdot \mathbf{x}|\, / \, |\mathbf{w}(0) \cdot \mathbf{x}|$')
g.set_title('Without $1 / \sqrt{d}$')

g.figure.set_size_inches(3.5, 2.7)

plt.tight_layout()

sns.move_legend(g, 'lower left', bbox_to_anchor=(1, 0))
plt.savefig('fig/no_root_d_scale.svg', bbox_inches='tight')
plt.show()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['root_d'] == True]

g = sns.lineplot(mdf, x='gamma0', y='norm_change', hue='n_dims', marker='o')
g.set_yscale('log')
g.set_ylim((1e-6, 10))

g.legend().set_title('$d$')

g.legend().set_title('$d$')
labs = g.get_xticklabels()
for text in labs:
    t = text.get_text()
    text.set_text('$10^{%s}$' % t)

g.set_xticklabels(labs)

g.set_xlabel(r'$\gamma$')
g.set_ylabel(r'$|\mathbf{\tilde{w}}(t) \cdot \mathbf{x}|\, / \, |\mathbf{w}(0) \cdot \mathbf{x}|$')
g.set_title('With $1 / \sqrt{d}$')

g.figure.set_size_inches(3.5, 2.7)

plt.tight_layout()
sns.move_legend(g, 'lower left', bbox_to_anchor=(1, 0))
plt.savefig('fig/root_d_scale.svg', bbox_inches='tight')
plt.show()
# %%
