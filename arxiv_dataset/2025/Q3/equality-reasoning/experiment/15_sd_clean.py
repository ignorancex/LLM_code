"""SD task plotting"""


# <codecell>
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm


import sys
sys.path.append('../')
from common import *
from train import *

def pred_rich_acc(n_points, a_raw=1.5):
    prefac = 2 / (13 * (np.pi - 2))
    pt = np.sqrt(prefac * (n_points**2 - n_points))
    neg_acc = norm.cdf(pt)
    return (neg_acc + 1) / 2

set_theme()

# <codecell>
df = collate_dfs('remote/15_sd_clean/data_div', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
adf = plot_df[
    (plot_df['n_dims'] == 256)
    & (plot_df['n_width'] == 1024)
    ]

mdf = adf[adf['name'].str.contains('gamma')]
mdf2 = adf[~adf['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_symbols', y='acc_best', hue='gamma0', marker='o', alpha=0.7)
g.figure.set_size_inches((3.5, 2.7))

xs = np.unique(mdf['n_symbols'])
acc_est = pred_rich_acc(xs, a_raw=1.5)
acc_est[0] = 0.75

tdf = pd.DataFrame({'n_symbols': xs, 'acc_best': acc_est})
tdf['name'] = 'Theory'
sns.lineplot(tdf, x='n_symbols', y='acc_best', hue='name', ax=g, palette=['red'], linestyle='dashed')

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, color='gray', linestyle='dashed')
g.text(2**8.5, 0.51, 'chance', color='gray', fontsize=10)

g.legend_.set_title('')

handles, labels = plt.gca().get_legend_handles_labels()
order = [6, 5, 4, 3, 2, 1, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    elif text != 'Theory':
        val = np.round(10**float(text), decimals=2)
        if val != 0:
            t.set_text(f'$\gamma = {val:.2f}$')
        else:
            t.set_text(r'$\gamma \approx 0.00$')

g.set_xlabel('# symbols ($L$)')
g.set_ylabel('Test accuracy')
g.set_xscale('log', base=2)


g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/sd_by_l.svg', bbox_inches='tight')
plt.show()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_symbols'] == 16)
    & (mdf['n_width'] == 1024)
    & mdf['name'].str.contains('gamma')
    ]

g = sns.lineplot(mdf, x='n_dims', y='acc_best', hue='gamma0', marker='o')
g.figure.set_size_inches((3.5, 2.7))

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, color='gray', linestyle='dashed')
g.text(2**4, 0.51, 'chance', color='gray', fontsize=10)

g.legend_.set_title('')

handles, labels = plt.gca().get_legend_handles_labels()
order = [5, 4, 3, 2, 1, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

for t in g.legend_.get_texts():
    text = t.get_text()
    val = np.round(10**float(text), decimals=2)
    if val != 0:
        t.set_text(f'$\gamma = {val:.2f}$')
    else:
        t.set_text(r'$\gamma \approx 0.00$')

g.set_xlabel('Input dimension ($d$)')
g.set_ylabel('Test accuracy')
g.set_xscale('log', base=2)

g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/sd_by_d.svg', bbox_inches='tight')
plt.show()


# <codecell>
### LAZY VAR SYMBOLS
df = collate_dfs('remote/15_sd_clean/lazy_var_symbols', concat=True)
# df = collate_dfs('remote/15_sd_clean/lazy_test', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_width'] == 4096)
    & (mdf['gamma0'] == -5)
    ]

mdf = mdf[['n_symbols', 'n_dims', 'acc_best']]
mdf = mdf.groupby(['n_symbols', 'n_dims'], as_index=False).mean()
mdf = mdf.pivot(index='n_symbols', columns='n_dims', values='acc_best')

mdf = mdf.iloc[::-1]
g = sns.heatmap(mdf, square=False)
g.figure.set_size_inches(3.5, 2.7)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 20 - 2 * xs + 0.5, color='black', linestyle='dashed')

g.set_xlabel('Input dimension ($d$)')
g.set_ylabel('# symbols ($L$)')

g.figure.savefig('fig/lazy_ndim_v_nsym.svg')
plt.show()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_width'] == 4096)
    & (mdf['gamma0'] == 0)
    ]

mdf = mdf[['n_symbols', 'n_dims', 'acc_best']]
mdf = mdf.groupby(['n_symbols', 'n_dims'], as_index=False).mean()
mdf = mdf.pivot(index='n_symbols', columns='n_dims', values='acc_best')

mdf = mdf.iloc[::-1]
g = sns.heatmap(mdf, vmin=0.5, vmax=1)
g.figure.set_size_inches(3.5, 2.7)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 16.5 - 0 * xs, color='black', linestyle='dashed')

g.set_xlabel('Input dimension ($d$)')
g.set_ylabel('# symbols ($L$)')

g.figure.savefig('fig/rich_ndim_v_nsym.svg')
plt.show()

# <codecell>
### LAZY VAR WIDTH
df = collate_dfs('remote/15_sd_clean/lazy_var_width', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df


# <codecell>
### NOISE
df = collate_dfs('remote/15_sd_clean/noise', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['config']['n_hidden'],
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['sig2'],
        row['train_task'].noise,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_width', 'n_symbols', 'n_dims', 'sig2', 'noise', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df


# <codecell>
df_bayes = collate_dfs('remote/15_sd_clean/bayes', concat=True)
df_bayes

# <codecell>
mdf = plot_df.copy()

mdf = mdf[(mdf['n_width'] == 1024)]
# sigs = [0, 0.1, 1, 2, 4]
sigs = [0, 1, 2, 4]
# all_n_dims = [64, 128, 256]
all_n_dims = [64]

for sig, n_dims in tqdm(list(itertools.product(sigs, all_n_dims))):
    cdf = mdf[(mdf['noise'] == sig) & (mdf['n_dims'] == n_dims)]
    bdf = df_bayes[(df_bayes['sig2'] == sig) & (df_bayes['n_dims'] == n_dims)]

    g = sns.lineplot(cdf, x='n_symbols', y='acc_best', hue='gamma0', marker='o')
    sns.lineplot(bdf, x='n_symbols', y='acc_unseen', hue='name', ax=g, linestyle='dashed', errorbar=('ci', False), palette=['red', 'magenta'])

    g.set_ylim((0.45, 1.02))
    g.axhline(y=1, color='white', linestyle='dashed', alpha=0)

    g.set_xscale('log', base=2)
    g.set_xlabel('# symbols ($L$)')
    g.set_ylabel('Test accuracy')
    g.set_title(f'$\sigma^2 = {sig}$')

    g.legend_.set_title('')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [5, 4, 3, 2, 1, 0, 6, 7]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    for t in g.legend_.get_texts():
        text = t.get_text()
        if 'Gen' in text:
            t.set_text('Bayes gen')
        elif 'Mem' in text:
            t.set_text('Bayes mem')
        elif text != 'Theory':
            if float(text[-1]) != 5:
                t.set_text('$\gamma$ = $10^{%s}$' % text)
            else:
                t.set_text(r'$\gamma \approx 0$')

    g.figure.set_size_inches(3, 2.6)
    g.figure.tight_layout()
    sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(f'fig/d_{n_dims}_sig2_{sig}.svg', bbox_inches='tight')
    plt.show()


# <codecell>
df = collate_dfs('remote/15_sd_clean/rich', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
adf = plot_df[
    (plot_df['n_dims'] == 256)
    & (plot_df['n_width'] == 1024)
    ]

mdf = adf[adf['name'].str.contains('gamma')]
mdf2 = adf[~adf['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_symbols', y='acc_best', hue='gamma0', marker='o', alpha=0.7)
g.figure.set_size_inches((3, 2.4))

xs = np.unique(mdf['n_symbols'])
acc_est = pred_rich_acc(xs, a_raw=1.5)
acc_est[0] = 0.75

tdf = pd.DataFrame({'n_symbols': xs, 'acc_best': acc_est})
tdf['name'] = 'Theory'
sns.lineplot(tdf, x='n_symbols', y='acc_best', hue='name', ax=g, palette=['red'], linestyle='dashed')

g.legend_.set_title('')

handles, labels = plt.gca().get_legend_handles_labels()
order = [6, 5, 4, 3, 2, 1, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    elif text != 'Theory':
        t.set_text('$\gamma$ = $10^{%s}$' % np.round(float(text), decimals=1))

g.set_xlabel('# symbols ($L$)')
g.set_ylabel('Test accuracy')


g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/rich.svg', bbox_inches='tight')
plt.show()

