"""
Match task accuracies
"""

# <codecell>
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../../')
from common import *
from train import *
from model.mlp import MlpConfig 
from task.same_different import SameDifferent

run_id = new_seed()
print('RUN ID', run_id)

run_split = 1

n_vocab = 2**np.arange(1, 14)
n_dims = [64, 128, 256]
sig2s = [0, 0.1, 1, 2, 4]
noise_scale = 1

### START TEST CONFIGS
# run_split = 1

# n_vocab = [4]
# n_dims = [2]
# sig2s = [0]
### END TEST CONFIGS

all_cases = list(itertools.product(sig2s, n_dims, n_vocab))
all_cases = split_cases(all_cases, run_split)
print('CASES', all_cases)

all_res = []

for sig2, d, L in tqdm(all_cases):
    d = int(d)
    L = int(L)

    sig2_orig = sig2

    sig2 = sig2 * noise_scale

    task = SameDifferent(n_symbols=L, n_dims=d, noise=sig2)
    test = SameDifferent(n_symbols=None, n_dims=d, batch_size=1024, noise=sig2)
    sig2 = sig2 + 1e-10

    xs, ys = next(test)

    g_preds = []
    g_logits = []
    m_preds = []
    m_logits = []

    for x in xs:
        z1, z2 = x
        g1 = log_gen_like_same(z1, z2, d, sig2)
        g2 = log_gen_like_diff(z1, z2, d, sig2)
        g_preds.append(1 if g1 > g2 else 0)
        g_logits.append(g1 - g2)

        m1 = log_mem_like_same(z1, z2, d, sig2, task.symbols)
        m2 = log_mem_like_diff(z1, z2, d, sig2, task.symbols)
        m_preds.append(1 if m1 > m2 else 0)
        m_logits.append(m1 - m2)

    all_res.extend([{
        'name': 'Bayes Gen',
        'n_symbols': L,
        'n_dims': d,
        'sig2': sig2_orig,
        'acc_unseen': np.mean(g_preds == ys),
    }, {
        'name': 'Bayes Mem',
        'n_symbols': L,
        'n_dims': d,
        'sig2': sig2_orig,
        'acc_unseen': np.mean(m_preds == ys),
    }])

df = pd.DataFrame(all_res)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
