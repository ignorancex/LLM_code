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
from task.same_different import SameDifferentPsvrt, gen_patches

run_id = new_seed()
print('RUN ID', run_id)

run_split = 9

train_iters = 100_000
n_hidden = 512

n_patches = [2, 3, 4, 5, 6, 7, 8, 9, 10]
patch_size = 5

n_trains = [1024]
log10_gs = np.linspace(-5, 0, num=6)
base_lr = 0.5

### START TEST CONFIGS
# run_split = 1

# train_iters = 100
# n_hidden = 512

# n_patches = [5]
# patch_size = 5

# n_trains = [16]
# gs = 1,
### END TEST CONFIGS

all_cases = []
test_tasks = []

for n_train, n_patch in itertools.product(n_trains, n_patches):
    train_set = gen_patches(patch_size, n_examples=n_train)

    for log10_gamma0 in log10_gs:
        gamma0 = 10**log10_gamma0

        # if log10_gamma0 > -5:
        #     gamma0 *= n_patch * patch_size

        gamma = gamma0
        lr = gamma0**2 * base_lr

        c = Case(rf'MLP ($\gamma_0=10^{ {log10_gamma0} }$)', 
            MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden, mup_scale=True, use_bias=False),
            train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce', 'optim': optax.sgd, 'lr': lr, 'gamma': gamma},
            train_task=SameDifferentPsvrt(patch_size=patch_size, n_patches=n_patch, inc_set=train_set),
            test_task=SameDifferentPsvrt(patch_size=patch_size, n_patches=n_patch, batch_size=1024),
            info={'log10_gamma0': log10_gamma0, 'n_symbols': n_train})
        all_cases.append(c)

all_cases = split_cases(all_cases, run_split)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

train_tasks = [c.train_task for c in all_cases]
test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=train_tasks, key_name='acc_seen')
eval_cases(all_cases, eval_task=test_tasks, key_name='acc_unseen')

for case in all_cases:
    case.state = None

    hist_acc = [m.accuracy.item() for m in case.hist['test']]
    case.info['acc_best'] = max(hist_acc)
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
