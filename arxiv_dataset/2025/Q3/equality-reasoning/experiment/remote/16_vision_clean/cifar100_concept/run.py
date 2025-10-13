"""
Match task accuracies
"""

# <codecell>
import time

import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../../')

from common import *
from train import *

from model.mlp import MlpConfig 
from task.same_different import SameDifferentCifar100

run_id = new_seed()
print('RUN ID', run_id)

run_split = 9
sleep_delay = True

train_iters = 50_000
n_hidden = 1024

n_trains = [90]
log10_gs = [-5, -1, 0]
base_lr = 0.01

preprocess = [True]

# layer_names = ['relu2_1',
#                'relu2_2',
#                'relu3_1',
#                'relu3_2',
#                'relu3_3',
#                'relu4_1',
#                'relu4_2',
#                'relu4_3']

layer_names = ['relu4_1',
               'relu4_2',
               'relu4_3']

### START TEST CONFIGS
# run_split = 1
# sleep_delay = False

# train_iters = 1000
# n_hidden = 512

# n_trains = [8]
# log10_gs = [0]
# preprocess = [True]
# layer_names = ['relu5_3']
### END TEST CONFIGS

if sleep_delay:
    print('start sleep')
    run_idx = sys.argv[1]
    run_idx = int(run_idx)

    if run_idx < run_split:
        time.sleep(10 * run_idx)
    print('end sleep')

all_cases = []
test_tasks = []

for prep, actv, n_train in itertools.product(preprocess, layer_names, n_trains):
    ps = np.random.permutation(np.arange(100))

    train_ps = ps[:n_train]
    test_ps = ps[n_train:]

    for log10_gamma0 in log10_gs:
        train_task = SameDifferentCifar100(ps=train_ps, preprocess_cnn=prep, actv_layer=actv, batch_size=1)
        xs, _ = next(train_task)
        xs = xs.reshape(1, -1)

        gamma0 = 10**log10_gamma0

        if log10_gamma0 > -5:
            gamma0 *= np.sqrt(xs.shape[-1])

        gamma = gamma0 * np.sqrt(xs.shape[-1])   # normalization on input
        lr = gamma0**2 * base_lr

        train_task.batch_size = 128

        c = Case(rf'MLP ($\gamma_0=10^{ {log10_gamma0} }$)', 
            MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden, mup_scale=True, use_bias=False),
            train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce', 'optim': optax.sgd, 'lr': lr, 'gamma': gamma},
            train_task=SameDifferentCifar100(ps=train_ps, preprocess_cnn=prep, actv_layer=actv),
            test_task=SameDifferentCifar100(ps=test_ps, preprocess_cnn=prep, actv_layer=actv),
            info={'log10_gamma0': log10_gamma0, 'n_classes': n_train, 'preprocess': prep, 'actv': actv})
        all_cases.append(c)


all_cases = split_cases(all_cases, run_split)
print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

train_tasks = [c.train_task for c in all_cases]
test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=train_tasks, key_name='acc_seen')
eval_cases(all_cases, eval_task=test_tasks, key_name='acc_unseen')

for case in all_cases:
    case.info['params'] = case.state.params
    case.state = None

    # Task vecs take up a lot of memory
    case.train_task = None
    case.test_task = None

    hist_acc = [m.accuracy.item() for m in case.hist['test']]
    case.info['acc_best'] = max(hist_acc)
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
