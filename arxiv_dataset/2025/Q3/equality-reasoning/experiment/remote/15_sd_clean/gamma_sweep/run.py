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

run_split = 12

train_iters = 25_000
n_vocab = [16]
log10_gs = np.linspace(-5, 0, num=11)
n_dims = [128, 256, 512, 1024]
n_widths = [4096]
base_lr = 0.1
include_root_d = [False, True]

n_layers = 1
sig2 = 0

### START TEST CONFIGS
run_split = 1

train_iters = 2
n_vocab = [16]
n_dims = [32]
log10_gs = [0]
n_widths = [4096]
include_root_d = [True]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for root_d, n_hidden, d, v in itertools.product(include_root_d, n_widths, n_dims, n_vocab):
    noise = sig2

    for log10_gamma0 in log10_gs:
        gamma0 = 10**log10_gamma0
        if root_d:
            gamma0 *= np.sqrt(d)

        gamma = gamma0
        lr = gamma**2 * base_lr

        all_cases.append(
            Case(rf'$\gamma=10^{ {log10_gamma0} }$',
                    MlpConfig(mup_scale=True, n_out=1, n_layers=1, n_hidden=n_hidden, use_bias=False),
                    train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce',
                                'optim': optax.sgd, 'lr': lr, 'gamma': gamma},
                    train_task=SameDifferent(n_symbols=v, n_dims=d, noise=noise),
                    test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024, noise=noise),
                    info={'log10_gamma0': log10_gamma0, 'root_d': root_d})
        )

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
    xs, _ = next(case.test_task)

    m = case.config.to_model()
    _, intm = m.apply({'params': case.hist['params'][0]}, xs, mutable='intermediates')
    actv_for = intm['intermediates']['actv'][0]

    _, intm = m.apply({'params': case.state.params}, xs, mutable='intermediates')
    actv_aft = intm['intermediates']['actv'][0]

    mean_norm = np.mean(np.abs(actv_aft - actv_for))

# mean_norm = np.mean(np.abs(actv_for))
# mean_norm

# # <codecell>

    case.info['norm_change'] = mean_norm

    case.state = None
    case.train_task.symbols = None
    case.test_task.symbols = None

    hist_acc = [m.accuracy.item() for m in case.hist['test']]
    case.info['acc_best'] = max(hist_acc)
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

# print('done!')

# %%
