import runner
from runner.utils import _get_timestr
import os
import shutil
import sys
import argparse

try:
    from .utils import *
except:
    from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--n_max_gpus', '-ng', type=int, default=2)
parser.add_argument('--n_multiplex', '-nm', type=int, default=4)
parser.add_argument('--seed_s', '-ss', type=int, default=0)
parser.add_argument('--seed_e', '-se', type=int, default=20)
parser.add_argument('-rmk', type=str, default='')
parser.add_argument('-train', action='store_true', default=False)

EXP = __import__(sys.argv[1])
args = EXP.add_args(parser).parse_args(sys.argv[2:])

exp_dir_base = os.path.expanduser(f'~/run/liv')
assert not os.system('df -h ' + exp_dir_base)
log_dir_base = os.path.join(exp_dir_base, f'{_get_timestr()}_{sys.argv[1]}_{args.rmk}')

env_pref = f'CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_MEM_FRACTION={0.95/args.n_multiplex:.3f} OMP_NUM_THREADS=4 '
root_cmd = env_pref + EXP.base_cmd(args)
hps = EXP.list_hps(args)
tasks = [proc(a | {'__info': Info(root_cmd=root_cmd, log_dir_base=log_dir_base)}) for a in hps]
print('\n'.join([t.cmd for t in tasks[-100:]]))
print(len(tasks))
if not args.train:
    sys.exit(0)

os.makedirs(log_dir_base, exist_ok=True)
shutil.copyfile(__file__, os.path.join(log_dir_base, 'main.py'))
shutil.copyfile(sys.argv[1] + '.py', os.path.join(log_dir_base, 'exp.py'))
with open(os.path.join(log_dir_base, 'exp.py'), 'a') as fout:
    print('#', ' '.join(sys.argv), file=fout)
with open(os.path.join(log_dir_base, 'NAME'), 'w') as fout:
    print(args.rmk, file=fout)
r = runner.Runner(
    n_max_gpus=args.n_max_gpus, n_multiplex=args.n_multiplex, n_max_retry=-1)
r.run_tasks(tasks)
with open(os.path.join(log_dir_base, 'COMPLETE'), 'w') as fout:
    fout.write('.')
