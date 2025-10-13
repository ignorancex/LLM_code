"""Common utilities"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')

def set_theme():
    sns.set_theme(style='ticks', font_scale=1.25, rc={
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.figsize': (4, 3)
    })


def new_seed():
    return np.random.randint(0, np.iinfo(np.int32).max)


def t(xs):
    return np.swapaxes(xs, -2, -1)


def split_cases(all_cases, run_split):
    run_idx = sys.argv[1]
    try:
        run_idx = int(run_idx) % run_split
    except ValueError:
        print(f'warn: unable to parse index {run_idx}, setting run_idx=0')
        run_idx = 0

    print('RUN IDX', run_idx)
    all_cases = np.array_split(all_cases, run_split)[run_idx]
    return list(all_cases)


def summon_dir(path: str, clear_if_exists=False):
    new_dir = Path(path)
    if new_dir.exists() and clear_if_exists:
        shutil.rmtree(new_dir)

    new_dir.mkdir(parents=True)
    return new_dir


def collate_dfs(df_dir, show_progress=False, concat=True):
    pkl_path = Path(df_dir)
    dfs = []

    it = pkl_path.iterdir()
    if show_progress:
        it = tqdm(list(it))
    
    for f in it:
        if f.suffix == '.pkl':
            try:
                df = pd.read_pickle(f)
                dfs.append(df)
            except:
                print(f'warn: fail to read {f}')

    if concat:
        dfs = pd.concat(dfs)

    return dfs


def log_gen_like_same(z1, z2, d, sig2):
    t1 = -d * (np.log(2 * np.pi) - np.log(d))
    t2 = (-d/2) * np.log(sig2)
    t3 = (-d / 2) * np.log(2 + sig2)

    a1 = (1 + sig2) / (2 + sig2)
    a2 = (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)
    a3 = (2 / (2 + sig2))
    a4 = np.dot(z1, z2)

    t4 = - d/(2 * sig2)
    t5 = a1 * a2 - a3 * a4

    t_exp = t4 * t5
    log_sol = t1 + t2 + t3 + t_exp
    return log_sol


def log_gen_like_diff(z1, z2, d, sig2):
    t1 = -d * (np.log(2 * np.pi) - np.log(d))
    t2 = -d * np.log(1 + sig2)

    a1 = 1 / (1 + sig2)
    a2 = (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)

    t4 = - d/2
    t5 = a1 * a2

    t_exp = t4 * t5
    log_sol = t1 + t2 + t_exp
    return log_sol


def log_mem_like_same(z1, z2, d, sig2, ss):
    L = len(ss)
    t1 = d * (np.log(d) -  np.log(2 * np.pi * sig2))

    a1 = (-d / (2 * sig2))
    a2 = (1/2) * (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)
    a3 = np.dot(z1, z2)

    p1 = t1 + a1 * (a2 - a3)

    d1 = -d / sig2
    d2 = ss - (z1 + z2) / 2
    d_exp = d1 * np.linalg.norm(d2, axis=1)**2

    log_sol = p1 + logsumexp(d_exp) - np.log(L)
    return log_sol


def log_mem_like_diff(z1, z2, d, sig2, ss):
    L = len(ss)
    t1 = d * (np.log(d) -  np.log(2 * np.pi * sig2))

    a1 = -d / (2 * sig2)
    e1s = np.reshape(a1 * np.linalg.norm(ss - z1, axis=1)**2, (-1, 1))
    e2s = np.reshape(a1 * np.linalg.norm(ss - z2, axis=1)**2, (1, -1))

    prods = e1s + e2s
    np.fill_diagonal(prods, -np.inf)

    t2 = -np.log(L * (L - 1))
    log_sol = t1 + t2 + logsumexp(prods)
    return log_sol