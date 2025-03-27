#!/usr/bin/env python
import argparse
import itertools
import numpy as np
import pandas as pd
import pingouin as pg
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import sys
from mlconfound.stats import full_confound_test, partial_confound_test
from mlconfound.simulate import simulate_y_c_yhat, identity, polynomial, sigmoid

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.path[0] + '/..')

parser = argparse.ArgumentParser(description="Validate 'mlconfound' on simulated data.")

parser.add_argument("--repetitions", help="Number of repetitions. Default: 100.",
                    action="store", type=int, default=100)

parser.add_argument("--mode", help="Confound testing mode: 'partial' or 'full'. Default: 'partial'. ",
                    choices=['partial', 'full'],
                    type=str, default='partial')

parser.add_argument("--delta", dest="delta", action="store", default=1.0,
                    help="Delta of the sinh_arcsinh transformation to set kurtosis. "
                         "Default: 1: no transformation",
                    type=float)
parser.add_argument("--epsilon", dest="epsilon", action="store", default=0.0,
                    help="Epsilon of the sinh_arcsinh transformation to set skewness. "
                         "Default: 0: no transformation",
                    type=float)

parser.add_argument("--nonlin-trf", dest="nonlin_trf",
                    help="Transformation to simulate non-linear dependnecies. "
                         "Default: identity",
                    choices=['identity', 'squared', 'sigmoid'],
                    type=str, default='identity')

parser.add_argument("--cat-y", help="y is categorical (binary). (default: continuous)",
                    action="store_true")
parser.add_argument("--cat-yhat", help="yhat is categorical (binary). (default: continuous)",
                    action="store_true")
parser.add_argument("--cat-c", help="c is categorical (binary). (default: continuous)",
                    action="store_true")

parser.add_argument("--random-seed", dest="random_seed", action="store", default=42,
                    help="Random seed. Default: 42", type=int)
parser.add_argument("--n-jobs", dest="n_jobs", action="store", default=-1,
                    help="Number of cores to use. Default: -1 (all available cores)", type=int)

parser.add_argument("--out_prefix", dest="out_prefix", action="store", default='validation',
                    help="Output file name prefix.", type=str)

parser.add_argument("--out", dest="out_file", action="store", default=None,
                    help="Output file name. Overrides --out_prefix", type=str)


if __name__ == '__main__':
    args = parser.parse_args()

    # see if file is already there
    if args.out_file is None:
        args2 = vars(args).copy()
        args2['out_prefix'] = None
        params = '_'.join(str(args2)[10:].split(', '))[:-1]
        filename = args.out_prefix + '_' + params + '.csv'
    else:
        filename = args.out_file
    if os.path.exists(filename):
        print("Already calculated. Skipping this simulation run.")
        sys.exit()

    if args.nonlin_trf == "identity":
        nonlin_trf = identity
    elif args.nonlin_trf == 'squared':
        nonlin_trf = polynomial
    elif args.nonlin_trf == 'sigmoid':
        nonlin_trf = sigmoid
    else:
        raise AttributeError("No such transformation:", args.nonlin_trf)

    if args.mode.startswith('partial'):
        confound_test = partial_confound_test
    elif args.mode.startswith('full'):  # option 1: full
        confound_test = full_confound_test
    else:
        raise AttributeError('No such mode', args.mode)

    ################# default simulation parameters #########################
    repetitions = args.repetitions
    num_perms = 1000

    all_n = [50, 100, 500, 1000]

    #     w  0    0.33  0.66  1    2
    #     r2 0%   10%   30%  50%  80%
    #     r  0    .3    .55   .7  .9
    all_yc = [0, 0.33, 0.66, 1]
    all_yyhat = [0, 0.33, 0.66, 1]
    #     w  0    0.2  0.4  0.6
    #     r2 0%   4%   12%  25%
    #     r  0    .2    .35   .75
    all_cyhat = [0, 0.2, 0.4, 0.6]  # H0 or H1
    #########################################################################

    print('Number of simulated:', np.prod([len(i) for i in [
        all_yc,
        all_yyhat,
        all_cyhat,
        all_n]]) * repetitions)

    all_param_configurations = itertools.product(
        all_yc,
        all_yyhat,
        all_cyhat,
        all_n)

    rng = np.random.default_rng(args.random_seed)

    results = pd.DataFrame()

    for w_yc, w_yyhat, w_cyhat, n in tqdm(list(all_param_configurations)):

        def workhorse(_random_state):
            # simulate

            y, c, yhat = simulate_y_c_yhat(w_yc=w_yc,
                                           w_yyhat=w_yyhat, w_cyhat=w_cyhat,
                                           n=n,
                                           bin_y=args.cat_y, 
                                           bin_c=args.cat_c, 
                                           bin_yhat=args.cat_yhat, 
                                           random_state=_random_state,
                                           delta=args.delta,
                                           epsilon=args.epsilon,
                                           nonlin_trf_fun=nonlin_trf)

            # test
            res_gam = confound_test(y, yhat, c,
                                    cat_y=args.cat_y,
                                    cat_yhat=args.cat_yhat,
                                    cat_c=args.cat_c,
                                    num_perms=num_perms,
                                    cond_dist_method='gam',
                                    random_state=_random_state,
                                    n_jobs=1,
                                    progress=False)

            # for efficiency
            res_lin = res_gam #confound_test(y, yhat, c,
                             #       cat_y=args.cat_y,
                             #       cat_yhat=args.cat_yhat,
                             #       cat_c=args.cat_c,
                             #       num_perms=num_perms,
                             #       cond_dist_method='linear',
                             #       random_state=_random_state,
                             #       n_jobs=1,
                             #       progress=False)

            # do partial correlation, too
            if args.mode.startswith('partial'):
                df = pd.DataFrame({
                    'x': c,
                    'y': yhat,
                    'c': y
                })
            elif args.mode.startswith('full'):
                df = pd.DataFrame({
                    'x': y,
                    'y': yhat,
                    'c': c

                })
            else:
                raise AttributeError("Invalid mode", args.mode)

            # partial correlation
            res_parcor_pearson = pg.partial_corr(data=df, x='x', y='y', covar='c',
                                    method='pearson')
            res_parcor_spearman = pg.partial_corr(data=df, x='x', y='y', covar='c',
                                    method='spearman')

            return res_lin.p, res_gam.p, res_parcor_pearson['p-val'].values[0], res_parcor_spearman['p-val'].values[0],\
               res_gam.r2_y_c, res_gam.r2_yhat_c, res_gam.r2_y_yhat, _random_state


        random_sates = rng.integers(np.iinfo(np.int32).max, size=repetitions)
        p_cpt_lin, p_cpt_gam, p_pc_pearson, p_pc_spearman, r2_y_c, r2_yhat_c, r2_y_yhat, random_seed = zip(*
                                                           Parallel(n_jobs=-1)(
                                                               delayed(workhorse)(rs) for rs in random_sates)
                                                           )

        # create DataFrame and save it
        results = results.append(pd.DataFrame({
            "p_cpt_lin": p_cpt_lin,
            "p_cpt_gam": p_cpt_gam,
            "p_pc_pearson": p_pc_pearson,
            "p_pc_spearman": p_pc_spearman,
            "r2_y_c": r2_y_c,
            "r2_yhat_c": r2_yhat_c,
            "r2_y_yhat": r2_y_yhat,
            "n": n,
            "w_yc": w_yc,
            "w_yyhat": w_yyhat,
            "w_cyhat": w_cyhat,
            "num_perms": num_perms,
            "random_seed": list(random_seed)
        }), ignore_index=True)

        # update file after every iteration...
        if args.out_file is None:
            args2 = vars(args).copy()
            args2['out_prefix'] = None
            params = '_'.join(str(args2)[10:].split(', '))[:-1]
            filename = args.out_prefix + '_' + params + '.csv'
            results.to_csv(filename)
        else:
            results.to_csv(args.out_file)
