import os
import pandas as pd 
import numpy as np
import argparse
from tqdm import tqdm
from hp_optim import datasets
from hetero_var import test_hetero_var_all_folds
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--project', type=str, default='models')
    args = parser.parse_args()
    datasets = np.array(list(datasets.keys()))
    if args.train:
        for dataset in datasets:
            print(f'######## Dataset {dataset}')
            os.system(f'python reg_benchmarks.py --dataset {dataset} --project {args.project}')
    else:
        res = np.empty((10,16))
        for i, dataset in tqdm(enumerate(datasets)):
            if dataset == 'song-year':
                n_ens = 1
            elif dataset == 'protein':
                n_ens = 5
            else:
                n_ens = 20
            mpiw_va = np.load(f'{args.project}/mpiw_va_{dataset}.npy')
            picp_va = np.load(f'{args.project}/picp_va_{dataset}.npy')
            picp_te = np.load(f'{args.project}/picp_te_{dataset}.npy')
            mpiw_va_1, mpiw_va_2 = mpiw_va[:,0].mean(), mpiw_va[:,1].mean()
            picp_va_1, picp_va_2 = picp_va[:,0].mean(), picp_va[:,1].mean()
            picp_te_1, picp_te_2 = picp_te[:,0].mean(), picp_te[:,1].mean()
            std_mpiw_va_1, std_mpiw_va_2 = mpiw_va[:,0].std(), mpiw_va[:,1].std()
            std_picp_va_1, std_picp_va_2 = picp_va[:,0].std(), picp_va[:,1].std()
            std_picp_te_1, std_picp_te_2 = picp_te[:,0].std(), picp_te[:,1].std()
            prop_sig, pse_avg, pse_std = test_hetero_var_all_folds(dataset, n_ens)
            res[i,:] = [i, prop_sig, pse_avg, pse_std,
                        mpiw_va_1, std_mpiw_va_1,
                        picp_va_1, std_picp_va_1,
                        mpiw_va_2, std_mpiw_va_2,
                        picp_va_2, std_picp_va_2,
                        picp_te_1, std_picp_te_1,
                        picp_te_2, std_picp_te_2]
            print(f'{dataset}: avg pse: {pse_avg:.2f}+/-{pse_std:.2f}')
        cols = ['dataset', 'prop_sig', 'avg_pse', 'std_pse',
                'mpiw_va_e', 'std_mpiw_va_e', 'picp_va_e', 'std_picp_va_e',
                'mpiw_va_r', 'std_mpiw_va_r', 'picp_va_r', 'std_picp_va_r',
                'picp_te_e', 'std_picp_te_e', 'picp_te_r', 'std_picp_te_r']
        df = pd.DataFrame(res, columns=cols)
        df.dataset = df.dataset.apply(lambda i: datasets[int(i)]) 
        print(df)            