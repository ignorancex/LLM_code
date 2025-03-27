import os
import pandas as pd 
import numpy as np
import argparse
from tqdm import tqdm
from clean_cls_data import datasets
from pathlib import Path
import calibration as cal
from scipy.stats import median_absolute_deviation as mad
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

def calib_error(ytrue, ypred):
    return cal.get_calibration_error(ypred, ytrue.astype('int'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--calib', action='store_true')
    parser.add_argument('--results_dir', type=str, default='data/classification/results')
    args = parser.parse_args()
    datasets = np.array(list(datasets.keys()))
    # calib_suffix = '_calib' if args.calib else ''
    calib_suffix = '_calib_platt' if args.calib else ''
    if args.train:
        for dataset in datasets:
            print(f'######## Dataset {dataset}')
            os.system(f'python cls_benchmarks.py --dataset {dataset}')
    else:
        res = np.empty((9,9))
        for i, dataset in enumerate(datasets):
            res_bs = pd.read_csv(Path(args.results_dir) / f'{dataset}_bs.csv')
            res_sm = pd.read_csv(Path(args.results_dir) / f'{dataset}_pim{calib_suffix}.csv')
            ytrue, ypred = np.load(Path(args.results_dir) / f'{dataset}_sm{calib_suffix}.npy') # in test set
            calib_err = np.array([calib_error(ytrue[k,:], ypred[k,:]) for k in range(ytrue.shape[0])])
            calib_err = [100*np.median(calib_err), 100*mad(calib_err)]
            acc_sm = res_sm['acc'].mean() 
            uacc_sm = [2*np.nanmedian(res_sm['uacc'].values), 
                       2*mad(res_sm['uacc'], nan_policy='omit')]
            uacc_bs = [np.nanmedian(res_bs['uacc']), 
                       mad(res_bs['uacc'], nan_policy='omit')]
            uacc_binom = 2*1.96*np.sqrt(acc_sm*(1-acc_sm)/ ytrue.shape[1])
            res[i,:] = [i] + [acc_sm] + calib_err + uacc_bs + uacc_sm + [uacc_binom]
        cols = ['dataset', 'acc_sm', 'calib_err_med', 'calib_err_mad', 'uacc_bs_med', 'uacc_bs_mad',
                'uacc_pim_med', 'uacc_pim_mad', 'uacc_binom']
        df = pd.DataFrame(res, columns=cols)
        df.dataset = df.dataset.apply(lambda i: datasets[int(i)]) 
        print(df)  
            