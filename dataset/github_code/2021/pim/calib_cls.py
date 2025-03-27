import pandas as pd 
import numpy as np
import argparse
from clean_cls_data import datasets
from pathlib import Path
import calibration as cal
from cls_benchmarks import run_pim
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

def calibrate(ytrue, ypred, binning=False, marginal=None):
    ytrue = np.squeeze(ytrue)
    init = cal.PlattBinnerCalibrator if binning else cal.PlattCalibrator 
    init = cal.PlattBinnerMarginalCalibrator if marginal else init
    calibrator = init(ytrue.shape[0], num_bins=10)
    calibrator.train_calibration(ypred, ytrue.astype(np.int32))
    return calibrator.calibrate(ypred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--results_dir', type=str, default='data/classification/results_tanh')
    args = parser.parse_args()
    datasets = np.array(list(datasets.keys()))

    for dataset in datasets:
        print(f'Calibrating predictions in {dataset}')
        ytrue, ypred = np.load(Path(args.results_dir) / f'{dataset}_sm.npy')
        ypred_calib = np.array([calibrate(ytrue[k,:], ypred[k,:]) for k in range(ytrue.shape[0])])
        np.save(Path(args.results_dir) / f'{dataset}_sm_calib_platt.npy', [ytrue, ypred_calib])
        print(f'Running PIM on calibrated predictions')
        ytrue, ypred = np.load(Path(args.results_dir) / f'{dataset}_sm_calib_platt.npy')
        acc_uacc = np.array([run_pim(ytrue[k,:], ypred[k,:])['ACC'] for k in range(ytrue.shape[0])]) 
        time = pd.read_csv(Path(args.results_dir) / f'{dataset}_pim.csv', usecols=['time']).values
        res_pim_calib = np.concatenate((acc_uacc, time), axis=1)
        pd.DataFrame(res_pim_calib, columns=['acc', 'uacc', 'time']).\
                    to_csv(Path(args.results_dir) / f'{dataset}_pim_calib_platt.csv', index=False)