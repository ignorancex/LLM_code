from statsmodels.stats.diagnostic import het_white
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
import statsmodels.api as sm
from hp_optim import load, reset_seeds
from sklearn.model_selection import train_test_split
from scipy.fft import fft
import argparse

def test_hetero_var(x, y):
    x = sm.add_constant(x)
    result = sm.OLS(y, x).fit()
    try:
        f_pvalue = het_white(result.resid, x)[-1]
        if np.isnan(f_pvalue): raise(ValueError)
    except:
        # Largets 3 datasets or Boston throwing NAN f-stat/f_pvalue 
        model = sm.OLS(result.resid**2, np.concatenate((x, x**2), 1))
        f_pvalue = model.fit().f_pvalue
    return f_pvalue, result.resid**2

def power_spectral_entropy(resid2):
    spec = fft(resid2)
    pow_spec = np.absolute(spec)**2
    pow_spec = pow_spec / np.sum(pow_spec)
    # Normalized power_spectral_entropy
    return -pow_spec.dot(np.log2(pow_spec)) / np.log2(len(spec)) 

def test_hetero_var_all_folds(dataset, n_ens):
    x, y = load(dataset)
    f_pvalues = np.empty((n_ens,))
    pse = np.empty((n_ens,))
    for seed in range(n_ens):
        reset_seeds(seed)
        # Randomly choose train and test set
        x_tr, _, y_tr, _ = train_test_split(x, y, 
                                test_size=0.1, random_state=seed+1)
        # Randomly choose validation set from 20% of previous training set
        x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, 
                                test_size=0.2, random_state=seed+1)
        # Tests on validation set
        f_pvalues[seed], resid2 = test_hetero_var(x_va, y_va)
        pse[seed] = power_spectral_entropy(resid2)
    # Percentage of significant tests and average pse
    return np.mean(f_pvalues <= 0.05), np.mean(pse), np.std(pse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yacht')
    parser.add_argument('--n_ens', type=int, default=20)
    args = parser.parse_args()
    prop_sig, pse_avg, pse_std = test_hetero_var_all_folds(args.dataset, args.n_ens)
    print(f'% of significant tests: {prop_sig}, Avg pse: {pse_avg:.1f}+/-{pse_std:.1f}')

