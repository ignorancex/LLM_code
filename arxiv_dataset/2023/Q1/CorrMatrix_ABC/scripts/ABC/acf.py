import numpy as np
import pandas as pd
import pylab as plt
from statsmodels.tsa.stattools import acf


def my_acf(x, nlags):
    n = len(x)
    a = np.zeros(nlags)
    mean = np.mean(x)

    for j in range(nlags):

        # Sum via loops
        for i in range(n):
            if j+i < n:
                a[j] = a[j] + (x[i] - mean) * (x[j+i] - mean)

        # Alternative sum
        #a[j] = np.sum((x[:n-j] - mean) * (x[j:] - mean))

        a[j] = a[j] / (n - j)

    return a / np.var(x)



def autocorr_by_hand(x, lag):
    # Slice the relevant subseries based on the lag
    y1 = x[:(len(x)-lag)]
    y2 = x[lag:]
    # Subtract the subseries means
    sum_product = np.sum((y1-np.mean(y1))*(y2-np.mean(y2)))
    # Normalize with the subseries stds
    return sum_product / ((len(x) - lag) * np.std(y1) * np.std(y2))



def acf_by_hand(x, lag):
    # Slice the relevant subseries based on the lag
    y1 = x[:(len(x)-lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x to calculate Cov
    sum_product = np.sum((y1-np.mean(x))*(y2-np.mean(x)))
    # Normalize with var of whole series
    return sum_product / ((len(x) - lag) * np.var(x))


n = 20
lmin = 2

l     = np.arange(2, 2+n)
clth  = l**1.5
sigma = 5 

cl = np.random.normal(loc=clth, scale=sigma, size=n)
nlags = 10

results = {}
results['acf']  = acf(cl, nlags=nlags-1, unbiased=False)
results['my_a'] = my_acf(cl, nlags)
results["acf_by_hand"] = [acf_by_hand(cl, lag) for lag in range(nlags)]
results["autocorr_by_hand"] = [autocorr_by_hand(cl, lag) for lag in range(nlags)]
results["autocorr"] = [pd.Series(cl).autocorr(lag) for lag in range(nlags)]

dat =  pd.DataFrame(results)
print(dat)
