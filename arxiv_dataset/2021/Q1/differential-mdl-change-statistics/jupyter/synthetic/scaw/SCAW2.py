import numpy as np
import scipy.linalg as sl
from scipy import special
from collections import deque
import functools as fts
import operator as op

from numba import jit

class MDLCPStat_adwin2:

    def __init__(self, lossfunc=None, how_to_drop = 'none'):
        self.lossfunc = lnml_gaussian if lossfunc is None else lossfunc
        self.how_to_drop = how_to_drop
        assert how_to_drop in ('none', 'cutpoint', 'all')
        
        self.buckets = deque([])
        self.size = []

    def combine(self, newT, buckets, M):

        # input: newT(np.array), buckets(deque), M(int)
        # output: new_buckets according to the "bucket rule"
        
        new = (newT, 0.0) # make a new datum into a bucket: form of (suffi_stats, log(length))
        new_buckets = buckets
        new_buckets.append(new)
        k = new_buckets[0][1] # get the largest capacity in the buckets
        n = len(new_buckets)
        indicator = n - 1 # indicate the point where we finish combining buckets
        for i in range(0, int(k) + 1):
            counter = 0
            while indicator >= 0 and new_buckets[indicator][1] == i :
                counter += 1
                indicator -= 1
            if counter > M:
                indicator += 1
                new_buckets.rotate(n - 2 - indicator) # move unnecessary buckets to the rightest
                hoge = new_buckets[n - 2]
                fuga = new_buckets[n - 1]
                combination = (hoge[0] + fuga[0], i + 1) # combine the lefest ones s.t. length = i
                new_buckets.pop() 
                new_buckets.pop() # delete two unnecessary buckets
                new_buckets.append(combination) # append a combination bucket
                n -= 1
                new_buckets.rotate(-(n - 1 - indicator)) # return to where they were
        return new_buckets

    def suffi_stats(self, X):

        # input: X (multi-dimension vector)
        # output: a component of the sufficient statistics
        
        X_ = X.reshape((1, -1))
        x = X_
        xxT = X_.reshape((-1, 1)) @ X_
        return np.array([x, xxT]) # x.shape = (1, m), xxT.shape = (m, m)
    
    def transform(self, X, d, delta,  M):

        # input: data(multi-dimension vector), epsilon
        # output: size(np.array)
        #         size[0] : window size
        #         size[1] : max(score)
        
        buckets = deque([])
        n = len(X)
        size = np.zeros((2, n))  # size[0] = window size, size[1] = score at the cutpoint
        for t in range(0, n):
            newT = self.suffi_stats(X[t])
            buckets = self.combine(newT, buckets, M) # combine buckets
            if len(buckets) == 1:
                size[:, t] = 2 ** buckets[0][1], 0 # doesn't cut any bucket
                continue
            score = _mdlcpstat_adwin2(self.lossfunc, buckets)
            nn = sum(map((lambda x: 2 ** x[1]), buckets))
            threshold= np.log(1/delta) + d/2 * np.log(nn) + (1 + delta) * np.log(nn) + np.log(len(buckets) - 1)  
            if max(score) > threshold:
                if self.how_to_drop == 'cutpoint': 
                    cut = np.argmax(score)
                    for j in range(0, cut + 1):
                        buckets.popleft()
                    capacity_sum = sum(map((lambda x: 2 ** x[1]), buckets))
                    size[:, t] = capacity_sum, max(score)
                if self.how_to_drop == 'all':
                    buckets = deque([])
                    size[:, t] = 0, max(score)
            else:
                capacity_sum = sum(map((lambda x: 2 ** x[1]), buckets))
                size[:, t] = capacity_sum, max(score)
        return size
    


def _mdlcpstat_adwin2(lossfunc, buckets):
    n = len(buckets)
    stat_list = np.zeros(n-1)
    entire = (sum(map((lambda x: x[0]), buckets)), sum(map((lambda x: 2 ** x[1]), buckets)))
    num = entire[1]
    for cut in range(1, n):
        former = (sum(map((lambda x: x[0]), [buckets[i] for i in range(0, cut)])), sum(map((lambda x: 2 ** x[1]), [buckets[i] for i in range(0, cut)])))
        latter = (sum(map((lambda x: x[0]), [buckets[i] for i in range(cut, n)])), sum(map((lambda x: 2 ** x[1]), [buckets[i] for i in range(cut, n)])))
        stat_list[cut - 1] = lossfunc(entire) - lossfunc(former) - lossfunc(latter)
    return stat_list    



@fts.lru_cache(maxsize = None) # save time when call a function with the same argument several times
def multigamma_ln(a,d):
    return special.multigammaln(a,d)


def lnml_gaussian(bucket):

    # calculate the lnml code length of a bucket assuming gaussian family

    multigammaln = multigamma_ln
    n,m = float(bucket[1]), bucket[0][0].shape[1]
    if n <= 0:
        return np.nan
    nu = m         ### given (optimal when nu = m)
    sigma = 1      ### given
    mu = bucket[0][0] / (n + nu)
    S = (bucket[0][1] + nu * sigma ** 2 * np.matrix(np.identity(m)))/(n + nu) - mu.T.dot(mu)
    detS = sl.det(S.astype(np.float64))
    log_lnml =  m/2 * ((nu + 1) * np.log(nu) - n * np.log(np.pi) - (n + nu + 1) * np.log(n + nu)) + multigammaln((n+nu)/2,m) - multigammaln(nu/2,m) + m * nu * np.log(sigma) - (n + nu)/2 * np.log(detS)
    return -1 * log_lnml
