import numpy as np
import scipy.linalg as sl
from scipy import special
from collections import deque
import functools as fts
import operator as op

class ADWIN2:

    # extend the original ADWIN2 to multi-dimensional cases
    # drop the oldest bucket at every dimension if there exists a dimension that insists they should shrink

    
    def __init__(self, a = None):
        self.a = 1 if a is None else a

    def shrink(self, score, var_data, delta): # data: supposed to be 1-dimensional
        n = len(score)
        istrue = np.zeros(n) # array each element of which indicates whether the corresponding stat is over threshold or not
        for i in range(0,n):
            n0 = i + 1 # the number o the data that is lefter than the cutpoint
            n1 = n + 1  - n0 
            m = 1 / (1 / n0 + 1 / n1) # harmonic mean 
            _delta = delta / np.log(n + 1) 
        #    epsilon = np.sqrt(((max - min) * r)  ** 2/(2 * m) * np.log(4/_delta)) # the threshold subject to the position of the cutpoint
        epsilon = np.sqrt(2/m * var_data * np.log(2/_delta)) + 2/(3 * m) * np.log(2/_delta)
        if score[i] > epsilon: 
            istrue[i] = True
        return any(istrue) # return 1 if there exists a stat which is over threshold

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
                combination = (hoge[0] + fuga[0], i + 1)# combine the lefest ones s.t. length = i
                new_buckets.pop() 
                new_buckets.pop() # delete two unnecessary buckets
                new_buckets.append(combination) # apped a combination bucket
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

    
    def transform(self, X, delta,  M):
        Xmat = X.reshape(len(X), -1)
        n, m = Xmat.shape
        multi_buckets = np.zeros((m,), dtype = object) # make  multi-dimensional buckets
        for i in range(0, m):
            multi_buckets[i] = deque([]) # initialization
        size = np.zeros(n,) 
        for t in range(0, n):
            for i in range(0, m):
                newT = self.suffi_stats(Xmat[t, i])
                multi_buckets[i] = self.combine(newT, multi_buckets[i], M) # combine buckets on each dimension
            if len(multi_buckets[0]) == 1: # if the number of buckets is 1
                size[t] = 2 ** multi_buckets[0][0][1]
                continue
            bucket_num = len(multi_buckets[0]) 
            score_lists = np.zeros((m, bucket_num - 1))
            shrink_list = np.zeros((m,))
            for i in range(0, m):
                score_lists[i], data_num = _adwin2(multi_buckets[i])
                var_data = sum(map((lambda x: x[0][1]), multi_buckets[i])) / data_num - (sum(map((lambda x: x[0][0]), multi_buckets[i])) / data_num) ** 2
                shrink_list[i] = self.shrink(score_lists[i], var_data, delta)
            while any(shrink_list) == 1 and len(multi_buckets[0]) > 1:
                for i in range(0, m):
                    multi_buckets[i].popleft()
                    bucket_num = len(multi_buckets[0])
                    if bucket_num == 1:
                        continue
                    score_lists = np.zeros((m, bucket_num - 1))
                    bucket_num = len(multi_buckets[0])
                    score_lists[i], data_num = _adwin2(multi_buckets[i])
                    var_data = sum(map((lambda x: x[0][1]), multi_buckets[i])) / data_num - (sum(map((lambda x: x[0][0]), multi_buckets[i])) / data_num) ** 2
                    shrink_list[i] = self.shrink(score_lists[i], var_data, delta)
            size[t] = sum(map((lambda x: 2 ** x[1]), multi_buckets[0]))
        return size

    

def _adwin2(buckets):
    
    # input: buckets
    # output: score_list, number of data
    
    n = len(buckets)
    score_list = np.zeros((n - 1,))
    for cut in range(1, n):
        former = (sum(map((lambda x: x[0][0]), [buckets[i] for i in range(0, cut)])), sum(map((lambda x: 2 ** x[1]), [buckets[i] for i in range(0, cut)])))
        latter = (sum(map((lambda x: x[0][0]), [buckets[i] for i in range(cut, n)])), sum(map((lambda x: 2 ** x[1]), [buckets[i] for i in range(cut, n)])))
        score_list[cut - 1] = abs(former[0] / former[1] - latter[0] / latter[1])
    return score_list, former[1] + latter[1]


##############################################################################




