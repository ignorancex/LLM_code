import os
import fnmatch
import pandas as pd
import numpy as np
import numba as nb
# from certify_K import certify_K
from certify_K_fast import certify_K
from operator import itemgetter
# from scipy.special import comb, factorial
from time import time
import argparse
from tqdm import tqdm
from numba import jit, njit
from numba.typed import Dict 
from numba.core import types
from multiprocessing import Pool, Manager


parser = argparse.ArgumentParser(description='Compute the number of data points in each region')
parser.add_argument("--fn", type=str, default='example', help="dataset")
parser.add_argument("--range", type=int, default=100, help="range of certified perturbation size")
parser.add_argument("--K", type=int, default=1, help="binary data")

args = parser.parse_args()


# global_comb = Manager().dict()
# global_powe = Manager().dict()

# nb_global_comb = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.int64)
# nb_global_powe = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.int64)
# global_comb = dict()
# global_powe = dict()


@njit
def factorial(x, y):
    n = np.float64(1.)
    for i in range(x+1, y+1):
        n *= i
    return n


@njit
def comb(n, k):
    if k >= n-k:
        return factorial(k, n) / factorial(0, n-k)
    else:
        return factorial(n-k, n) / factorial(0, k)


@njit
def powe(k, p):
    return k ** p


# @njit
def my_comb(d, m, global_comb):
    if (d, m) not in global_comb:
        global_comb[(d, m)] = comb(d, m)
    # global_comb[(d, m)] = comb(d, m)
    
    return global_comb[(d, m)], global_comb


def my_powe(k, p, global_powe):
	if (k, p) not in global_powe:
		global_powe[(k, p)] = powe(k, p)

	return global_powe[(k, p)], global_powe


def certified_loop(r_range, m_range, global_d, real_ttl, K, fn):
    # nb_global_comb = dict()
    # nb_global_powe = dict()
    with Pool() as pool:
        args_list = [(r, m_range, global_d, real_ttl, K, fn) for r in range(r_range[0], r_range[1])]
        result_list = pool.map(inner_loop, args_list)
    # for r in range(r_range[0], r_range[1]):
    #     ttl = 0
    #     complete_cnt = []
    #     for m in tqdm(range(m_range[0], m_range[1])):
    #         # start = time()
    #         for n in range(m, min(m+r, global_d)+1):
    #             if r == 0 and m == 0 and n == 0:
    #                 c = 1
    #             elif (r == 0 and m != n) or np.min([m, n]) < 0 or np.max([m, n]) > global_d or m + n < r:
    #                 c = 0
    #             elif r == 0:
    #                 comb_res, nb_global_comb = my_comb(global_d, m, nb_global_comb)
    #                 powe_res, nb_global_powe = my_powe(K, m, nb_global_powe)
    #                 c = comb_res * powe_res
    #             else:
    #                 c = 0
    #                 # the number which are assigned to the (d-r) dimensions
    #                 for i in range(max(0, n-r), min(m, global_d-r, int(np.floor((m+n-r) * 0.5))) + 1):
    #                     if (m+n-r) / 2 < i:
    #                         break
    #                     x = m - i
    #                     y = n - i
    #                     j = x + y - r
    #                     # j = 0 ## if K = 1
    #                     # the second one implies n <= m+r
    #                     if j < 0 or x < j:
    #                         continue
    #                     powe_res, nb_global_powe = my_powe(K-1, j, nb_global_powe)
    #                     comb_res_1, nb_global_comb = my_comb(r, x-j, nb_global_comb)
    #                     comb_res_2, nb_global_comb = my_comb(r-x+j, j, nb_global_comb)
    #                     tmp = powe_res * comb_res_1 * comb_res_2
    #                     if tmp != 0:
    #                         powe_res, nb_global_powe = my_powe(K, i, nb_global_powe)
    #                         comb_res, nb_global_comb = my_comb(global_d-r, i, nb_global_comb)
    #                         tmp *= comb_res * powe_res
    #                         c += tmp
                
    #             if c != 0:
    #                 complete_cnt.append(((m, n), c))
    #                 ttl += c
    #                 # symmetric between d, m, n, r and d, n, m, r
    #                 if n > m:
    #                     ttl += c
            
    #         if m % 100 == 0:
    #             # print('r = {}, m = {}/{}, ttl ratio = {}, # of dict = {}'.format(r, m, m_range[1], ttl / real_ttl, len(complete_cnt)))
    #             print(fn, len(nb_global_powe), len(nb_global_comb))
        
    #     np.save('list_counts/{}/complete_count_{}'.format(fn, r), complete_cnt)

        # del complete_cnt 
        # del global_comb, global_powe

        # nb_global_comb = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.int64)
        # nb_global_powe = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.int64)
        # nb_global_comb = dict()
        # nb_global_powe = dict()
        
        # global_comb = dict()
        # global_powe = dict()


def inner_loop(args):
    r, m_range, global_d, real_ttl, K, fn = args
    ttl = 0
    nb_global_comb = dict()
    nb_global_powe = dict()
    complete_cnt = []
    for m in tqdm(range(m_range[0], m_range[1])):
        # start = time()
        for n in range(m, min(m+r, global_d)+1):
            if r == 0 and m == 0 and n == 0:
                c = 1
            elif (r == 0 and m != n) or min(m, n) < 0 or max(m, n) > global_d or m + n < r:
                c = 0
            elif r == 0:
                comb_res, nb_global_comb = my_comb(global_d, m, nb_global_comb)
                powe_res, nb_global_powe = my_powe(K, m, nb_global_powe)
                c = comb_res * powe_res
                if np.isnan(c):
                    print([comb_res, powe_res, global_d, m, K])
                    return
                    # continue
            else:
                c = 0
                # the number which are assigned to the (d-r) dimensions
                for i in range(max(0, n-r), min(m, global_d-r, int(np.floor((m+n-r) * 0.5))) + 1):
                    if (m+n-r) / 2 < i:
                        break
                    x = m - i
                    y = n - i
                    j = x + y - r
                    # j = 0 ## if K = 1
                    # the second one implies n <= m+r
                    if j < 0 or x < j:
                        continue
                    powe_res, nb_global_powe = my_powe(K-1, j, nb_global_powe)
                    comb_res_1, nb_global_comb = my_comb(r, x-j, nb_global_comb)
                    comb_res_2, nb_global_comb = my_comb(r-x+j, j, nb_global_comb)
                    tmp = powe_res * comb_res_1 * comb_res_2
                    if tmp != 0:
                        powe_res, nb_global_powe = my_powe(K, i, nb_global_powe)
                        comb_res, nb_global_comb = my_comb(global_d-r, i, nb_global_comb)
                        tmp *= comb_res * powe_res
                        c += tmp
                        if np.isnan(c):
                            print([tmp, comb_res, powe_res, comb_res_1, comb_res_2, K, i, global_d-r])
                            return
                            # continue
                
            if c != 0:
                complete_cnt.append(((m, n), c))
                ttl += c
                # symmetric between d, m, n, r and d, n, m, r
                if n > m:
                    ttl += c
            
        if m % 100 == 0:
            # print('r = {}, m = {}/{}, ttl ratio = {}, # of dict = {}'.format(r, m, m_range[1], ttl / real_ttl, len(complete_cnt)))
            print(fn, len(nb_global_powe), len(nb_global_comb))
            # print(complete_cnt)
    
    # np.save('list_counts/{}/complete_count_{}'.format(fn, r), np.array(complete_cnt, dtype=object))


# @njit
# def certified_loop(r_range, m_range, global_d, real_ttl, K, fn):
#     nb_global_comb = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.int64)
#     nb_global_powe = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.int64)
#     for r in range(r_range[0], r_range[1]):
#         ttl = 0
#         complete_cnt = []
#         for m in range(m_range[0], m_range[1]):
#             # start = time()
#             for n in range(m, min(m+r, global_d)+1):
#                 if r == 0 and m == 0 and n == 0:
#                     c = 1
#                 elif (r == 0 and m != n) or np.min(m, n) < 0 or np.max(m, n) > global_d or m + n < r:
#                     c = 0
#                 elif r == 0:
#                     if (global_d, m) not in nb_global_comb:
#                         comb_res = np.math.factorial(global_d) // (np.math.factorial(m) * np.math.factorial(global_d - m))
#                         # comb_res = comb(global_d, m)
#                         nb_global_comb[(global_d, m)] = comb_res
#                     else:
#                         comb_res = nb_global_comb[(global_d, m)]
#                     if (K, m) not in nb_global_powe:
#                         powe_res = K ** m
#                         nb_global_powe[(K, m)] = powe_res
#                     else:
#                         powe_res = nb_global_powe[(K, m)]
#                     c = comb_res * powe_res
#                 else:
#                     c = 0
#                     # the number which are assigned to the (d-r) dimensions
#                     for i in range(max(0, n-r), min(m, global_d-r, int(np.floor((m+n-r) * 0.5))) + 1):
#                         if (m+n-r) / 2 < i:
#                             break
#                         x = m - i
#                         y = n - i
#                         j = x + y - r
#                         # j = 0 ## if K = 1
#                         # the second one implies n <= m+r
#                         if j < 0 or x < j:
#                             continue
#                         if (K-1, j) not in nb_global_powe:
#                             powe_res = (K-1) ** j
#                             nb_global_powe[(K-1, j)] = powe_res
#                         else:
#                             powe_res = nb_global_powe[(K-1, j)]
#                         if (r, x-j) not in nb_global_comb:
#                             comb_res_1 = np.math.factorial(r) // (np.math.factorial(r) * np.math.factorial(r - x + j))
#                             # comb_res_1 = comb(r, x-j)
#                             nb_global_comb[(r, x-j)] = comb_res_1
#                         else:
#                             comb_res_1 = nb_global_comb[(r, x-j)]
#                         if (r-x+j, j) not in nb_global_comb:
#                             comb_res_2 = np.math.factorial(r-x+j) // (np.math.factorial(r-x+j) * np.math.factorial(r - x))
#                             # comb_res_2 = comb(r-x+j, j)
#                             nb_global_comb[(r-x+j, j)] = comb_res_2
#                         else:
#                             comb_res_2 = nb_global_comb[(r-x+j, j)]
#                         tmp = powe_res * comb_res_1 * comb_res_2
#                         if tmp != 0:
#                             if (K, i) not in nb_global_powe:
#                                 powe_res = K ** i
#                                 nb_global_powe[(K, i)] = powe_res
#                             else:
#                                 powe_res = nb_global_powe[(K, i)]
#                             if (global_d-r, i) not in nb_global_comb:
#                                 comb_res = np.math.factorial(global_d-r) // (np.math.factorial(global_d-r) * np.math.factorial(global_d-r-i))
#                                 # comb_res = comb(global_d-r, i)
#                                 nb_global_comb[(global_d-r, i)] = comb_res
#                             else:
#                                 comb_res = nb_global_comb[(global_d-r, i)]
#                             tmp *= comb_res * powe_res
#                             c += tmp
                
#                 if c != 0:
#                     complete_cnt.append(((m, n), c))
#                     ttl += c
#                     # symmetric between d, m, n, r and d, n, m, r
#                     if n > m:
#                         ttl += c
            
#             if m % 100 == 0:
#                 print('r = {}, m = {:10d}/{:10d}, ttl ratio = {:.4f}, # of dict = {}'.format(r, m, m_range[1], ttl / real_ttl, len(complete_cnt)))
#                 print(fn, len(nb_global_powe), len(nb_global_comb))
        
#         np.save('list_counts/{}/complete_count_{}'.format(fn, r), complete_cnt)

#         # del complete_cnt 
#         # del global_comb, global_powe

#         nb_global_comb = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.int64)
#         nb_global_powe = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.int64)
        
#         # global_comb = dict()
#         # global_powe = dict()




if __name__ == "__main__":


    global_d =  1 * 3000


    K = args.K
    r_range = [0, args.range]


    m_range = [0, global_d+1]



    print('fn =', args.fn, 'Range of L0 norm =', r_range, 'm_range =', m_range, 'global_d:', global_d, 'data type:', K)

    real_ttl = (K+1)**global_d

    certified_loop(r_range, m_range, global_d, real_ttl, K, args.fn)


    # dataset = 'german'

    # df_list_study_x = []
    # for pa in [0.6]:
    #     for px in [5e-3, 5e-2, 5e-1, 5]:

    #         file_name = "gcn_" + dataset + "_" + str(int(10 * pa)) + "_" + str(int(1000 * px)) + ".txt"
    #         path = "exps_param_study/"

    #         try:
    #             df = pd.read_csv(os.path.join(path, file_name), sep='\t')
    #             df_list_study_x.append(df.loc[:, "Rx"].to_numpy().tolist())
    #         except FileNotFoundError:
    #             continue

    dataset = 'credit'

    df_list_study_a = []
    for pa in [0.6, 0.7, 0.8, 0.9]:
        for px in [5e-2]:

            file_name = "gcn_" + dataset + "_" + str(int(10 * pa)) + "_" + str(int(1000 * px)) + ".txt"
            path = "exps_param_study/"

            print(os.path.join(path, file_name))

            try:
                df = pd.read_csv(os.path.join(path, file_name), sep='\t')

                pa_bar_list = df.loc[:, "pABar"].to_numpy()
                
                corr_list = df.loc[:, "correct"].to_numpy()

                final = (pa_bar_list * corr_list).tolist()

                ra_list = [certify_K(pAHat, pa, global_d, [0, args.range], "example") for pAHat in pa_bar_list]
                df_list_study_a.append(ra_list)
            except FileNotFoundError:
                continue

    # print('german')
    # print(df_list_study_x)

    print('credit')
    print(df_list_study_a)
    
