#!/bin/python
# bench other python packages

from __future__ import absolute_import
from __future__ import print_function
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import *
import numpy as np
import math
import sys
import random
import time
import statistics

l = 1<<15
n = 128
w = 28


def random_binary_array(length: int, weight=0):
    """
    :param length:
    :return:  [0, 1, 0, 0, 1, ... ]
    """
    r = []
    if weight == 0:
        for i in range(0, length):
            r.append(random.randint(0, 1))
    else:
        r = np.array([1] * weight + [0] * (length - weight))
        np.random.shuffle(r)

    return r


def random_binary_string(length: int):
    """
    :param length:
    :return: 110011...1
    """
    r = ""
    for i in range(0, length):
        r += str(random.randint(0, 1))

    return bin(int(r))


def xor_binary_array(e1, e2):
    assert (len(e1) == len(e2))

    res = [0 for i in range(0, len(e1))]
    for i in range(0, len(e1)):
        res[i] = (e1[i] + e2[i]) % 2

    return res


def weight_binary_array(e):
    return sum(e)


def create_test_list(length: int, n: int, minimal_weight: int):
    """
    because kd_tree dont let us choose the actual weight diff we have to choose big n to ensure that the closest vector is also the nearest neighbor with high propability
    :param length:
    :param minimal_weight:
    :return:
    """
    assert(n > minimal_weight)

    X = []
    for i in range(0, length):
        X.append(random_binary_array(n))

    # return just the set and dont implant the golden element.
    if minimal_weight == 0:
        return None, np.asarray(X), -1

    # implant the 'closest' vector
    index = random.randint(0, length-1)
    res = random_binary_array(n, minimal_weight)

    X[index] = xor_binary_array(X[index], res)
    # check_correctness(X[index], res, minimal_weight)
    return res, np.asarray(X), index


def create_test_zero_list(length: int, n: int, minimal_weight: int):
    """
    instead of choosing a random element and add it to one element in the list. We zero out one random element in the list.
    :param length:
    :param minimal_weight:
    :return:
    """
    assert(n > minimal_weight)

    X = []
    for i in range(0, length):
        X.append(random_binary_array(n))

    # return just the set and dont implant the golden element.
    if minimal_weight == 0:
        return None, np.asarray(X), -1

    # implant the 'closest' vector
    index = random.randint(0, length-1)
    res = random_binary_array(n, minimal_weight)

    for i in range(0, n):
        X[index][i] = 0

    # check_correctness(X[index], res, minimal_weight)
    return res, np.asarray(X), index


def check_correctness(e1, e2, w):
    assert (len(e1) == len(e2))

    res = [0 for _ in range(0, len(e1))]
    w_ = 0
    for i in range(0, len(e1)):
        res[i] = (e1[i] + e2[i]) % 2
        w_ += res[i]

    #print("Correctness")
    #print(e1)
    #print(e2)
    print(res, " ", w_, "=", w)

def quadratic_search(X, Y, weight=0, find_all=False):
    """
    :param X:       List 1
    :param Y:       List 2
    :param weight:  if set to zero the smallest element is searched
    :param find_all:
    :return:
    """
    r = []
    smallest_w = 100000000
    smallest_x = []
    smallest_y = []
    smallest_i = 0
    smallest_j = 0

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            w_ = weight_binary_array(xor_binary_array(x, y))
            if (w_ == weight) and (weight > 0):
                r.append((x, y, i, j))

                if find_all:
                    return r

            if (weight == 0) and (w_ < smallest_w):
                smallest_w, smallest_x, smallest_y, smallest_i, smallest_j = w_, x, y, i, j

    if weight == 0:
        return [smallest_x, smallest_y, smallest_i, smallest_j]

    return r


def test_quadratic_zero():
    query, X, index = create_test_zero_list(l, n, w)
    result = quadratic_search([query], X)

    check_correctness(result[0], result[1], w)
    print(result)
    print("index: ", index , "is:", result[2], result[3])

    for i in range(n):
        if X[index][i] != result[1][i]:
            print("incorrect")
            print(X[index])
            print(result[1])
            break


def test_scikit_zero():
    _, X, _ = create_test_list(l, n, w)
    # clear out a 'random' element
    for i in range(0, n):
        X[0][i] = 0

    print("X0", X[0])
    kdt = KDTree(X, leaf_size=30, metric='l1')

    # choose a small element. (w needs to be really small.)
    query = random_binary_array(n, w)
    dist, ind = kdt.query([query], k=1)
    sol = xor_binary_array(query, X[ind[0][0]])

    # check_correctness(query, X[ind[0][0]], dist[0][0])
    print(sol)
    print("index: ", ind[0][0])
    print("weight:", dist[0][0], w, weight_binary_array(sol))


def test_scikit():
    query, X, sol_index = create_test_list(l, n, w)
    kdt = KDTree(X, leaf_size=30, metric='l1')
    # kdt.query(X, k=2, return_distance=False)

    dist, ind = kdt.query([query], k=1)
    sol = xor_binary_array(query, X[ind[0][0]])

    # check_correctness(query, X[ind[0][0]], dist[0][0])
    print(sol)
    print("index: ", ind[0][0], sol_index)
    print("weight:", dist[0][0], w, weight_binary_array(sol))

    #assert (ind[0][0] == sol_index)


def bench_Quadratic(X, Y):
    return quadratic_search(X, Y, w, True)


def bench_KDTRee(X, Y):
    kdt = KDTree(X, leaf_size=30, metric='l1')

    r = 0
    for y in Y:
        dist, ind = kdt.query([y], k=1)
        r += dist[0][0]
        # print(dist, ind)
        # print("Query", y, " found:", X[ind])
    return r


def bench_BallTree(X, Y):
    tree = BallTree(X, leaf_size=2, metric='l1')

    r = 0
    for y in Y:
        dist, ind = tree.query([y], k=1)
        r += dist[0][0]
    return r


def bench_scikit():
    functions = [bench_Quadratic]# , bench_KDTRee, bench_BallTree]
    times = {f.__name__: [] for f in functions}

    # create a list of 'l' 'small' elements to query
    Y = []
    for i in range(0, l):
        Y.append(random_binary_array(n, w))

    Y = np.asarray(Y)

    for func in functions:
        query, X, sol_index = create_test_zero_list(l, n, w)

        for i in range(3):
            t0 = time.time()
            func(X, Y)
            times[func.__name__].append(time.time() - t0)

    for name, numbers in times.items():
        print('FUNCTION:', name, 'Used', len(numbers), 'times')
        print('\tMEDIAN', statistics.median(numbers))
        print('\tMEAN  ', statistics.mean(numbers))
        print('\tSTDEV ', statistics.stdev(numbers))


if __name__ == "__main__":
    bench_scikit()
    # test_quadratic_zero()
    #test_scikit_zero()
    #exit(1)
    #test_scikit()