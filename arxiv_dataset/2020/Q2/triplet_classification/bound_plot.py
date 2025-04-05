#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def func(prior=0.7):
    p, n = prior, 1 - prior
    pi_t = 1 - p*n
    A = (p*p*p + 2*p*p*n) / pi_t
    B = (2*p*n*n + n*n*n) / pi_t
    a = p*p + n*n + A*A
    b = 2*p*n + A*B
    c = n*n + B*B + p*p
    deno = np.abs(a*c - b*b)
    nomi1 = np.abs(p*(c*p - b*n))
    nomi2 = np.abs(n*(a*n - b*p))
    nomi3 = np.abs(p*(c*A - b*B))
    nomi4 = np.abs(n*(a*B - b*A))
    nomi5 = np.abs(p*(c*n - b*p))
    nomi6 = np.abs(n*(a*p - b*n))
    nomi = nomi1 + nomi2 + nomi3 + nomi4 + nomi5 + nomi6
    return nomi / deno


def plot_bound():
    ps = np.arange(0.05, 0.95, 0.005)
    ps = np.array([_p for _p in ps if np.abs(_p - 0.5) > 0.005])
    bs = func(ps)

    plt.plot(ps, bs)
    plt.xticks(np.arange(0.1, 1, 0.1))
    plt.xlabel('Class Prior')
    plt.ylabel('The Coefficient Term')
    plt.savefig('bound.eps')


if __name__ == '__main__':
    plot_bound()
