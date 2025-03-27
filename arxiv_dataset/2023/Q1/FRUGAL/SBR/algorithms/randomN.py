#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, Jianfeng Chen <jchen37@ncsu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, _distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]
sys.path.append(f'{root}/src')

from model.utilities import get_best


# Assumping all objectives are to maximize
# Random evaluatate N hyperparameters. return the best ones.
# for the multiobjs, return the pareto frontier
def exec_(dataset, HP_obj, eval_func, N):
    Values = list()
    HPs = list()
    for repeat in range(N):
        if repeat % 10 == 1:
            print(f'Random hyperparemter generated for {dataset} {repeat}/{N}')
        rnd_hp, _, _ = HP_obj.get_rnd_hp_without_range()
        values = eval_func(dataset, rnd_hp)

        HPs.append(rnd_hp)
        Values.append(values)
    best_indices = get_best(Values, ignore_idx=[2])

    res = [(HPs[i], Values[i]) for i in best_indices]
    return res