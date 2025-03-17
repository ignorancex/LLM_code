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

from __future__ import division

import os
import sys
import collections
import random
import pdb
import numpy as np

cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]
sys.path.append(f'{root}/src')
from model import ML


def get_diff_weights(values, best_values, epsilon_lst, lambda_list):
    """
    return the diff_weights
    """
    diff_weight = 0.0

    if sum(i[0] * i[1] for i in zip(values, lambda_list)) - sum(
            i[0] * i[1] for i in zip(best_values, lambda_list)) > sum(
            i[0] * i[1] for i in zip(epsilon_lst, lambda_list)):
        diff_weight += 1
    elif sum(i[0] * i[1] for i in zip(values, lambda_list)) - sum(
            i[0] * i[1] for i in zip(best_values, lambda_list)) < -1 * sum(
            i[0] * i[1] for i in zip(epsilon_lst, lambda_list)):
        diff_weight -= 1

    # diff_weight += collections.Counter(
    #     (values - best_values) > epsilon_lst)[True] / len(epsilon_lst)
    # diff_weight -= collections.Counter(
    #     (values - best_values) < -1 * epsilon_lst)[True] / len(epsilon_lst)
    return diff_weight


def get_best_values(values1, values2, lambda_list):
    if sum(i[0] * i[1] for i in zip(values1, lambda_list)) >= sum(i[0] * i[1] for i in zip(values2, lambda_list)):
        return values1
    else:
        return values2


# Assumping all objectives are to maximize
def exec_(dataset, HP_obj, eval_func, N1, epsilon_lst):
    weights = dict()
    epsilon_lst = np.array(epsilon_lst)
    for name in HP_obj.pres + HP_obj.learns:
        weights[name] = 0

    # Step1: find the best preprocessor and best learner
    best_values = np.array([np.NINF for _ in range(len(epsilon_lst))])

    lambda_list = [0.25, 0.25, 0.50]  # TODO change here

    for _ in range(N1):
        rnd_hp, pre, learner = HP_obj.get_rnd_hp_without_range()
        # print(rnd_hp)
        values = eval_func(dataset, rnd_hp)
        diff_weight = get_diff_weights(values, best_values, epsilon_lst, lambda_list)

        weights[pre] += diff_weight
        weights[learner] += diff_weight

        # best_values = np.maximum(best_values, values)
        best_values = get_best_values(best_values, values, lambda_list)

    best_pre, best_learner = HP_obj.pres[0], HP_obj.learns[0]
    for i in HP_obj.pres:
        if weights[i] > weights[best_pre]:
            best_pre = i
    for j in HP_obj.learns:
        if weights[j] > weights[best_learner]:
            best_learner = j

    print('best_pre', best_pre)
    print('best_learner', best_learner)

    RES = list()
    ks = np.random.dirichlet(
        np.ones(len(epsilon_lst)), size=10)  # TODO change size?

    for k in ks:
        hp_lst, v_lst, w_lst, values_lst = list(), list(), list(), list()

        ep = epsilon_lst.dot(k)

        for i in range(20):
            if i < 10:  # turn in process of stage II
                rnd_hp, _, _ = HP_obj.get_rnd_hp_without_range(
                    best_pre, best_learner)
            else:
                best_hp, worse_hp = hp_lst[np.argmax(w_lst)], hp_lst[np.argmin(
                    w_lst)] # Todo check here, w_lst has issue?
                rnd_hp = HP_obj.get_ran_between_half_of(best_hp, worse_hp)

            values = eval_func(dataset, rnd_hp)
            tmp_v = values.dot(k)

            if i == 0:
                hp_lst.append(rnd_hp)
                v_lst.append(tmp_v)
                w_lst.append(0.0)
                values_lst.append(values)
                continue

            if tmp_v > max(v_lst) + ep: # TODO check the logic here
                w_lst.append(max(w_lst) + 1)
            elif tmp_v < min(v_lst) - ep:
                w_lst.append(min(w_lst) - 1)
            else:
                w_lst.append(sum(w_lst) / len(w_lst))

            hp_lst.append(rnd_hp)
            v_lst.append(tmp_v)
            values_lst.append(values)
        best_idx = np.argmax(v_lst)
        RES.append((hp_lst[best_idx], values_lst[best_idx]))

    return RES


# if __name__ == '__main__':
#     random.seed(2019)
#     FARSEC_HP = ML.get_HP_obj()
#     exec_('derby', FARSEC_HP, ML.evaluation, 20, [0.2, 0.2, 0.2])
