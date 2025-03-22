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

import os
import sys
import random
import numpy as np
import pandas as pd
import pdb

cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]
sys.path.append(f'{root}/src')
from model.utilities import get_score


class Hyperparameter:
    def __init__(self, preprocessing_names, learner_names):
        self.pres = preprocessing_names  # [SMOTE, ...]
        self.learns = learner_names  # ['DT', 'RF', ...]
        self.hp_names = dict(
        )  # {'SMOTE':['m','k','r'], 'RF':['n_estimators',...]}
        self.ranges = dict()

    def register_hp(self, alg, name, lst):
        if alg not in self.hp_names:
            self.hp_names[alg] = [name]
        else:
            self.hp_names[alg].append(name)

        self.ranges[f'{alg}_{name}'] = lst

    def get_rnd_hp_without_range(self, pre=None, learner=None):
        # pdb.set_trace()
        res_HP = dict()

        if pre is None:
            pre = random.choice(self.pres)
        res_HP[pre] = dict()
        for hp in self.hp_names[pre]:
            res_HP[pre][hp] = random.choice(self.ranges[f'{pre}_{hp}'])

        if learner is None:
            learner = random.choice(self.learns)
        res_HP[learner] = dict()
        for hp in self.hp_names[learner]:
            res_HP[learner][hp] = random.choice(self.ranges[f'{learner}_{hp}'])

        return res_HP, pre, learner

    def get_ran_between_half_of(self, best_hp, worse_hp):
        res = {l: dict() for l in best_hp.keys()}
        for l in res:
            for para in self.hp_names[l]:
                if isinstance(self.ranges[f'{l}_{para}'][0], str):
                    res[l][para] = best_hp[l][para]
                else:  # numpy.int64 or numpy.float64
                    mid = (best_hp[l][para] + worse_hp[l][para]) / 2
                    idx_a = np.searchsorted(self.ranges[f'{l}_{para}'], mid)
                    idx_b = np.searchsorted(self.ranges[f'{l}_{para}'],
                                            best_hp[l][para])
                    if idx_a == idx_b:
                        res[l][para] = best_hp[l][para]
                        continue
                    if idx_a > idx_b:
                        idx_a, idx_b = idx_b, idx_a
                    res[l][para] = self.ranges[f'{l}_{para}'][
                        np.random.randint(idx_a, idx_b)]
        return res

    def flatten_hp(self, hp_dict):
        res = ""
        for alg in hp_dict.keys():
            for sub in self.hp_names[alg]:
                res += f"{alg}.{sub}={hp_dict[alg][sub]} | "
        return res

    def unflatten_hp(self, hp_str):
        res = dict()
        for component in hp_str.split(' '):
            if len(component) <= 2:
                continue
            alg, rest = component.split('.')[0], component.split('.')[1]
            sub, v = rest.split('=')[0], rest.split('=')[1]
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            if alg not in res: res[alg] = dict()
            res[alg][sub] = v
        return res