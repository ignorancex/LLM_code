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
import datetime
import argparse
import json
import random
import pdb
import numpy as np
import time

cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]
sys.path.append(f'{root}/src/')

from model import ML
from algorithms import randomN, epsilon


def run_random100(model):
    FARSEC_HP = ML.get_HP_obj()
    res = randomN.exec_(model, FARSEC_HP, ML.evaluation, 30)

    # writing results to file
    outfile = open(f'{root}/results/{model}_random100.txt', 'a+')
    outfile.write(
        f'** {datetime.datetime.today().strftime("%b-%d-%Y %H:%M:%S")}\n')

    for resi in res:
        hp, values = resi[0], resi[1]
        outfile.write(f'H {FARSEC_HP.flatten_hp(hp)}\n')
        outfile.write(f'V {np.array2string(values)[1:-1]}\n')
    outfile.write('## END\n')
    outfile.close()


def run_epsilon(model):
    FARSEC_HP = ML.get_HP_obj()
    epsilon_list = [0.2, 0.2, 0.2]
    epsilon_start_time = time.time()
    res = epsilon.exec_(model, FARSEC_HP, ML.evaluation, 100, epsilon_list)
    epsilon_execution_time = time.time() - epsilon_start_time

    # writing results to file
    outfile = open(f'{root}/results/{model}_epsilon.txt', 'a+')
    outfile.write(
        f'** {datetime.datetime.today().strftime("%b-%d-%Y %H:%M:%S")}\n')

    outfile.write(f'Epsilon {epsilon_list}\n')
    outfile.write(f'Execution time: {epsilon_execution_time}\n')

    for resi in res:
        hp, values = resi[0], resi[1]
        outfile.write(f'H {FARSEC_HP.flatten_hp(hp)}\n')
        outfile.write(f'V {np.array2string(values)[1:-1]}\n')
    outfile.write('## END\n')
    outfile.close()


# initial call as "python experiments/main.py -h"
# NOTE this func is just a task dispatcher
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrance of the program.')
    parser.add_argument(
        '-db',
        '--database',
        help='Name of the database',
        default="ambari",
        metavar='DB')
    parser.add_argument(
        '-r',
        '--repeat',
        help='Number of repeats',
        type=int,
        metavar='N',
        default=1)
    parser.add_argument(
        '-alg',
        '--algorithm',
        help='Algorithm - random100/epsilon/...',
        choices=['random100', 'epsilon'],
        metavar='ALG',
        required=True)
    parser.add_argument(
        '-seed', '--seed', help='Seed of random number', default=None)
    args = vars(parser.parse_args())

    if args['seed'] is not None:
        random.seed(int(args['seed']))

    for r in range(args['repeat']):
        if args['algorithm'] == 'random100':
            run_random100(args['database'])
        if args['algorithm'] == 'epsilon':
            run_epsilon(args['database'])