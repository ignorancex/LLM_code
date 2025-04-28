# -*- coding: utf-8 -*-
# @Time    : 5/11/24 11:33 AM
# @Author  : Fu-An Chao
# @Affiliation  : NTNU
# @Email   : fuann@ntnu.edu.tw
# @File    : collect_mdd.py

# collect mdd results of repeated experiment.

import argparse
import os
import re
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, help="directory to dump experiments")
args = parser.parse_args()

metrics_list = [
    "TA", "FR", "FA",
    "Correct Diag", "Error Diag",
    "Recall", "Precision", "F1",
    "FAR", "FRR",
    "DER", "WER", "SER"
]

# init
metrics_all = defaultdict(list)
for metrics in metrics_list:
    metrics_all[metrics]

# for each repeat experiment
for i in range(0, 10):
    cur_exp_dir = args.exp_dir + '/' + str(i)
    if os.path.isfile(cur_exp_dir + '/mdd_result.txt'):
        rf = open(cur_exp_dir + '/mdd_result.txt', "r")
        for line in rf.readlines():
            line = line.strip().split(" ")
            if len(line) == 1:
                continue
            metrics, values = line[0], line[1:]

            # exception (Correct Diag)
            if values[0] == "Diag:":
                metrics = metrics+" "+values[0]
                value = values[1]
            else:
                value = values[0]

            metrics = re.sub(r"[:%]", "", metrics)
            if metrics in metrics_list:
                metrics_all[metrics].append(float(value))

        rf.close()

# write results
if not os.path.exists(args.exp_dir):
    os.mkdir(args.exp_dir)

result_f = open(args.exp_dir + '/result_mdd.txt', "w")

for metrics in metrics_list:
    result = np.array(metrics_all[metrics])
    result_mean = np.mean(result, axis=0)
    result_std = np.std(result, axis=0)
    result_f.write(f"{metrics}: {result_mean:.4f}, {result_std:.4f} \n")

result_f.close()
