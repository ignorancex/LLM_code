from future.builtins import next

import os
import csv
import re
import logging
import optparse
import collections
import numpy as np
from scipy import stats

import dedupe
from unidecode import unidecode

from utils import *

import scipy.io

exp = 'results_1/'
input_file = 'csv_example_messy_input.csv'
H_input_file = exp + 'H_sample.mat'
F_input_file = exp + 'F_sample.npy'
zeus_output_file = exp + 'VAL_F-MWSP_csv_example_output.csv'
val_golden_file = exp + 'VAL_csv_example_golden.csv'

def saveCluster(singleton_id, data_d, cluster_membership, canonical_keys, output_file, input_file):

    with open(output_file, 'w') as f_output, open(input_file) as f_input:
        writer = csv.writer(f_output)
        reader = csv.reader(f_input)

        heading_row = next(reader)
        heading_row.insert(0, 'confidence_score')
        heading_row.insert(0, 'Cluster ID')
        for key in canonical_keys:
            heading_row.append('canonical_' + key)

        writer.writerow(heading_row)

        for row in reader:
            row_id = int(row[0])
            if row_id in cluster_membership:
                cluster_id = cluster_membership[row_id]["cluster id"]
                canonical_rep = cluster_membership[row_id]["canonical representation"]
                row.insert(0, cluster_membership[row_id]['confidence'])
                row.insert(0, cluster_id)
                for key in canonical_keys:
                    row.append(canonical_rep[key].encode('utf8'))
                writer.writerow(row)
            elif row_id in data_d:
                row.insert(0, None)
                row.insert(0, singleton_id)
                singleton_id += 1
                for key in canonical_keys:
                    row.append(None)
                writer.writerow(row)


print('importing data ...')
val_data_d, val_label_d, gold_fieldnames = readGoldData(val_golden_file)
                
H_sol_cluster = scipy.io.loadmat(H_input_file)['H']['sol_cluster'][0][0]

# Correct Indices
H_sol_cluster_corr = {}
F = np.load(F_input_file).item()
for rid, cid in enumerate(H_sol_cluster):
    if rid+1 in F['mapN2O']:
        oid = F['mapN2O'][rid+1]
        #print(oid, cid[0])
        H_sol_cluster_corr[oid] = cid[0]
    else:
        print(rid, cid)
        print('ERROR! index not found')
        exit(1)
        


cluster_membership = {}
max_cid = -float('inf')
for rid, cid in H_sol_cluster_corr.items():
    if cid != 0:
        if cid > max_cid:
            max_cid = cid
        cluster_d = [val_data_d[rid]]
        canonical_rep = dedupe.canonicalize(cluster_d)
        cluster_membership[rid] = {
            "cluster id" : cid,
            "canonical representation" : canonical_rep,
            "confidence": 0.
        }

singleton_id = max_cid + 1
saveCluster(singleton_id,
            val_data_d,
            cluster_membership,
            canonical_rep.keys(),
            zeus_output_file, input_file)
