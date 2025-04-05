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
input_file = 'settlements.json'
H_input_file = exp + 'H_sample.mat'
F_input_file = exp + 'F_sample.npy'
zeus_output_file = exp + 'VAL_F-MWSP_settle_output.csv'
val_golden_file = exp + 'VAL_settle_golden.csv'

def saveCluster(singleton_id, data_d, cluster_membership, output_file, input_file):

    ## TODO: Refine this to remove input_file
    with open(output_file, 'w') as f_output, open(input_file) as f_input:
        writer = csv.writer(f_output)

        heading_row = ['id']
        heading_row.insert(0, 'confidence_score')
        heading_row.insert(0, 'Cluster ID')
        writer.writerow(heading_row)

        for line in f_input:
            row = json.loads(line)
            row_id = int(row['id'])
            row_output = [row_id]
            if row_id in cluster_membership:
                cluster_id, score = cluster_membership[row_id]
            elif row_id in data_d:
                cluster_id, score = singleton_id, None
                singleton_id += 1
            else:
                continue
            row_output.insert(0, score)
            row_output.insert(0, cluster_id)
            writer.writerow(row_output)

print('importing data ...')
val_data_d, val_label_d = readGoldData(input_file, val_golden_file)
                
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
        cluster_membership[rid] = (cid, 0.)

singleton_id = max_cid + 1
saveCluster(singleton_id,
            val_data_d,
            cluster_membership,
            zeus_output_file, input_file)
