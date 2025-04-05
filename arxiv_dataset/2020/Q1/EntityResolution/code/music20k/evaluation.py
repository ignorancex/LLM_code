from future.builtins import next
from future.utils import viewitems

import os
import csv
import re
import logging
import optparse
import collections
import itertools
import numpy as np
from scipy import stats
import scipy.io
from sklearn import metrics

import dedupe
from unidecode import unidecode

from utils import *

exp = 'results/'
H_input_file = exp + 'H_sample.mat'

input_file = 'musicbrainz-20-A01.csv.dapo'
F_input_file = exp + 'F_sample.npy'
zeus_output_file = exp + 'VAL_F-MWSP_music20k_output.csv'
val_golden_file = exp + 'VAL_music20k_golden.csv'
manual_clusters = exp + 'VAL_music20k_golden.csv'
dedupe_clusters_HC = exp + 'VAL_music20k_output.csv'
dedupe_clusters_FMWSP = exp + 'VAL_F-MWSP_music20k_output.csv'
row_id_name = 'TID'

'''
Save observation ids and cluster ids
'''
def saveCluster(singleton_id, data_d, cluster_membership, output_file, input_file):
    with open(output_file, 'w') as f_output, open(input_file) as f_input:
        writer = csv.writer(f_output)
        reader = csv.reader(f_input)

        heading_row = next(reader)
        heading_row.insert(0, 'confidence_score')
        heading_row.insert(0, 'Cluster ID')
        writer.writerow(heading_row)

        for row in reader:
            row_id = int(row[0])
            if row_id in cluster_membership:
                cluster_id, score = cluster_membership[row_id]
            elif row_id in data_d:
                cluster_id, score = int(round(singleton_id)), None
                singleton_id += 1
            else:
                continue
            row.insert(0, score)
            row.insert(0, cluster_id)
            writer.writerow(row)

'''
Transform the .mat file to a python data structure
'''        
def transform_H():
    print('importing data ...')
    val_data_d, val_label_d, gold_fieldnames = readGoldData(input_file, val_golden_file)
                
    H_sol_cluster = scipy.io.loadmat(H_input_file)['H']['sol_cluster'][0][0]

    # Correct Indices
    H_sol_cluster_corr = {}
    F = np.load(F_input_file, allow_pickle=True).item()
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
            cluster_membership[rid] = (int(round(cid)), 0.)

    singleton_id = max_cid + 1
    saveCluster(singleton_id,
                val_data_d,
                cluster_membership,
                zeus_output_file, input_file)

'''
Compute precision, recall and F1 score by matching the predicted cluster ids
and true cluster ids
'''
def evaluateDuplicates(found_dupes, true_dupes):
    true_positives = found_dupes.intersection(true_dupes)
    false_positives = found_dupes.difference(true_dupes)
    uncovered_dupes = true_dupes.difference(found_dupes)

    print('found duplicate: %d' % len(found_dupes))

    precision = 1 - len(false_positives) / float(len(found_dupes))
    print('Precision: %0.3f' % precision)

    recall = len(true_positives) / float(len(true_dupes))
    print('Recall: %0.3f' % recall)

    F1 = 2 * precision * recall / (precision + recall)
    print('F1: %0.3f' % F1)

'''
Additional clustering evaluation measures
'''    
def evaluateMeasures(found_list, true_list):

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_list, found_list))
    print("Completeness: %0.3f" % metrics.completeness_score(true_list, found_list))
    print("V-measure: %0.3f" % metrics.v_measure_score(true_list, found_list))
    print("Adjusted Rand Index: %0.3f"
                % metrics.adjusted_rand_score(true_list, found_list))
    print("Adjusted Mutual Information: %0.3f"
                % metrics.adjusted_mutual_info_score(true_list, found_list))
    print("Fowlkes Mallows: %0.3f" % metrics.fowlkes_mallows_score(true_list, found_list))

'''
Read a csv file as a python dictionary
'''    
def dupePairs(filename, rowname, row_id_name) :
    dupe_d = collections.defaultdict(list)
    dupe_l = []

    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            #print(row)
            dupe_d[row[rowname]].append(row[row_id_name])
            if row[rowname] != 'x':
                dupe_l.append(row[rowname])

    if 'x' in dupe_d :
        del dupe_d['x']

    dupe_s = set([])
    for (unique_id, cluster) in viewitems(dupe_d) :
        # print(unique_id, len(cluster), cluster)
        if len(cluster) > 1:
            #print(unique_id, len(cluster), cluster)
            for pair in itertools.combinations(cluster, 2):
                dupe_s.add(frozenset(pair))
               
    return dupe_s, dupe_l
    
if __name__ == '__main__':

    transform_H()
    print(' ')
    '''
    print('Hierarchical Clustering | ', dedupe_clusters_HC)
    true_dupes, true_list = dupePairs(manual_clusters, 'True Id', row_id_name)
    test_dupes, test_list = dupePairs(dedupe_clusters_HC, 'Cluster ID', row_id_name)
    print(len(true_list))
    print(len(test_list))
    assert(len(true_list) == len(test_list))
    evaluateDuplicates(test_dupes, true_dupes)
    evaluateMeasures(test_list, true_list)
    '''
    print(' ')

    print('F-MWSP Clustering | ', dedupe_clusters_FMWSP)
    true_dupes, true_list = dupePairs(manual_clusters, 'True Id', row_id_name)
    test_dupes, test_list = dupePairs(dedupe_clusters_FMWSP, 'Cluster ID', row_id_name)
    print(len(true_list))
    print(len(test_list))
    assert(len(true_list) == len(test_list))
    evaluateDuplicates(test_dupes, true_dupes)
    evaluateMeasures(test_list, true_list)
