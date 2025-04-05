from future.utils import viewitems

import csv
import collections
import itertools
from sklearn import metrics

exp = 'results_2/'
manual_clusters = exp + 'VAL_music20k_golden.csv'
dedupe_clusters = exp + 'VAL_F-MWSP_music20k_output.csv'
row_id_name = 'TID'

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

def evaluateMeasures(found_list, true_list):

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_list, found_list))
    print("Completeness: %0.3f" % metrics.completeness_score(true_list, found_list))
    print("V-measure: %0.3f" % metrics.v_measure_score(true_list, found_list))
    print("Adjusted Rand Index: %0.3f"
                % metrics.adjusted_rand_score(true_list, found_list))
    print("Adjusted Mutual Information: %0.3f"
                % metrics.adjusted_mutual_info_score(true_list, found_list))
    print("Fowlkes Mallows: %0.3f" % metrics.fowlkes_mallows_score(true_list, found_list))
    
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

print('WARNING! Check the row_id_name')
print(row_id_name)

print(dedupe_clusters)

true_dupes, true_list = dupePairs(manual_clusters, 'True Id', row_id_name)
test_dupes, test_list = dupePairs(dedupe_clusters, 'Cluster ID', row_id_name)

#print('test_dupes')
#print(test_dupes)
print(len(true_list))
print(len(test_list))
assert(len(true_list) == len(test_list))

evaluateDuplicates(test_dupes, true_dupes)
evaluateMeasures(test_list, true_list)


