from glob import glob
import gzip
from os.path import abspath, exists, isdir
from sys import argv
from numpy import array, random
from sklearn.metrics import roc_auc_score
from os import makedirs
from common import get_set_preds, fmax_score, load_properties, get_predictor_path_bag_weight, full_ensemble
from pandas import concat, read_csv, DataFrame, to_numeric
import math


def get_max_predictions(predictors, seed, fold, set):
    max_p = ''
    max_w = 0

    path, bag, weight = get_predictor_path_bag_weight(predictors[0])
    if weight > max_w:
        max_w = weight
        max_p = path

    for bp in predictors[1:]:
        path, bag, weight = get_predictor_path_bag_weight(bp)
        if weight > max_w:
            max_w = weight
            max_p = path
    
    y_true, y_score = get_set_preds(max_p, set, bag, fold, seed)
    perf = fmax_score(y_true, y_score)
    return (y_true, y_score)



def BEST_bp():
    y_true = DataFrame(columns = ["label"])
    y_score = DataFrame(columns = ["prediction"])
    string = ""
    for fold in range(fold_count):
        ensemble_bps = full_ensemble(project_path, size, fold, seed, metric)
        inner_y_true, inner_y_score = get_max_predictions(ensemble_bps, seed, fold, "test")
        y_true = concat([y_true, inner_y_true], axis = 0)
        y_true['label'] = to_numeric(y_true['label'])
        y_score = concat([y_score, inner_y_score], axis = 0)
        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))
        print("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))
        print(len(y_true))
    string += ("final,%f\n" % fmax_score(y_true, y_score))
  
    filename = '%s/BASELINE/ORDER%i/BP_bp%i_seed%i_rnd.%s' % (project_path, seed, size, seed, metric)

    with open(filename, 'w') as f:
    	f.write(string)
    f.close()
    print(filename)
    


project_path = "path/RL/%s" % argv[1].replace("/", "")
assert exists(project_path)

p 	   = load_properties(project_path)
fold_count = int(p['foldCount'])
seeds 	   = int(p['seeds'])
metric     = p['metric']
size = int(argv[2])
seed = int(argv[3])
dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

#print "Starting . . ."
if not exists("%s/BASELINE/" % project_path):
    makedirs("%s/BASELINE/" % project_path)

for o in range(seeds):
    if not exists("%s/BASELINE/ORDER%i" % (project_path, o)):
        makedirs("%s/BASELINE/ORDER%i" % (project_path, o))

BEST_bp()
#print "\nDone!"




