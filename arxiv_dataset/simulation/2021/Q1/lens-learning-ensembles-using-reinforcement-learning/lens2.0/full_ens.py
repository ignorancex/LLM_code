from glob import glob
from os.path import abspath, exists, isdir
from sys import argv
from numpy import array, random
from sklearn.metrics import roc_auc_score
from os import makedirs
from pandas import concat, read_csv, DataFrame, to_numeric
import math, gzip
from common import aggregate_predictions, load_properties, fmax_score, get_set_preds, full_ensemble


def FULL_ens():
    y_true = DataFrame(columns = ["label"])
    y_score = DataFrame(columns = ["prediction"])
    string = ""
    for fold in range(fold_count):
        ensemble_bps = full_ensemble(project_path, size, fold, seed, metric)
        inner_y_true, inner_y_score = aggregate_predictions(ensemble_bps, "test", fold, seed, RULE)
        y_true = concat([y_true, inner_y_true], axis = 0)
        y_score = concat([y_score, inner_y_score], axis = 0)
        y_true['label'] = to_numeric(y_true['label'])
        inner_y_score = DataFrame(inner_y_score)
        inner_y_score.rename(columns={0: 'prediction'}, inplace=True)
        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))
    string += ("final,%f\n" % fmax_score(y_true, y_score))
    filename = '%s/BASELINE/ORDER%i/FE_bp%i_seed%i_%s_rnd.fmax' % (project_path, seed, size, seed, RULE)

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
RULE = argv[4]
dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

#print "Starting . . ."
if not exists("%s/BASELINE/" % project_path):
    makedirs("%s/BASELINE/" % project_path)

for o in range(seeds):
    if not exists("%s/BASELINE/ORDER%i" % (project_path, o)):
        makedirs("%s/BASELINE/ORDER%i" % (project_path, o))

FULL_ens()
#print "\nDone!"




