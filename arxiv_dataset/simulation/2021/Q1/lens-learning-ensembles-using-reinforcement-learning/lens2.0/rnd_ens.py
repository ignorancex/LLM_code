from glob import glob
import gzip
from os.path import abspath, exists, isdir
from sys import argv
from numpy import array, random
from sklearn.metrics import roc_auc_score
from os import makedirs
from utilities import load_properties, fmax_score, get_set_preds
from pandas import concat, read_csv, DataFrame
import math


def read_ens(file_name):
    with open(file_name, 'r') as f:
        content = f.readline()
        ens = content.split("::",1)
        if ",)" not in ens[1]:
            x = (ens[1].strip())[1:-1]
        else:
            x = (ens[1].strip())[1:-2]
        predictors = map(int, x.split(","))
    f.close()
    return predictors

def get_ens_bps(ensemble, filename_fold):
    with open(filename_fold, 'r') as f:
	content = [line.strip() for line in f]
        for line in content:
            if "Base predictors:" in line:
                index = content.index(line)
                break
        dirnames = [bp for bp in content[index+1:index+1+size]] 
        ensemble_bps = [dirnames[bp-1] for bp in ensemble]
    f.close()
    #print "DIRNAMES:"
    #for d in dirnames:
    #    print d
    return ensemble_bps

def aggregate_predictions(predictors, seed, fold, set):
    set = 'test'
    denom = 0
    path, bag, weight = get_path_bag_weight(predictors[0])
    
    denom = ((denom + weight) if RULE == 'WA' else (denom + 1)) 
    y_true, y_score = get_set_preds(path, set, bag, fold, seed)
    y_score = weight * y_score    
    
    for bp in predictors[1:]:
        path, bag, weight = get_path_bag_weight(bp)
        denom  += weight
        y_true, y_score_current = get_set_preds(path, set, bag, fold, seed)
        if RULE == 'WA':
            y_score = y_score.add(weight * y_score_current)        
        else:
            y_score = y_score.add(y_score_current)

    y_score = y_score/denom  
    perf    = fmax_score(y_true, y_score)
    #print perf
    return (y_true, y_score)


def get_path_bag_weight(predictor):
    path = predictor.split(",")[0].split("_bag")[0]
    bag  = int(predictor.split("_bag")[1].split(",")[0])
    weight = float(predictor.split(",")[1])
    return path, bag, weight

def full_ensemble():
    order_file = "%s/ENSEMBLES/order_of_seed%s.txt" % (project_path, seed)
    with open(order_file, 'r') as f:
        content = f.read().splitlines()
    f.close()

    subset = []
    random.seed(int(seed))

    num_bins = 3  # good, medium, weak
    for bin in range(num_bins):
        num_sel  = int(math.floor(size/num_bins) + 1 if (size % num_bins > bin) else int(math.floor(size/num_bins)))
        start    = int(math.floor(len(content)/num_bins) * (bin))
        end      = int(math.floor(len(content)/num_bins) * (bin + 1))
        selected = random.choice(range(start, end), num_sel, replace=False)
        subset.extend([content[bp] for bp in selected])
    random.shuffle(subset)
    #print "SUBSET:"
    #for s in subset:
    #    print s
    #print "\n"
    return subset


def FULL_ens():
    y_true = DataFrame(columns = ["label"])
    y_score = DataFrame(columns = ["prediction"])
    string = ""
    for fold in range(fold_count):
        ensemble_bps = full_ensemble()
        inner_y_true, inner_y_score = aggregate_predictions(ensemble_bps, seed, fold, "test")
        y_true = concat([y_true, inner_y_true], axis = 0)
        y_score = concat([y_score, inner_y_score], axis = 0)
	string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))
	print(length(y_true))
    string += ("final,%f\n" % fmax_score(y_true, y_score))
    filename = '%s/BASELINE/ORDER%i/FE_bp%i_seed%i_%s.fmax' % (results_path, seed, size, seed, RULE)

    with open(filename, 'wb') as f:
    	f.write(string)
    f.close()
    print filename 
    

project_path = "path/RL/%s" % argv[1].replace("/", "")
results_path = "pathRL/%s" % argv[1].replace("/", "")
assert exists(project_path)

p 	   = load_properties(project_path)
fold_count = int(p['foldCount'])
seeds 	   = int(p['seeds'])

size = int(argv[2])
seed = int(argv[3])
RULE = argv[4]
dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

#print "Starting . . ."
if not exists("%s/BASELINE/" % results_path):
    makedirs("%s/BASELINE/" % results_path)

for o in range(seeds):
    if not exists("%s/BASELINE/ORDER%i" % (results_path, o)):
        makedirs("%s/BASELINE/ORDER%i" % (results_path, o))

FULL_ens()
#print "\nDone!"




