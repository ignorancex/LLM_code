import sklearn.metrics
from numpy import random, nanmax
from pandas import concat, read_csv, DataFrame
import math
import utilities

def load_properties(dirname):
    properties = [_.split('=') for _ in open(dirname + '/config.txt').readlines()]
    d = {}
    for key, value in properties:
        d[key.strip()] = value.strip()
    return d

def cluster_cmd(): 
    return 'rc.py --cores 1 --walltime 00:10 --queue low'

def get_num_cores():
    #return -1 #spawning processes on all the available cores of the local machine
    return 1

def get_fold_ens(fileName):
    with open(fileName, 'r') as f:
        content = f.readline()
        ens = content.split(':: (',1)[1].split(')')[0]
        ens = (ens[:-1] if ens[-1] == ',' else ens)
        predictors = map(int, ens.split(","))
    f.close()
    return predictors

def load_arff_headers(filename):
    dtypes = {}
    for line in open(filename):
        if line.startswith('@data'):
            break
        if line.startswith('@attribute'):
            _, name, dtype = line.split()
            if dtype.startswith('{'):
                dtype = dtype[1:-1]
            dtypes[name] = set(dtype.split(','))
    return dtypes


def f_score(labels, predictions, beta = 1.0, pos_label = 1):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return f1


def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return nanmax(f1)


def get_set_preds(dirname, set, bag, fold, seed):
    filename = '%s/%s-b%s-f%s-s%s.csv.gz' % (dirname, set, bag, fold, seed)
    df  = read_csv(filename, skiprows = 1, compression = 'gzip')
    y_true = df.ix[:,1:2]
    y_score = df.ix[:,2:3]
    return y_true, y_score 


def get_bps_rnd_bins(project_path, seed, metric, size, fold):
    order_file = "%s/ENSEMBLES/order_of_seed%s_%s.txt" % (project_path, seed, metric)
    with open(order_file, 'r') as f:
        content = f.read().splitlines()
    f.close()

    max_num_clsf = len(content)
    bps_weight = {}
    subset = []
    random.seed(int(seed))

    num_bins = 3  # good, medium, weak
    interval = int(math.floor(max_num_clsf/num_bins))
    rem = max_num_clsf%num_bins

    for bin in range(num_bins):
        num_sel  = int(math.floor(size/num_bins) + 1 if (size % num_bins > bin) else int(math.floor(size/num_bins)))
        if bin == 0:
            start = 0
            end   = (interval + 1 if rem else interval)
        elif bin == 1:
            start = (interval + 1 if rem else interval)
            end   = (2*interval + rem if rem else 2*interval)
        else:
            start = (2*interval + rem if rem else 2*interval)
            end   = max_num_clsf

        selected = random.choice(range(start, end), num_sel, replace=False)
        subset.extend([content[bp] for bp in selected])

    for i in range(len(subset)):
        index = i + 1
        bps_weight[index] = float(subset[i].split(",")[1])

    return subset, bps_weight




def get_bps(project_path, seed, metric, size): #to vary the ensemble size, select randomly w/o replacement, an equal number of good, medium, and weak performance bps
    order_file = "%s/ENSEMBLES/order_of_seed%s_%s.txt" % (project_path, seed, metric)
    with open(order_file, 'r') as f:
        content = f.read().splitlines()
    f.close()

    max_num_clsf = len(content)    
    bps_weight = {}
    subset = []
    random.seed(int(seed))

    num_bins = 3  # good, medium, weak
    interval = int(math.floor(max_num_clsf/num_bins))
    rem = max_num_clsf%num_bins

    for bin in range(num_bins):
        num_sel  = int(math.floor(size/num_bins) + 1 if (size % num_bins > bin) else int(math.floor(size/num_bins)))
        if bin == 0:
            start = 0
            end   = (interval + 1 if rem else interval)
        elif bin == 1:
            start = (interval + 1 if rem else interval)
            end   = (2*interval + rem if rem else 2*interval)
        else:
            start = (2*interval + rem if rem else 2*interval)
            end   = max_num_clsf

        selected = random.choice(range(start, end), num_sel, replace=False)
        subset.extend([content[bp] for bp in selected])

    for i in range(len(subset)):
        index = i + 1
        bps_weight[index] = float(subset[i].split(",")[1])

    return subset, bps_weight

def get_bps_rnd_and_weights(project_path, seed, metric, size, fold):
    order_file = "%s/ENSEMBLES_FOLD%i/order_of_seed%s_%s.txt" % (project_path, fold, seed, metric)
    with open(order_file, 'r') as f:
        content = f.read().splitlines()
    f.close()

    subset = []
    random.seed(int(seed)) #this is needed to ensure that bp10 \in bp20 \in bp30 etc.
    selected = random.choice(len(content), size, replace = False)
    subset.extend([content[bp] for bp in selected])

    bps_weight = {}
    for i in range(len(subset)):
        index = i + 1
        bps_weight[index] = float(subset[i].split(",")[1])


    #print("SUBSET:")
    #for s in subset:
    #    print(s)
    #print("\n")

    return subset, bps_weight

def get_stacker_size(project_path, size, seed, fold):
    #path/RL/apply/STACK_RESULTS/ORDER0/stack_bp10_seed0_L1Log.fmax
    filename = "%s/STACK_RESULTS/ORDER%i/stack_bp%i_seed%i_L1Log.fmax" % (project_path, seed, size, seed)
    curr_fold = -1
    enforced_size = -1
    with open(filename, 'r') as f:
        content = f.readlines()
        for line in content:
            curr_fold += 1     
            if curr_fold == fold:
                enforced_size = line.split("::")[1]
    f.close()
    return int(enforced_size)
    
def bps2string(predictors):
    str = "\n* * * * *\nBase predictors and their weight (performance on the validation, over all folds):\n"
    index = 1
    for p in predictors:
        str += "%i :: %s\n" % (index, p)
        index += 1
    str += "* * * * *\n"
    return str


def get_path(predictor):
    path = (predictor.split("/weka")[0])
    return path

def get_path_bag_weight(predictor):
    path = (predictor.split("/")[0].split(":: ")[1] if "::" in predictor else predictor.split("_bag")[0])
    bag  = int(predictor.split("_bag")[1].split(",")[0])
    weight = float(predictor.split(",")[1].strip())
    return path, bag, weight

#def aggregate_predictions(predictors, seed, fold, set, RULE):
#    denom = 0
#    path, bag, weight = get_path_bag_weight(predictors[0])
#
#    denom = ((denom + weight) if RULE == 'WA' else (denom + 1))
#    y_true, y_score = get_set_preds(path, set, bag, fold, seed)
#    y_score = weight * y_score
#
#    for bp in predictors[1:]:
#        path, bag, weight = get_path_bag_weight(bp)
#        denom  += weight
#        y_true, y_score_current = get_set_preds(path, set, bag, fold, seed)
#        y_score = (y_score.add(weight * y_score_current) if RULE =='WA' else y_score.add(y_score_current))
#
#    y_score = y_score/denom
#    perf    = fmax_score(y_true, y_score)
#    return (y_true, y_score)



