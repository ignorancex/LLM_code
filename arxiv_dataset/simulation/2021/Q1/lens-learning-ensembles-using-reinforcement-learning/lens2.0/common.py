"""
    datasink: A Pipeline for Large-Scale Heterogeneous Ensemble Learning
    Copyright (C) 2013 Sean Whalen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
"""

from numpy import argmax, argmin, argsort, corrcoef, mean, nanmax, sqrt, triu_indices_from, where, random, seterr, count_nonzero
import numpy.testing as npt
from numpy.random import choice
from pandas import DataFrame, concat, read_csv, Series
from scipy.io.arff import loadarff
import sklearn.metrics
from os.path import exists, getsize
from itertools import chain, combinations
import pickle
from sklearn.linear_model import LogisticRegression

def argsortbest(x):
    return argsort(x) if greater_is_better else argsort(x)[::-1]

def average_pearson_score(x):
    if isinstance(x, DataFrame):
        x = x.values
    rho = corrcoef(x, rowvar = 0)
    return mean(abs(rho[triu_indices_from(rho, 1)]))

def get_best_performer(df, one_se = False):
    if not one_se:
        return df[df.score == best(df.score)].head(1)
    se = df.score.std() / sqrt(df.shape[0] - 1)
    if greater_is_better:
        return df[df.score >= best(df.score) - se].head(1)
    return df[df.score <= best(df.score) + se].head(1)

def confusion_matrix_fpr(labels, predictions, false_discovery_rate = 0.1):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions)
    max_fpr_index = where(fpr >= false_discovery_rate)[0][0]
    #print sklearn.metrics.confusion_matrix(labels, predictions > thresholds[max_fpr_index])

def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    seterr(divide = 'ignore', invalid = 'ignore') #to supress the warning
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return nanmax(f1)

def load_arff(filename):
    return DataFrame.from_records(loadarff(filename)[0])

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

def load_properties(dirname):
    properties = [_.split('=') for _ in open(dirname + '/config.txt').readlines()]
    d = {}
    for key, value in properties:
        d[key.strip()] = value.strip()
    return d

def rmse_score(a, b):
    return sqrt(mean((a - b)**2))

def unbag(df, bag_count):
    cols = []
    bag_start_indices = range(0, df.shape[1], bag_count)
    names = [_.split('.')[0] for _ in df.columns.values[bag_start_indices]]
    for i in bag_start_indices:
        cols.append(df.ix[:, i:i+bag_count].mean(axis = 1))
    df = concat(cols, axis = 1)
    df.columns = names
    return df

def get_set_probs(dirname, seed, fold, set):
    filename = '%s/%s-pred-f%s-s%i.csv.gz' % (dirname, set, fold, seed)
    df = read_csv(filename, skiprows = 1, compression = 'gzip')
    y_score = df.ix[:,2:3]
    return y_score

def get_ens_bps(ensemble, order_file):
    with open(order_file, 'r') as f:
        content = f.read().splitlines()
        dirnames = [content[bp-1] for bp in ensemble]
    f.close()
    return dirnames

def get_ens_dim(fileName):
    if not exists(fileName) or (getsize(fileName) == 0):
        return -100
    else:
        with open(fileName, 'r') as f:
            content = f.readline()
            ens = content.split("::", 1)
            if ",)" not in ens[1]:
                x = (ens[1].strip())[1:-1]
            else:
                x = (ens[1].strip())[1:-2]
            predictors = map(int, x.split(","))
        f.close()
        return len(predictors)


def get_max_test(fileName):
    if not exists(fileName) or (getsize(fileName) == 0):
        return -100
    else:
        with open(fileName, 'r') as f:
            content = f.readline()
            info = content.split("maxTest=", 1)
            max_test = float(info[1].split(",", 1)[0])
        f.close()
        return max_test


def get_rl_test(fileName):
    if not exists(fileName) or (getsize(fileName) == 0):
        return -100
    else:
        with open(fileName, 'r') as f:
            content = f.readline()
            info = content.split("test=", 1)
            rl_test = float(info[1].split(",", 1)[0])
        f.close()
        return rl_test



def get_max_test_dim(fileName):
    if not exists(fileName) or (getsize(fileName) == 0):
        return -100
    else:
        with open(fileName, 'r') as f:
            content = f.readline()
            info = content.split("maxEns=", 1)
            ens = info[1].split("::", 1)
            if ",)" not in ens[0]:
                x = (ens[0].strip())[1:-2]
            else:
                x = (ens[0].strip())[1:-3]
            predictors = map(int, x.split(","))
        f.close()
        return len(predictors)


def get_gold_ens(fileName):
    max_ens = ""
    max_val = ""
    with open(fileName, 'r') as f:
        for line in f:
            content = line.split(":")
            current_ens = content[0]
            test_val = content[2] #this is the test value!
            if test_val > max_val:
                max_val = test_val
                max_ens = current_ens
    f.close()
    predictors = map(int, max_ens.split(","))
    return predictors

def get_ens_age(fileName):
    if not exists(fileName) or (getsize(fileName) == 0):
        return -100
    else:
        with open(fileName, 'r') as f:
            content = f.readline()
            ens = content.split("age=", 1)
            age = int(ens[1].split(" ", 1)[0])
        f.close()
        return age

def average(bps, seed, fold, set):
    y_true, y_score = get_set_preds(bps[0], seed, fold, set)
    for bp in bps[1:]:
        y_true, y_score_current = get_set_preds(bp, seed, fold, set)
        y_score = y_score.add(y_score_current)
    y_score = y_score.apply(lambda x: x/(len(bps)))
    return (y_true, y_score)

def normalize(d, target = 1.0):
   raw = sum(d.values())
   factor = target/raw
   return {key:value*factor for key,value in d.items()}

def get_normalized_weights(bps, seed, fold):
    weights_dict = {}
    for bp in bps:
        #path/RL/apply/weka.classifiers.functions.Logistic_bag7, 0.1818331197213185
        weight = bp.split(", ")[1]
        weights_dict[bp] = float(weight)
    return normalize(weights_dict)

def get_weights(bps, seed, fold):
    weight = {}
    for bp in bps:
        weight[bp] = get_weight(bp, seed, fold)
    return weight

def get_weight(bp, seed, fold):
    validation_file = '%s/validation-s%i.csv' % (bp, seed)
    weight = get_val_perf(validation_file, fold)
    return weight

def get_val_perf(filename, cvFold):
    if not exists(filename) or (getsize(filename) == 0):
         return "FNF"
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        if (content[cvFold+1].split(",")[0] == str(cvFold)):
            return float(content[cvFold+1].split(",")[2])
        else:
            #print "Not all folds present in the file!"
            return "FNF"

def get_ens_denom_wa(bps, seed, fold):
    weights = get_weights(bps, seed, fold)
    return sum(weights.values())

def get_bag_from_base_predictor(bp):
    bag = (bp.split("_bag")[1]).split(", ")[0]
    return int(bag)

def get_path_from_bps(bps):
    path = bps.split("_bag")[0]
    return path

def get_set_preds(path, set, bag, fold, seed):
    filename = '%s/%s-b%i-f%i-s%i.csv.gz' % (path, set, bag, fold, seed)
    df = read_csv(filename, skiprows = 1, compression = 'gzip')
    y_true = df.iloc[:,1:2]
    y_score = df.iloc[:,2:3]
    return y_true, y_score 

def weighted_average(bps, seed, fold, set):
    weights = get_normalized_weights(bps, seed, fold)
    path_0 = get_path_from_bps(bps[0])
    bag_0 = get_bag_from_base_predictor(bps[0])
    y_true, y_score = get_set_preds(path_0, set, bag_0, fold, seed)
    y_score = weights[bps[0]] * y_score
    for bp in bps[1:]:
        path_i = get_path_from_bps(bp)
        bag_i = get_bag_from_base_predictor(bp)
        y_true, y_score_current = get_set_preds(path_i, set, bag_i, fold, seed)
        y_score += weights[bp] * y_score_current
    return (y_true, y_score)

def create_dataset_from_predictions_for_stacker(predictors, seed, fold, set):
    path, bag, weight = get_predictor_path_bag_weight(predictors[0])
    y_labels, y_predictions = get_set_preds(path, set, bag, fold, seed)
    i = 1
    y_predictions.rename(columns={'prediction': ("bp_%i" % i)}, inplace=True)
    for bp in predictors[1:]:
        i += 1
        path, bag, weight = get_predictor_path_bag_weight(bp)
        y_labels, y_predictions_current = get_set_preds(path, set, bag, fold, seed)
        y_predictions_current.rename(columns={'prediction': "bp_%i" % i}, inplace=True)
        y_predictions = concat([y_predictions, y_predictions_current], axis = 1)
        #y_predictions = y_predictions.join(y_predictions_current) #does the same thing as previous line
    return (y_labels, y_predictions)

def get_predictor_path_bag_weight(predictor): 
    # path/RL/pf1/weka.classifiers.meta.AdaBoostM1_bag8, 0.195804195804 
    # 1 :: path/RL/apply/weka.classifiers.functions.Logistic/valid-b7-f0-s0.csv.gz
    if " :: " in predictor:
        path = (predictor.split("_bag")[0]).split(":: ")[1]
    else:
        path = predictor.split("_bag")[0]
    bag  = int((predictor.split("_bag")[1]).split(",")[0])
    weight = float(predictor.split(",")[1])
    return path, bag, weight


def full_ensemble(project_path, size, fold, seed, metric):
    predictors = {}
    bps_weight = {}

    order_file = "%s/ENSEMBLES_FOLD%i/order_of_seed%s_%s.txt" % (project_path, fold, seed, metric)
    with open(order_file, 'r') as f:
        content = f.read().splitlines()
    f.close()

    subset = []
    random.seed(int(seed)) #this is needed to ensure that bp10 \in bp20 \in bp30 etc.
    selected = random.choice(len(content), size, replace = False)
    subset.extend([content[bp] for bp in selected])

    #print("SUBSET:")
    #for s in subset:
    #    print(s)
    #print("\n")

    for i in range(len(subset)):
        index = i + 1
        predictors[index] = subset[i]
        bps_weight[index] = float(subset[i].split(",")[1])
    return subset



def l2reg(bps, seed, fold, set):
    train_labels, train_dataset = create_dataset_from_predictions_for_stacker(bps, seed, fold, "valid")
    test_labels,   test_dataset = create_dataset_from_predictions_for_stacker(bps, seed, fold, "test")
    stacker = LogisticRegression(random_state = 0, penalty = 'l2', solver = 'liblinear')
    stacker.fit(train_dataset, train_labels.values.ravel())

    inner_y_size = count_nonzero(stacker.coef_)
    #print(inner_y_size)


    test_predictions = DataFrame(stacker.predict_proba(test_dataset)[:,1])
    test_predictions.rename(columns={0: 'prediction'}, inplace = True)
    return (test_labels, test_predictions)

def product(bps, seed, fold, set):
    y_true, y_score = get_set_preds(bps[0], seed, fold, set)
    for bp in bps[1:]:
        y_true, y_score_current = get_set_preds(bp, seed, fold, set)
        y_score = y_score.mul(y_score_current)
    return (y_true, y_score)

def prod_one_minus_prob(bps, seed, fold, set):
    y_true, y_score = get_set_preds(bps[0], seed, fold, set)
    y_score = 1 - y_score
    for bp in bps[1:]:
        y_true, y_score_current = get_set_preds(bp, seed, fold, set)
        y_score_current = 1 - y_score_current
        y_score = y_score.mul(y_score_current)
    return (y_true, y_score)

def contrast(bps, seed, fold, set):
    x1, y1 = product(bps, seed, fold, set)
    x2, y2 = prod_one_minus_prob(bps, seed, fold, set)
    npt.assert_array_equal(x1, x2)
    y_true = x1
    y_score = y1 - y2
    return (y_true, y_score)

def aggregate_predictions(bps, set, fold, seed, RULE):
    if RULE == "L2":
        return l2reg(bps, seed, fold, set)
    elif RULE == "PROD":
        return product(bps, seed, fold, set)
    elif RULE == "AVG":
        return average(bps, seed, fold, set)
    elif RULE == "CON":
        return contrast(bps, seed, fold, set)
    elif RULE == "WA":
        return weighted_average(bps, seed, fold, set)
    else: #default RULE = "AVG"
        return average(bps, seed, fold, set)


def get_ens_perf(filename):
    if not exists(filename) or (getsize(filename) == 0):
         return -100
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        if (content[len(content)-1].split(",")[0] == "final"):
            return float(content[len(content)-1].split(",")[1])
        else:
            return -100

def get_ens_perf_naive(filename, fold_count):
    if not exists(filename) or (getsize(filename) == 0):
         return -100
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        performance = 0
        for fold in range(fold_count):  
            performance += float(content[fold].split(',')[1])
        performance /= fold_count    
        return performance

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def rev_powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in reversed(range(len(s)+1))))[1:-1]

def powerset_extd(iterable, sz):
    classifiers = list(iterable)
    existing = [classifiers[c] for c in range(sz)]
    nodes = [e for e in list(powerset(classifiers)) if e not in list(powerset(existing))]
    ##print "\t* left to compute: %i nodes (ensembles)" % len(nodes)
    return nodes


def get_ordered_bps(order_file):
    with open(order_file, 'r') as f:
        content = f.read().splitlines()
    f.close()
    return content

def get_partial_set_bps(path, order, size, part, first, last):
    order_file = '%s/ENSEMBLES/order%s.txt' % (path, order)
    dirnames = get_ordered_bps(order_file)
    classifiers = dirnames[:size]
    return (itertools.islice(powerset(classifiers), first, last+1)) # itertools.islice(generator, from(incl.), to(excl.)) 


def get_bp_perf(filename):
    if not exists(filename) or (getsize(filename) == 0):
         return -100
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        if (content[len(content)-1].split(",")[0] == "score"):
            return float(content[len(content)-1].split(",")[2])
        else:
            return -100

def get_fold_value(filename, fold):
    if not exists(filename) or (getsize(filename) == 0):
        #print "File does not exist or is empty!"
        exit(1)
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        row = fold + 1 #because we have a header
        return float(content[row].split(",")[2])

def get_global_id(path, bps):
    global_file = '%s/partial/global_order' % path 
    with open(global_file, 'rb') as f:
        glob_ord = pickle.load(f)
    ens = list('0' * len(glob_ord))    
    for bp in bps:
        pos = glob_ord[bp]
        ens[pos-1] = "1" #[pos-1] because array starts at zero
    id = "".join(ens)
    return "%s" % id
   
def get_classifier_with_global_id(path, lst):
    global_file = '%s/partial/global_order' % path
    with open(global_file, 'rb') as f:
        glob_ord = pickle.load(f)
    x = []
    for p in lst:
        for bp, pos in glob_ord.items():
            if pos == p+1: #because glob_ord considers positions > 0
                x.append(bp)
                break
    return x



def exists_largest_sub_ens(path, bps, fold, seed, RULE):
    #print bps
    id = get_global_id(path, bps)
    ens = list(id)
    ans = False
    sid = ""
    extra = []
    nonzeroind = [i for i, e in enumerate(ens) if e != '0']
    lst = rev_powerset(nonzeroind)
    for e in lst:
        sub_ens_id = list('0' * 18)
        mask = list('0' * 18)
        for index in e:
            sub_ens_id[index] = '1'
        sid = "".join(sub_ens_id)
        ens_file_numer = '%s/partial/%s/valid%s_f%s_s%s.numerator_%s.txt' % (path, RULE, sid, fold, seed, RULE)
        if exists(ens_file_numer):
            ans = True
            #print "\t\t\t---------------------> %s  exists" % (",".join([str(elem) for elem in e]))
            #print "\t\t\t%r " % nonzeroind
            #print "\t\t\t---------------------> extra = %s" % (",".join([str(elem) for elem in nonzeroind if elem not in e]))
            rest_of_clsf = [elem for elem in nonzeroind if elem not in e]
            #print "\t\t\t%r" % rest_of_clsf
            
            extra = get_classifier_with_global_id(path, rest_of_clsf)
            
            if len(rest_of_clsf) > 1:
                #print "the long way - must implement the fast way (common.py)!!!"
                ans = False
            break
    return ans, sid, extra


def get_ens_numerator_wa(path, id, seed, fold, RULE):
    set = "valid"
    filename = '%s/partial/%s/%s%s_f%s_s%s.numerator_%s.txt' % (path, RULE, set, id, fold, seed, RULE)
    #print "\t\t\tgetting_ens_numerator_wa :: %s" % filename
    if exists(filename):
        df = read_csv(filename)
        y_true = df.ix[:,0:1]
        y_score = df.ix[:,1:2]
    else:
        print("*ERROR: Partial file (numerator) %s does not exist!" % sid)
    return (y_true, y_score)

def compute_ens_numerator_wa(bps, path, seed, fold, RULE):
    weights = get_weights(bps, seed, fold)
    id = get_global_id(path, bps)
    set = "valid"
    y_true, y_score = get_set_preds(bps[0], seed, fold, set)
    y_score = weights[bps[0]] * y_score
    for bp in bps[1:]:
        y_true, y_score_current = get_set_preds(bp, seed, fold, set)
        y_score = y_score.add(weights[bp] * y_score_current)
    val_score = fmax_score(y_true, y_score)

    df = DataFrame()
    df = df.append(y_true)
    df.insert(1, 'prediction', y_score)
    filename = '%s/partial/%s/%s%s_f%s_s%s.numerator_%s.txt' % (path, RULE, set, id, fold, seed, RULE)
    df.to_csv(filename, index=False) 

    return val_score


# # # # # # #

diversity_score = average_pearson_score
score = sklearn.metrics.roc_auc_score
