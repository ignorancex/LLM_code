import time, gzip, copy, math
from glob import glob
from os.path import abspath, exists, isdir
from os import makedirs
from sys import argv
from numpy import array, argmax, random, count_nonzero
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from common import load_properties, fmax_score, get_set_preds, create_dataset_from_predictions_for_stacker, get_predictor_path_bag_weight, full_ensemble
from pandas import concat, read_csv, DataFrame, to_numeric
from shutil import copyfile
#from xgboost import XGBClassifier


def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    print(lst)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s\n" % (name, coef) for coef, name in lst)

def STACK_run():
    start_time = time.time()
    y_true = DataFrame(columns = ['label'])
    y_score = DataFrame(columns = ['prediction'])
    y_size = 0.0
    string = ""

    for fold in range(fold_count):
        classifiers = full_ensemble(project_path, size, fold, seed, metric) 
        names = []
        for c in classifiers:
            names.append(c)

        inner_y_true_train, inner_train_dataset = create_dataset_from_predictions_for_stacker(classifiers, seed, fold, "valid")
        inner_y_true, inner_test_dataset = create_dataset_from_predictions_for_stacker(classifiers, seed, fold, "test")
	
        "L1Log"
        stacker = LogisticRegression(random_state = 0, penalty = 'l1', solver = 'liblinear')	
        stacker.fit(inner_train_dataset, inner_y_true_train.values.ravel())

        #print("\tstacker.coef_\n\t", stacker.coef_)
        #print("Lasso model: ", pretty_print_linear(stacker.coef_[0], names, sort = False))
        inner_y_size = count_nonzero(stacker.coef_) 
        y_size += inner_y_size

        inner_y_score_ndarray = stacker.predict_proba(inner_test_dataset)[:,1]
        inner_y_score = DataFrame(inner_y_score_ndarray)
        inner_y_score.rename(columns={0: 'prediction'}, inplace=True)
	
        y_true = concat((y_true, inner_y_true), axis=0)
        y_true['label'] = to_numeric(y_true['label'])
        y_score = concat([y_score, inner_y_score], axis = 0)
        string += ("fold_%i,%f::%i\n" % (fold, fmax_score(inner_y_true, inner_y_score), inner_y_size))	

    y_size /= fold_count	
    string += ("final,%f::%f\n" % (fmax_score(y_true, y_score), y_size))    
    
    dst = "%s/STACK_RESULTS/ORDER%i/stack_bp%i_seed%i_%s.fmax" % (project_path, seed, size, seed, "L1Log")
    with open(dst, 'w') as f:
        f.write(string)
    f.close()

    seconds  = time.time() - start_time
    print("\t%s (%s)"  % (dst, (time.strftime('%H:%M:%S', time.gmtime(seconds)))))

   


  

project_path = "path/RL/%s" % argv[1].replace("/", "")
assert exists(project_path)

p = load_properties(project_path)
fold_count = int(p['foldCount'])
metric = p['metric']
size  = int(argv[2])
seed  = int(argv[3])

#print "Starting. . .\n"
if not exists("%s/STACK_RESULTS/" % project_path):
    makedirs("%s/STACK_RESULTS/" % project_path)

if not exists("%s/STACK_RESULTS/ORDER%i" % (project_path, seed)):
    makedirs("%s/STACK_RESULTS/ORDER%i" % (project_path, seed))

STACK_run()
#print "\nDone!"          


