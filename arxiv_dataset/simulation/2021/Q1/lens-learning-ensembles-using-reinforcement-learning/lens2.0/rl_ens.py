from glob import glob
import gzip
from os.path import abspath, exists, isdir
from sys import argv
from numpy import array
from sklearn.metrics import roc_auc_score
from os import makedirs
from common import load_properties, fmax_score, get_set_preds, aggregate_predictions
from pandas import concat, read_csv, DataFrame, to_numeric


def read_ens(file_name):
    with open(file_name, 'r') as f:
        content = f.readline()
	#print(file_name)
	#print (content)
        ens = (content.split("::",1)[1]).split("[",1)[0] 
        if ",)" not in ens:
            x = (ens.strip())[1:-1]
        else:
            x = (ens.strip())[1:-2]
        predictors = map(int, x.split(","))
    f.close()
    return predictors

def get_ens_bps(ensemble, filename_fold):
    with open(filename_fold, 'r') as f:
        content = [line.strip() for line in f]
        for line in content:
            if "Base predictors" in line:
                index = content.index(line)
                break
        dirnames = [bp for bp in content[index+1:index+1+size]] 
        ensemble_bps = [dirnames[bp-1] for bp in ensemble]
    f.close()
    return ensemble_bps

def RL_ens():
    y_true = DataFrame(columns = ["label"])
    y_score = DataFrame(columns = ["prediction"])
    string = ""
    for fold in range(fold_count):
        filename_fold = '%s/RL_OUTPUT/ORDER%s/bp%s_fold%s_seed%s_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.fmax' % (project_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)
        ensemble = read_ens(filename_fold)
        ensemble_bps = get_ens_bps(ensemble, filename_fold)
        inner_y_true, inner_y_score = aggregate_predictions(ensemble_bps, set, fold, seed, RULE)
        y_true = concat([y_true, inner_y_true], axis = 0)
        y_true['label'] = to_numeric(y_true['label'])
        y_score = concat([y_score, inner_y_score], axis = 0)

        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))

    string += ("final,%f\n" % fmax_score(y_true, y_score))
    filename = '%s/RL_RESULTS/ORDER%i/RL_bp%i_seed%i_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.fmax' % (project_path, seed, size, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)

    with open(filename, 'w+') as f:
    	f.write(string)
    f.close()
    print(filename)
    


project_path = "path/RL/%s" % argv[1].replace("/", "")
assert exists(project_path)

p 	     = load_properties(project_path)
fold_count   = int(p['foldCount'])
seeds 	     = int(p['seeds'])

size	  = int(argv[2])
age 	  = argv[3]
seed      = int(argv[4])
epsilon   = argv[5]
RULE 	  = argv[6]
strategy  = argv[7]
start 	  = argv[8]
exit 	  = argv[9]
conv 	  = argv[10]
algo 	  = argv[11]
dirnames  = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

#print "Starting . . ."
if not exists("%s/RL_RESULTS/" % project_path):
    makedirs("%s/RL_RESULTS/" % project_path)

for o in range(seeds):
    if not exists("%s/RL_RESULTS/ORDER%i" % (project_path, o)):
        makedirs("%s/RL_RESULTS/ORDER%i" % (project_path, o))
RL_ens()
#print "\nDone!"




