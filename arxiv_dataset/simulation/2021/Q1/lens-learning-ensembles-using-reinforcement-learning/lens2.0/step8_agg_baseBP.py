from glob import glob
import gzip
from os.path import abspath, exists, isdir, getsize
from sys import argv, exit
from random import randrange
from numpy import array
from sklearn.metrics import roc_auc_score
from os import makedirs
from common import load_properties, fmax_score
from pandas import concat, read_csv, DataFrame
from itertools import product


def get_ens_perf(filename):
    #print filename
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

def get_ens_dim(filename):
    if not exists(filename) or (getsize(filename) == 0):
        return -100
    else:
        with open(filename, 'r') as f:
            content = f.readline()
            ens = content.split("::",1)
            if ",)" not in ens[1]:
                x = (ens[1].strip())[1:-1]
            else:
                x = (ens[1].strip())[1:-2]
            predictors = map(int, x.split(","))
        f.close()
        return len(predictors)


# # #
def resultsFE_RULE(RULE):
    filename = '%s/RESULTS/FE/RESULTS_FE_%s_fmax.csv' % (project_path, RULE)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                fe_file = '%s/BASELINE/ORDER%s/FE_bp%i_seed%s_%s.fmax' % (project_path, seed, size, seed, RULE)
                val = get_ens_perf(fe_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(FE VALUES - %s)\t :: %s' % (RULE, filename))

def resultsFE(RULE):
    if not exists("%s/RESULTS/FE/" % project_path):
        makedirs("%s/RESULTS/FE" % project_path)
    resultsFE_RULE(RULE)

# # #
def resultsFE_rnd_RULE(RULE):
    filename = '%s/RESULTS/FE/RESULTS_FE_%s_rnd_fmax.csv' % (project_path, RULE)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                fe_file = '%s/BASELINE/ORDER%s/FE_bp%i_seed%s_%s_rnd.fmax' % (project_path, seed, size, seed, RULE)
                val = get_ens_perf(fe_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(FE_rnd VALUES - %s)\t :: %s' % (RULE, filename))

def dimensionsFE_rnd(RULE):
    if not exists("%s/RESULTS/FE/" % project_path):
        makedirs("%s/RESULTS/FE" % project_path)

    filename = '%s/RESULTS/FE/RESULTS_FE_%s_rnd_dim.csv' % (project_path, RULE)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                fe_file = '%s/BASELINE/ORDER%s/FE_bp%i_seed%s_%s_rnd.fmax' % (project_path, seed, size, seed, RULE)
                val = get_fe_dim(fe_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(FE-%s DIMS)\t :: %s' % (RULE, filename))


def resultsFE_rnd(RULE):
    if not exists("%s/RESULTS/FE/" % project_path):
        makedirs("%s/RESULTS/FE" % project_path)
    resultsFE_rnd_RULE(RULE)
   

def resultsCES_RULE_v(RULE, v):
    filename = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_fmax_%s.csv' % (project_path, RULE, start, v)
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                ces_file = '%s/CES_RESULTS/ORDER%s/CES_bp%i_seed%s_%s_start-%s_%s.fmax' % (project_path, seed, size, seed, RULE, start, v)
                val = get_ens_perf(ces_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(CES VALUES %s - %s)\t :: %s' % (v, RULE, filename))


def get_dims_v(RULE, v):
    filename = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_dim_%s.csv' % (project_path, RULE, start, v)    
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                dimension = 0.0
                for fold in range(fold_count):
                    ces_file = '%s/CES_OUTPUT/ORDER%s/bp%i_fold%i_seed%s_%s_start-%s_%s.fmax' % (project_path, seed, size, fold, seed, RULE, start, v)
                    current = get_ens_dim(ces_file)
                    dimension += current
                dim = dimension / float(fold_count)
                seed_line+=("%s," % dim)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(CES DIMS %s - %s)\t :: %s' % (v, RULE, filename))

def resultsCES_v(RULE, v):
    if not exists("%s/RESULTS/CES/" % project_path):
        makedirs("%s/RESULTS/CES" % project_path)
    resultsCES_RULE_v(RULE, v)

# # #

def resultsCES_RULE(RULE):
    filename = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_fmax.csv' % (project_path, RULE, start)
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                ces_file = '%s/CES_RESULTS/ORDER%s/CES_bp%i_seed%s_%s_start-%s.fmax' % (project_path, seed, size, seed, RULE, start)
                val = get_ens_perf(ces_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(CES VALUES - %s)\t :: %s' % (RULE, filename))


def get_dims(RULE):
    filename = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_dim.csv' % (project_path, RULE, start)
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                dimension = 0.0
                for fold in range(fold_count):
                    ces_file = '%s/CES_OUTPUT/ORDER%s/bp%i_fold%i_seed%s_%s_start-%s.fmax' % (project_path, seed, size, fold, seed, RULE, start)
                    current = get_ens_dim(ces_file)
                    dimension += current
                dim = dimension / float(fold_count)
                seed_line+=("%s," % dim)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(CES DIMS - %s)\t :: %s' % (RULE, filename))

def resultsCES(RULE):
    if not exists("%s/RESULTS/CES/" % project_path):
        makedirs("%s/RESULTS/CES" % project_path)
    resultsCES_RULE(RULE)
# # #

def resultsBP():
    if not exists("%s/RESULTS/BP/" % project_path):
        makedirs("%s/RESULTS/BP" % project_path)

    filename = '%s/RESULTS/BP/RESULTS_BP_fmax.csv' % (project_path)
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                bp_file = '%s/BASELINE/ORDER%s/BP_bp%i_seed%s.fmax' % (project_path, seed, size, seed)
                val = get_ens_perf(bp_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(BEST PREDICTOR VALUES)\t :: %s' % (filename))

# # #

def resultsSTACK():
    if not exists("%s/RESULTS/STACK/" % project_path):
        makedirs("%s/RESULTS/STACK" % project_path)

    #path/RL/drosophila/STACK_RESULTS/ORDER0/stack_bp100_seed0_L1Log.fmax
    filename = '%s/RESULTS/STACK/RESULTS_STACK_fmax.csv' % (project_path)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                stack_file = '%s/STACK_RESULTS/ORDER%s/stack_bp%i_seed%s_L1Log.fmax' % (project_path, seed, size, seed)
                val = get_stack_perf(stack_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(STACK VALUES)\t :: %s' % (filename))



def dimensionsSTACK():
    if not exists("%s/RESULTS/STACK/" % project_path):
        makedirs("%s/RESULTS/STACK" % project_path)

    #path/RL/drosophila/STACK_RESULTS/ORDER0/stack_bp100_seed0_L1Log.fmax
    filename = '%s/RESULTS/STACK/RESULTS_STACK_dim.csv' % (project_path)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                stack_file = '%s/STACK_RESULTS/ORDER%s/stack_bp%i_seed%s_L1Log.fmax' % (project_path, seed, size, seed)
                val = get_stack_dim(stack_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(STACK DIMS)\t :: %s' % (filename))


def get_stack_perf(filename):
    #print filename
    if not exists(filename) or (getsize(filename) == 0):
         return -100
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        if (content[len(content)-1].split(",")[0] == "final"):
            return float(content[len(content)-1].split(",")[1].split("::")[0])
        else:
            return -100


def get_fe_dim(filename):
    print(filename)
    if not exists(filename) or (getsize(filename) == 0):
         return -100
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        print(content)
        if (content[len(content)-1].split(",")[0] == "final"):
            return float(content[len(content)-1].split(",")[1].split("::")[1])
        else:
            return -100


def get_stack_dim(filename):
    #print filename
    if not exists(filename) or (getsize(filename) == 0):
         return -100
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        if (content[len(content)-1].split(",")[0] == "final"):
            return float(content[len(content)-1].split(",")[1].split("::")[1])
        else:
            return -100
# # #


def resultsBP_rnd():
    if not exists("%s/RESULTS/BP/" % project_path):
        makedirs("%s/RESULTS/BP" % project_path)

    filename = '%s/RESULTS/BP/RESULTS_BP_rnd_fmax.csv' % (project_path)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                bp_file = '%s/BASELINE/ORDER%s/BP_bp%i_seed%s_rnd.fmax' % (project_path, seed, size, seed)
                val = get_ens_perf(bp_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(BEST PREDICTOR VALUES)\t :: %s' % (filename))

# # #

project_path = "path/RL/%s" % argv[1]
p = load_properties(project_path)
fold_count   = int(p['foldCount'])
seeds        = int(p['seeds'])
RULES        = ['L1', 'L2']
start        = '1'
max_num_clsf = 180


print("Starting. . .(#seeds = %s)" % seeds)

if not exists("%s/RESULTS/" % project_path):
    makedirs("%s/RESULTS/" % project_path)

#resultsBP_rnd()
for rule in RULES:
    #resultsFE_rnd(rule)
    #dimensionsFE_rnd(rule)
    
    resultsSTACK()
    dimensionsSTACK()

print ("Done!")

