from glob import glob
import gzip
from os.path import abspath, exists, isdir, getsize
from sys import argv, exit
from random import randrange
from numpy import array, nan
from os import makedirs
from common import load_properties
from pandas import concat, read_csv, DataFrame
from itertools import product
from datetime import timedelta


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


def get_ens_dim(filename):
    if not exists(filename) or (getsize(filename) == 0):
        return -100
    else:
        with open(filename, 'r') as f:
            content = f.readline()
	    #print(content)
            ens = content.split("::",1)
	    #print ens[1]
            ana = ens[1].split("[", 1)
            #print ana[0]
            ens[1] = ana[0]
            if ",)" not in ens[1]:
                x = (ens[1].strip())[1:-1]
            else:
                x = (ens[1].strip())[1:-2]
            predictors = map(int, x.split(","))
        f.close()
	#print len(predictors)
        return len(list(predictors))


def get_ens_time(filename):
    if not exists(filename) or (getsize(filename) == 0):
        return "00:00:00"
    else:
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line.startswith('time'):
                    time = line.split(" ",1)[1].strip()
        f.close()
        return time


def get_ens_space(filename):
    if not exists(filename) or (getsize(filename) == 0):
        return float('-inf')
    else:
        with open(filename, 'r') as f:
            content = f.readline()
            info = content.split("% (")
            space = float(info[1].split("nodes")[0])
            #print space
        f.close()
        return space

# # #
def get_perf(epsilon, age, conv, exit, strategy, RULE, algo, start):
    filename = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.fmax.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                rl_file = '%s/RL_RESULTS/ORDER%s/RL_bp%i_seed%s_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.fmax' % (project_path, seed, size, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)
                print(rl_file)
                val = get_ens_perf(rl_file)
                if val < 0:
                    val = nan
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(RL VALUES - %s)\t :: %s' % (RULE, filename))


def get_dims(epsilon, age, conv, exit, strategy, RULE, algo, start):
    filename = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.dim.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                dimension = 0.0
                for fold in range(fold_count):
                    rl_file = '%s/RL_OUTPUT/ORDER%s/bp%i_fold%s_seed%s_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.fmax' % (project_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)
                    current = get_ens_dim(rl_file)
                    dimension += current
                dim = dimension / float(fold_count)
                seed_line+=("%s," % dim)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(RL DIMS - %s)\t :: %s' % (RULE, filename))


def get_time(epsilon, age, conv, exit, strategy, RULE, algo, start):
    filename = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.time.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                times = []
                for fold in range(fold_count):
                    rl_file = '%s/RL_OUTPUT/ORDER%s/bp%i_fold%s_seed%s_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.fmax' % (project_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)
                    current = get_ens_time(rl_file)
                    #print current
                    times.append(current)
                t = str(timedelta(seconds=sum(map(lambda f: int(f[0])*3600 + int(f[1])*60 + int(f[2]), map(lambda f: f.split(':'), times)))/len(times))) #AVERAGE
                #print "avg=%s" % t
                t = str(timedelta(seconds=sum(map(lambda f: int(f[0])*3600 + int(f[1])*60 + int(f[2]), map(lambda f: f.split(':'), times)))))
                #print "sum=%s" % t
                #print "\n"
                seed_line+=("%s," % t)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(RL TIME - %s)\t :: %s' % (RULE, filename))


def get_space(epsilon, age, conv, exit, strategy, RULE, algo, start):
    filename = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.space.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start)
    with open(filename, 'w') as f:
        title = "".join(("ens%i, " %i) for i in range(1, (max_num_clsf+1))).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in range(1, (max_num_clsf+1)):
                space = 0
                for fold in range(fold_count):
                    rl_file = '%s/RL_OUTPUT/ORDER%s/bp%i_fold%s_seed%s_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.fmax' % (project_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)
                    current = get_ens_space(rl_file)
                    #print current
                    space += current
                s = space /float(fold_count)
                #print s
                seed_line+=("%s," % s)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('(RL SPACE - %s)\t :: %s' % (RULE, filename))






def resultsRL():
    if not exists("%s/RESULTS/RL/" % project_path):
        makedirs("%s/RESULTS/RL" % project_path)

    for parameters in all_parameters:
        epsilon, age, conv, exit, strategy, RULE, algo, start = parameters
        get_perf(epsilon, age, conv, exit, strategy, RULE, algo, start)
        get_dims(epsilon, age, conv, exit, strategy, RULE, algo, start)
	#get_time(epsilon, age, conv, exit, strategy, RULE, algo, start)
        #get_space(epsilon, age, conv, exit, strategy, RULE, algo, start)

# # #
project_path = "path/RL/%s" % argv[1]
p = load_properties(project_path)
fold_count = int(p['foldCount'])
seeds = int(p['seeds'])


exits            = [0]
algos            = ['Q']
#start_states    = ['0', 'best']
start_states     = ['0']
task             = 10
max_num_clsf     = 180


RULES           = ['L2']
epsilons        = ['0.01', '0.10', '0.25', '0.50']

conv_iters      = [10]
ages            = [0]
strategies      = ['greedyrnd', 'diversityrnd', 'euclideanrnd', 'correlationrnd', 'yulernd', 'kapparnd', 'gsernd'] #['diversityrndv2', 'euclideanrndv2', 'correlationrndv2', 'yulerndv2', 'kapparndv2']
all_parameters1 = list(product(epsilons, ages, conv_iters, exits, strategies, RULES, algos, start_states))

conv_iters       = ['0']
ages             = ['500000']
strategies       = ['pessimeuclid', 'pessimcos', 'pessimcorr', 'pessimyule', 'pessimkappa', 'pessimisticrnd'] #, 'backtrackrnd']
all_parameters2  = list(product(epsilons, ages, conv_iters, exits, strategies, RULES, algos, start_states))


all_parameters = all_parameters1 + all_parameters2
#all_parameters = all_parameters1



print("Starting. . .(#seeds = %s)" % seeds)

if not exists("%s/RESULTS/" % project_path):
    makedirs("%s/RESULTS/" % project_path)

resultsRL()

print("Done!")










