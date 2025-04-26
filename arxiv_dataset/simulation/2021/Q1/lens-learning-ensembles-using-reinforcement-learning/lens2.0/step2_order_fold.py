from os.path import exists, isdir
from os import makedirs
from sys import argv
from glob import glob
import random
import gzip
import pickle
from numpy import array
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score
from common import load_properties, fmax_score
from pandas import concat, read_csv, DataFrame
from collections import OrderedDict

print ("Starting. . .\n")

# ensure project directory exists
project_path    = "path/RL/%s" % argv[1]
assert exists(project_path)

# load and parse project properties
p          	= load_properties(project_path)
fold_count 	= int(p['foldCount'])
seeds 	   	= int(p['seeds'])
bags 	   	= int(p['bags'])
classifierDir 	= p['classifierDir']
metric          = p['metric']
assert (metric in ['fmax', 'auROC'])


for fold in range(fold_count):
    if not exists("%s/ENSEMBLES_FOLD%i/" % (project_path, fold)):
        makedirs("%s/ENSEMBLES_FOLD%i/" % (project_path, fold))


dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))
seed_list = range(seeds) #so that we have 10 repetitions for each experiment
bag_list  = range(bags) #so that we have ten times 18 models
fold_list = range(fold_count)

for fold in fold_list:
	for seed in seed_list:
		dir_dict = {}
		order_fn = '%s/ENSEMBLES_FOLD%i/order_of_seed%i_%s.txt' % (project_path, fold, seed, metric) 
		with open(order_fn, 'w') as order_file:
			for dirname in dirnames:
				print("dirname %s" % dirname)
				for bag in bag_list:
					print("fold = %i seed = %s, bag = %s" % (fold, seed, bag))
					x1 = DataFrame(columns = ["label"])
					x2 = DataFrame(columns = ["prediction"])
	        			#for fold in range(fold_count):
					filename = '%s/valid-b%i-f%s-s%i.csv.gz' % (dirname, bag, fold, seed)
					print(filename)
					df = read_csv(filename, skiprows = 1, compression = 'gzip')
					y_true = df.iloc[:,1:2]
					y_score = df.iloc[:,2:3]
					x1 = df.iloc[:,1:2]
					x2 = df.iloc[:,2:3]
	        			#x1 = concat([x1, y_true], axis = 0)
	        			#x2 = concat([x2, y_score], axis = 0)
					f_max_score = fmax_score(y_true,y_score)
                			#print f_max_score
                			#f1_score = f_score(x1,x2)
                			#print f1_score

					if metric == "fmax":
						dir_dict["%s_bag%i" % (dirname, bag)] = fmax_score(x1,x2)    	
					#if metric == "f1score":
					#	dir_dict["%s_bag%i" % (dirname, bag)] = f_score(x1,x2)
					#if metric == "auROC":
					#	dir_dict ["%s_bag%i" % (dirname, bag)] = roc_auc_score(x1,x2)
					d_sorted_by_value = OrderedDict(sorted(dir_dict.items(), key=lambda x: (-x[1], x[0])))
			for key, v in d_sorted_by_value.items():
			#for key, v in dir_dict.items():
				order_file.write("%s, %s \n" % (key, v))
			order_file.close()
			print(order_fn)

print("\nDone!")






