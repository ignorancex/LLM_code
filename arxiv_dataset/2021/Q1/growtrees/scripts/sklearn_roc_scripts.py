
import sys
import os
import argparse

import random
import json
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

import pickle
import numpy as np

from sklearn import metrics 
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Load xgboost model to predict test data.')
parser.add_argument('dataset', type=str)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--train_method', type=str, default="greedy")
parser.add_argument('--num_attacks', type=int, default=5000)
parser.add_argument('--num_classes', type=str, default="2")
parser.add_argument('--attack', action="store_true", default=False)
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()

train_method = args.train_method
dataset = args.dataset


if args.dataset in ["breast_cancer", "cod-rna", "higgs"]:
	fstart = "0"
else:
	fstart = "1"


print("dataset:", dataset)
if dataset == "cod-rna":
	data_path = "data/cod-rna_s.t"
	nfeat = "8"
	args.num_classes = 2
	args.num_attacks = 5000
elif dataset == "binary_mnist":
	data_path = "data/binary_mnist0.t"
	nfeat = "784"
	args.num_classes = 2
	args.num_attacks = 100
elif dataset == "ijcnn":
	data_path = "data/ijcnn1s0.t"
	nfeat = "22"
	args.num_classes = 2
	args.num_attacks = 100
elif dataset == "breast_cancer":
	data_path = "data/breast_cancer_scale0.test"
	nfeat = "10"
	args.num_classes = 2
	args.num_attacks = 137

elif dataset == "fashion":
	data_path = "data/fashion.test0"
	nfeat = "10"
	args.num_classes = 10
	args.num_attacks = 100

elif dataset == "twitter":
	data_path = "twitter_spam/twitter_spam_reduced.test.libsvm"
	nfeat = "25"
	args.num_classes = 2
	args.num_attacks = 100
	fstart = "0"
else:
	print("no such dataset")
	exit()

if args.data_path is None:
	#data_path = "../DevRobustTrees/data/cod-rna_s.t"
	sample_tail = "_n"+str(args.num_attacks)
else:
    data_path = args.data_path
    sample_tail = "_c"+"_n"+str(args.num_attacks)

def eval(y, y_p):
    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        print(tp, tn, fp, fn)
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        tpr = tp/float(tp+fn)
        return acc, fpr, tpr
    except ValueError:
        return accuracy_score(y, y_p), None, None


def get_roc_curve(model_path, data_path, nfeat, fstart):
    #model = xgb.Booster()
    #model.load_model(model_path)
    loaded_model = pickle.load(open(model_path, 'rb'))
    fstart, nfeat = int(fstart), int(nfeat)

    print("data loaded from ", data_path)
    if data_path.endswith(".pickle"):
        x_test = pickle.load(open(data_path, 'rb'))
        y_test = np.ones(x_test.shape[0])
    elif data_path.endswith(".csv"):
        x_test = np.loadtxt(data_path, delimiter=',', usecols=list(range(1, nfeat+1)))
        y_test = np.loadtxt(data_path, delimiter=',', usecols=0).astype(int)
    else:
        # libsvm file format
        x_test, y_test = datasets.load_svmlight_file(data_path)
        x_test = x_test.toarray()
        #if fstart > 0:
            #x_test = np.hstack((np.zeros((x_test.shape[0],fstart)),x_test))
        y_test = y_test[:,np.newaxis].astype(int)
        
    #dtest = xgb.DMatrix(x_test, label=y_test)
    #preds = model.predict(dtest)
    #x_test = x_test[:, 1:]
    x_test = np.around(x_test, decimals=6)
    #preds = loaded_model.predict(x_test)
    preds = loaded_model.predict_proba(x_test)[:, 1]
    print((preds<0.1).sum(), (preds>0.5).sum(), (y_test==0).sum(), (y_test==1).sum(), preds.shape)
    results = loaded_model.score(x_test, y_test)
    print("sklearn rf acc score:", results)
    y_pred = [1 if p > 0.5 else 0 for p in preds]
    acc, fpr, tpr = eval(y_test, y_pred)
    print("accuracy: ", acc, "fpr: ", fpr, "tpr:", tpr)
    print("accuracy_score: {:.4f}".format(accuracy_score(y_test, y_pred)))
    
    fps, tps, thresholds = metrics.roc_curve(y_test, preds)
    auc = metrics.auc(fps, tps)
    print("AUC: {:.5f}".format(auc))
    return fps, tps, thresholds, auc


model_root = "models/sk-rf/"
model_paths = {"breast_cancer": {
                    "nature": model_root + "sklearn_breast_cancer_best_eps0.0_nt20_d4.pickle",
                    "robust": model_root + "sklearn_breast_cancer_heuristic_eps0.3_nt20_d4.pickle",
                    "greedy": model_root + "sklearn_breast_cancer_robust_eps0.3_nt80_d8.pickle" 
                    },
                "ijcnn":{
                    "nature": model_root + "sklearn_ijcnn_best_eps0.0_nt100_d14.pickle",
                    "robust": model_root + "sklearn_ijcnn_heuristic_eps0.03_nt100_d12.pickle",
                    "greedy": model_root + "sklearn_ijcnn_robust_eps0.03_nt60_d8.pickle"
                    },
                "cod-rna":{
                    "nature": model_root + "sklearn_cod-rna_best_eps0.0_nt40_d14.pickle",
                    "robust": model_root + "sklearn_cod-rna_heuristic_eps0.03_nt20_d14.pickle",
                    "greedy": model_root + "sklearn_cod-rna_robust_eps0.03_nt40_d14.pickle"
                    },
                "binary_mnist":{
                    "nature": model_root + "sklearn_binary_mnist_best_eps0.0_nt20_d14.pickle",
                    "robust": model_root + "sklearn_binary_mnist_heuristic_eps0.3_nt100_d12.pickle",
                    "greedy": model_root + "sklearn_binary_mnist_robust_eps0.3_nt100_d14.pickle"
                    }
        }

lists = ["nature", "robust", "greedy"]
markers = ['o', '*', 'v']
names = ["natural", "Chen's", "ours"]
dataset_names = {"breast_cancer":"breast-cancer", "ijcnn":"ijcnn1", "cod-rna":"cod-rna", "binary_mnist":"MNIST 2 vs. 6"}
markevery = 0.1

from matplotlib.pyplot import figure
figure(figsize=(2.8, 1.8))

for i, train_method in enumerate(lists):
    
    print("=================[{}]=====================".format(train_method))
    model_path = model_paths[dataset][train_method]
    print("model path:", model_path)

    fps, tps, thresholds, auc = get_roc_curve(model_path, data_path, nfeat, fstart)
    #print(fps, tps, thresholds)
    plt.plot(fps, tps, label=names[i]+ "({:.5f})".format(auc), marker=markers[i], markevery=markevery)


#plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc='lower right')
if not os.path.exists("roc_plots/"):
    os.mkdir("roc_plots/")
    print("make dir", "roc_plots/")
plt.title(dataset_names[dataset])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig("roc_plots/srf_"+dataset+"_roc.pdf", bbox_inches='tight')


