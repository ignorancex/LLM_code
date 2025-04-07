# Modified from https://github.com/coastalcph/treebank-leakage/blob/master/regress_from_spreadsheet.py

from sklearn.linear_model import LinearRegression as LinR
import sys
import numpy as np
from sklearn.model_selection import cross_val_score
import pickle
import re
from matplotlib import pyplot as plt
import scipy

def main():
    with open(sys.argv[1], "rb") as f:
        leakage_dict = pickle.load(f)

    # with open(sys.argv[2], "rb") as f:
    #     size_dict = pickle.load(f)

    with open(sys.argv[2], "rt") as f:
        # lines = [l.strip().split('\t') for l in f.readlines()[2:101]]
        lines=[l.strip().split('\t') for l in f.readlines()[1:]]

    # with open("Unlabeled\\tree_counts.dict", "rb") as f:
    #     size_dict = pickle.load(f)
    # y1 = 9 # UAS Stanza
    # y1 = 10 # LAS Stanza
    # y1 = 4 # UAS UDify
    # y1 = 5 # LAS UDify
    x1 = int(sys.argv[3])
    y1 = int(sys.argv[4])
    X = []
    Y = []
    xs = []
    ys = []
    labels = []

    for line in lines:
        treebank = line[0]
        if not (re.compile("^\s*$").match(line[9]) or line[9]=="-"):
        # if not line[y1]=="-":
            if treebank in leakage_dict:# and int(line[x1]) == 0:# and treebank not in count_dict:# and count_dict[treebank] > 5000:
                # if treebank in count_dict:
                #     X.append([count_dict[treebank], leakage_dict[treebank]])
                # else:
                #     X.append([0, leakage_dict[treebank]])
                X.append([float(line[x1]), leakage_dict[treebank]])
                # X.append([leakage_dict[treebank]])
                # X.append([int(line[x1])])
                # X.append([size_dict[treebank], leakage_dict[treebank]])
                Y.append(float(line[y1]))
                xs.append(leakage_dict[treebank])
                # xs.append(int(line[x1]))
                ys.append(float(line[y1]))
                labels.append(treebank)
            else:
                print("Missing", treebank)

    fig, ax = plt.subplots(figsize=(50,50))
    ax.scatter(xs, ys)
    plt.xlabel("Leakage")
    plt.ylabel("UAS Score")
    for idx, label in enumerate(labels):
        ax.annotate(label, (xs[idx], ys[idx]))
        
    # plt.savefig("Leakage vs Number of Morphologies", dpi=300)
    plt.show()

    print("|X|:", len(X))
    print("# of Lines:", len(lines)-1)
    reg = LinR(normalize=True).fit(X, Y)
    print("Regression Score:", reg.score(X, Y))
    print("Explained Variance:", cross_val_score(reg,X,Y,cv=5,scoring='explained_variance').mean())
    print("Negative Mean Absolute Error:", cross_val_score(reg,X,Y,cv=5,scoring='neg_mean_absolute_error').mean())
    print("Spearman coefficient:", scipy.stats.spearmanr(xs,ys).correlation)
if __name__ == "__main__":
    main()