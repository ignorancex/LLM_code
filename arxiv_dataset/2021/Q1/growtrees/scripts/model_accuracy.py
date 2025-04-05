#! /usr/bin/env python
import sys
import os
import random
import json
import argparse
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Load xgboost model to predict test data.')
    parser.add_argument('--model_path', type=str, help='binary xgboost model path.', required=True)
    parser.add_argument('--test_data', type=str, help='test data file name.', required=True)
    parser.add_argument('--nfeat', type=int, help='number of features.', required=True)
    parser.add_argument('--fstart', type=int, help='whether the feature starts from 0 or 1.', required=True)
    return parser.parse_args()


def eval(y, y_p):
    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        print(tp, tn, fp, fn)
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        return acc, fpr
    except ValueError:
        return accuracy_score(y, y_p), None

def main(args):
    # load the trained model
    model = xgb.Booster()
    model.load_model(args.model_path)
    print("data loaded from", args.test_data)
    if args.test_data.endswith(".pickle"):
        x_test = pickle.load(open(args.test_data, 'rb'))
        y_test = np.ones(x_test.shape[0])
    elif args.test_data.endswith(".csv"):
        x_test = np.loadtxt(args.test_data, delimiter=',', usecols=list(range(1, args.nfeat+1)))
        y_test = np.loadtxt(args.test_data, delimiter=',', usecols=0).astype(int)
    else:
        # libsvm file format
        x_test, y_test = datasets.load_svmlight_file(args.test_data)
        x_test = x_test.toarray()
        if args.fstart > 0:
            x_test = np.hstack((np.zeros((x_test.shape[0],args.fstart)),x_test))
        y_test = y_test[:,np.newaxis].astype(int)
        '''
        x_test, y_test = datasets.load_svmlight_file(args.test_data,
                                       n_features=args.nfeat,
                                       multilabel=False,
                                       zero_based=(args.fstart==0),
                                       query_id=False)
        '''

        print(np.sum(y_test))

    print(x_test.shape)
    dtest = xgb.DMatrix(x_test, label=y_test)
    preds = model.predict(dtest)
    y_pred = [1 if p > 0.5 else 0 for p in preds]

    # get accuracy from preds and y_test
    acc, fpr = eval(y_test, y_pred)
    print("accuracy: ", acc, "fpr: ", fpr)
    print("accuracy_score: ", accuracy_score(y_test, y_pred))
    return


if __name__=='__main__':
    args = parse_args()
    main(args)
