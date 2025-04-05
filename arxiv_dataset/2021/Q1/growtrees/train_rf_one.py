from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import argparse
import os


def eval(y, y_p):
    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        acc = (tp + tn) / float(tp + tn + fp + fn)
        fpr = fp / float(fp + tn)
        return acc, fpr
    except ValueError:
        return accuracy_score(y, y_p), None


def parse_args():
    parser = argparse.ArgumentParser(description='Train one sklearn RF model.')
    parser.add_argument('--train', '--train_data', type=str, help='train data file name.', required=True)
    parser.add_argument('--validation', '--validation_data', type=str, help='test data file name.', default=None,
                        required=False)
    parser.add_argument('--test', '--test_data', type=str, help='test data file name.', required=True)
    parser.add_argument('-m', '--model_path', type=str, help='save sklearn model pickle path.', required=True)
    parser.add_argument('-b', '--binary_class', default=False, help='whether it is binary class.', action='store_true')
    parser.add_argument('-n', '--nfeat', type=int, help='number of features.', required=True)
    parser.add_argument('-z', '--zero_start', default=False, help='whether the feature starts from 0.',
                        action='store_true')
    parser.add_argument('-r', '--robust', default=False, help='whether to use robust training.', action='store_true')
    parser.add_argument('-s', '--splitter', type=str, default='best', choices=['best', 'robust', 'even', 'heuristic'],
                        help='choose the splitter.', required=False)
    parser.add_argument('-e', '--eps', type=float, default=0.0, help='robust epsilon range.', required=False)
    parser.add_argument('-c', '--criterion', type=str, default='gini', help='the splitting criterion.', required=False)
    parser.add_argument('--nt', type=int, help='number of decision trees.', required=True)
    parser.add_argument('-d', '--max_depth', type=int, help='maximum tree depth.', required=True)
    parser.add_argument('--max_features', help='number of features to consider when looking for the best split',
                        type=float, required=False)
    parser.add_argument('-v', '--verbose', help='verbose training', type=int, choices=[0, 1], default=0)
    parser.add_argument('--round_data', type=int, help='round train and test data', required=False, default=20)
    return parser.parse_args()


def main(train, validation, test, model_path, binary_class=False, nfeat=None, zero_start=False, robust=False,
         splitter="best", eps=0.0, criterion="gini", nt=None, max_depth=None, max_features="auto", verbose=0,
         round_data=20):
    print("training data path:", train)
    print("testing data path:", test)

    if train.endswith("pickle"):
        with open(train, "rb") as fin:
            x_train, y_train = pickle.load(fin)
    else:
        x_train, y_train = datasets.load_svmlight_file(train,
                                                       n_features=nfeat,
                                                       multilabel=(not binary_class),
                                                       zero_based=zero_start)
        x_train = x_train.toarray().astype(float)

    if validation is not None:  # assume validation data always supplied as pickle
        with open(validation, "rb") as fin:
            x_validation, y_validation = pickle.load(fin)

    if test.endswith("pickle"):
        with open(test, "rb") as fin:
            x_test, y_test = pickle.load(fin)
    else:
        x_test, y_test = datasets.load_svmlight_file(test,
                                                     n_features=nfeat,
                                                     multilabel=(not binary_class),
                                                     zero_based=zero_start)
        x_test = x_test.toarray().astype(float)

    clf = RandomForestClassifier(
        robust=robust,
        epsilon=eps,
        splitter=splitter,
        verbose=verbose, criterion=criterion,
        n_estimators=nt, max_depth=max_depth, random_state=0,
        n_jobs=64,
        max_features=max_features)
    clf.fit(np.around(x_train, decimals=round_data), y_train)

    print("Model params: ", clf.get_params())

    if validation is not None:
        y_hat_validation = clf.predict(np.around(x_validation, decimals=round_data))
        validation_acc, validation_fpr = eval(y_validation, y_hat_validation)
        print("RF Validation Accuracy: ", validation_acc, "FPR: ", validation_fpr)

    y_hat = clf.predict(np.around(x_test, decimals=round_data))
    test_acc, test_fpr = eval(y_test, y_hat)
    print("RF Test Accuracy: ", test_acc, "FPR: ", test_fpr)

    pickle.dump(clf, open(model_path, "wb"))
    print('saved model at {}'.format(model_path))

    '''
    # save to json
    json_path = '%s.json' % model_path.split('.pickle')[0]
    cmd = 'python3 save_sklearn_rf_to_json.py \
            --model_path %s \
            --output %s' % (model_path, json_path)
    print(cmd)
    os.system(cmd)
    '''

    if validation is not None:
        return validation_acc, validation_fpr, test_acc, test_fpr
    else:
        return test_acc, test_fpr


if __name__ == '__main__':
    args = parse_args()
    main(args.train, args.validation, args.test, args.model_path, args.binary_class, args.nfeat, args.zero_start,
         args.robust, args.splitter,
         args.eps, args.criterion, args.nt, args.max_depth, args.max_features, args.verbose, args.round_data)
