import argparse
import logging
import sys

import numpy as np
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, DATASET_NAMES, RANDOM_SEED
from src.polarity.baseline.utils import dataset_get_x_y, get_stats_string, get_table_result_string, \
    evaluate_baseline_model, get_sum_info
from src.polarity.data.loader import DATASET_LOADERS

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

CLASSIFIERS_MAP = {
    "lr": LogisticRegression
}

VECTORIZERS_MAP = {
    'cv': CountVectorizer,
    'tfidf': TfidfVectorizer,
    'none': None
}


def main(args):
    data_loader = DATASET_LOADERS[args.dataset_name](args.max_train_data, args.binary)
    if args.use_kfold is True:
        run_kfold(args, data_loader)

    if args.use_grid is True:
        run_grid_search(args, data_loader)

    if args.use_train_test is True:
        run_train_test(args, data_loader)


def build_pipeline(char_vectorizer, word_vectorizer, char_ngram_range, word_ngram_range, max_features, classifier):
    if (char_vectorizer is None) and (word_vectorizer is None):
        raise Exception("You have tu use at least one vectorizer")

    features = []
    if char_vectorizer is not None:
        features.append(
            ('char-ngram-vec', char_vectorizer(max_features=max_features, ngram_range=char_ngram_range, analyzer='char')))

    if word_vectorizer is not None:
        features.append(('word-ngram-vec', word_vectorizer(max_features=max_features, ngram_range=word_ngram_range)))

    tg_pipeline = Pipeline([
        ('features', FeatureUnion(features)),
        ('classifier', classifier)
    ])

    return tg_pipeline

char_ngram_ranges = [(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5),
                                     (3, 6)]

word_ngram_ranges = [(1,1),(1,2),(1,3)]
lower_casing_ranges = [True]
max_features_ranges = np.arange(10000, 350000, 60000)

parmeters_both_vectorizers = {
    'features__char-ngram-vec__ngram_range': char_ngram_ranges,
    'features__char-ngram-vec__max_features': max_features_ranges,

    'features__word-ngram-vec__ngram_range' : word_ngram_ranges,
    'features__word-ngram-vec__max_features' : max_features_ranges

    # 'features__word-ngram-vec__lowercase' : lower_casing_ranges,
    # 'features__char-ngram-vec__lowercase': lower_casing_ranges,

}

parameters_word_ngram_vectorizer = {
    'features__word-ngram-vec__ngram_range' : word_ngram_ranges,
    'features__word-ngram-vec__max_features' : max_features_ranges
}

parameters_char_ngram_vectorizer = {
    'features__char-ngram-vec__ngram_range': char_ngram_ranges,
    'features__char-ngram-vec__max_features': max_features_ranges
}

# Tohle si pustim pro zajimavost, ale cross validaci
# pak pustim se stejnym nastavenim jako to ma tigi aby to bylo porovnatelne
def run_grid_search(args, data_loader):
    logger.info("Running grid search")

    data_train = data_loader.get_train_dev_data()
    data_test = data_loader.get_test_data()
    X_train, y_train = dataset_get_x_y(data_train, 'text', 'label')
    X_test, y_test = dataset_get_x_y(data_test, 'text', 'label')

    clas_names = data_loader.get_class_names()
    print("Train data:" + str(get_sum_info(X_train, data_train['label_text'], clas_names)))
    print("Test data:" + str(get_sum_info(X_test, data_train['label_text'], clas_names)))

    clas_names = data_loader.get_class_names()

    refit_score = 'f1_macro'
    average = 'macro'
    char_vectorizer = VECTORIZERS_MAP[args.char_ngram_vectorizer]
    word_vectorizer = VECTORIZERS_MAP[args.word_ngram_vectorizer]

    if (char_vectorizer is not None) and (word_vectorizer is not None):
        parameters = parmeters_both_vectorizers
    elif char_vectorizer is not None:
        parameters = parameters_char_ngram_vectorizer
    elif word_vectorizer is not None:
        parameters = parameters_word_ngram_vectorizer
    else:
        raise Exception("Something is wrong...")

    classifier = CLASSIFIERS_MAP[args.classifier]()
    word_ngram_range = (int(args.word_ngram_range.split(',')[0]), int(args.word_ngram_range.split(',')[1]))
    char_ngram_range = (int(args.char_ngram_range.split(',')[0]), int(args.char_ngram_range.split(',')[1]))

    tg_pipeline = build_pipeline(char_vectorizer=char_vectorizer,
                                 word_vectorizer=word_vectorizer,
                                 char_ngram_range=char_ngram_range,
                                 word_ngram_range=word_ngram_range,
                                 max_features=args.max_features,
                                 classifier=classifier)

    grid_pipeline = GridSearchCV(tg_pipeline, parameters, refit=refit_score,
                                 scoring = ['f1_macro','f1_micro','accuracy'],
                                 verbose=10, cv=args.n_folds, n_jobs=8)
                                 # verbose=10, cv=2, n_jobs=7)
    logger.info("Training...")
    t0 = time()
    model = grid_pipeline.fit(X_train, y_train)
    train_test_time = time() - t0
    logger.info("Training done")

    f1, accuracy, precision, recall, grid_string, grid_string_full = evaluate_baseline_model(model, X_test, y_test,
                                                                           refit_score,average,True)
    logger.info("Grid string full:" + grid_string_full)
    result_string,_ = get_table_result_string(f'{grid_string} | grid run default parameters(do not consider): {args}',
                                            f1, 0, accuracy, 0, precision, 0, recall, 0, train_test_time)
    print(result_string)


def run_train_test(args, data_loader):
    logger.info("Running train/test")

    labels = data_loader.get_classes()
    clas_names = data_loader.get_class_names()

    data_train = data_loader.get_train_dev_data()
    data_test = data_loader.get_test_data()
    X_train, y_train = dataset_get_x_y(data_train, 'text', 'label')
    X_test, y_test = dataset_get_x_y(data_test, 'text', 'label')

    print("Train data:" + str(get_sum_info(X_train, data_train['label_text'].astype(str).values.tolist(), clas_names)))
    print("Test data:" + str(get_sum_info(X_test, data_train['label_text'].astype(str).values.tolist(), clas_names)))

    classifier = CLASSIFIERS_MAP[args.classifier]()
    word_ngram_range = (int(args.word_ngram_range.split(',')[0]), int(args.word_ngram_range.split(',')[1]))
    char_ngram_range = (int(args.char_ngram_range.split(',')[0]), int(args.char_ngram_range.split(',')[1]))

    tg_pipeline = build_pipeline(char_vectorizer=VECTORIZERS_MAP[args.char_ngram_vectorizer],
                                 word_vectorizer=VECTORIZERS_MAP[args.word_ngram_vectorizer],
                                 char_ngram_range=char_ngram_range,
                                 word_ngram_range=word_ngram_range,
                                 max_features=args.max_features,
                                 classifier=classifier)
    logger.info("Training...")
    t0 = time()
    model = tg_pipeline.fit(X_train, y_train)
    train_test_time = time() - t0
    logger.info("Training done")

    average='macro'
    f1, accuracy, precision, recall, grid_string, grid_string_full = evaluate_baseline_model(model, X_test, y_test,
                                                                                             None, average, False)

    result_string, _ = get_table_result_string(f'train test: {args}',
                                            f1, 0, accuracy, 0, precision, 0, recall, 0, train_test_time)
    print(result_string)


def run_kfold(args, data_loader):
    logger.info("Running kfold")
    data = data_loader.load_entire_dataset()
    X, y = dataset_get_x_y(data, 'text', 'label')


    clas_names = data_loader.get_class_names()
    print("Data:" + str(get_sum_info(X, data['label_text'], clas_names)))

    classifier = CLASSIFIERS_MAP[args.classifier]()
    word_ngram_range = (int(args.word_ngram_range.split(',')[0]), int(args.word_ngram_range.split(',')[1]))
    char_ngram_range = (int(args.char_ngram_range.split(',')[0]), int(args.char_ngram_range.split(',')[1]))

    tg_pipeline = build_pipeline(char_vectorizer=VECTORIZERS_MAP[args.char_ngram_vectorizer],
                                 word_vectorizer=VECTORIZERS_MAP[args.word_ngram_vectorizer],
                                 char_ngram_range=char_ngram_range,
                                 word_ngram_range=word_ngram_range,
                                 max_features=args.max_features,
                                 classifier=classifier)

    mean_f1_macro, std_macro_f1, mean_acc, std_acc, mean_prec, std_prec, mean_recall, std_recall, train_test_time = \
        kfold_training(args.n_folds, X, y, tg_pipeline, clas_names, print_stats=True)

    logger.info("Training done")

    result_str, _  = get_table_result_string(f'kfold run: {args}',mean_f1_macro, std_macro_f1, mean_acc, std_acc,
                                                                  mean_prec, std_prec, mean_recall, std_recall,
                                                                  train_test_time)
    print(result_str)



def kfold_training(splits, X, Y, alg, classes_pred, print_stats=True):
    average = 'macro'

    # kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=666)
    kfold = KFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED)

    accuracy = []
    precision = []
    recall = []
    macro_f1 = []

    precisions = []
    recalls = []
    f1_scores = []

    t0 = time()
    for train_index, test_index in kfold.split(X=X, y=Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = alg.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy_tmp = accuracy_score(y_test, y_pred)
        macro_f1_tmp = f1_score(y_test, y_pred, average=average)
        precision_tmp = precision_score(y_test, y_pred, average=average)
        recall_tmp = recall_score(y_test, y_pred, average=average)

        accuracy.append(accuracy_tmp)
        macro_f1.append(macro_f1_tmp)
        precision.append(precision_tmp)
        recall.append(recall_tmp)

        precisions_tmp = precision_score(y_test, y_pred, average=None)
        recalls_tmp = recall_score(y_test, y_pred, average=None)
        f1_scores_tmp = f1_score(y_test, y_pred, average=None)

        precisions.append(precisions_tmp)
        recalls.append(recalls_tmp)
        f1_scores.append(f1_scores_tmp)

        if print_stats:
            print('           ', classes_pred)
            print('precision:', precisions_tmp)
            print('recall:   ', recalls_tmp)
            print('f1 score: ', f1_scores_tmp)

            print('----Average----')
            print('accuracy ', accuracy_tmp)
            print('f1 macro score: ', macro_f1_tmp)
            print('precision', precision_tmp)
            print('recall', recall_tmp)
            # print_confusion_matrix(y_test,y_pred)
            print('-' * 70)

    train_test_time = time() - t0

    mean_f1_macro = np.mean(macro_f1)
    mean_acc = np.mean(accuracy)
    mean_prec = np.mean(precision)
    mean_recall = np.mean(recall)

    mean_precisions = np.mean(precisions, axis=0)
    mean_recalls = np.mean(recalls, axis=0)
    mean_f1_scores = np.mean(f1_scores, axis=0)

    printStr = get_stats_string(mean_acc, mean_f1_macro, 0.0, mean_precisions, mean_recalls, mean_f1_scores,
                                classes_pred)

    std_macro_f1 = np.std(macro_f1)
    std_acc = np.std(accuracy)
    std_prec = np.std(precision)
    std_recall = np.std(recall)
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("f1 score: %.4f%% (+/- %.4f%%)" % (mean_f1_macro, std_macro_f1))
    print("accuracy: %.4f%% (+/- %.4f%%)" % (mean_acc, std_acc))
    print("precision: %.4f%% (+/- %.4f%%)" % (mean_prec, std_prec))
    print("recall: %.4f%% (+/- %.4f%%)" % (mean_recall, std_recall))
    print("--------------")
    print(printStr)

    return mean_f1_macro, std_macro_f1, mean_acc, std_acc,mean_prec, std_prec, mean_recall, std_recall, train_test_time


if __name__ == '__main__':
    # sys.argv.extend(['--dataset_name', 'csfd'])
    # sys.argv.extend(['--use_train_test'])
    # sys.argv.extend(['--use_kfold'])
    # sys.argv.extend(['--use_grid'])
    # sys.argv.extend(['--binary'])
    # sys.argv.extend(['--word_ngram_vectorizer','cv'])
    # sys.argv.extend(['--char_ngram_vectorizer','none'])

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Required parameters
    parser.add_argument("--dataset_name",
                        required=True,
                        choices=DATASET_NAMES,
                        help="The dataset that will be used, they correspond to names of folders")

    # Non required parameters
    # Parameters for kfold only option
    parser.add_argument("--use_kfold",
                        action='store_true',
                        help="If used, kfold is used for testing with 10 folds")

    # Parameters for grid search
    parser.add_argument("--use_grid",
                        action='store_true',
                        help="If used, grids search is used")

    parser.add_argument("--use_train_test",
                        action='store_true',
                        help="If used, train/test evaluation is performed")


    # Shared parameters for kfold, grid, train/test
    parser.add_argument("--classifier",
                        default='lr',
                        choices=['lr'],
                        help="Type of classifier one of 'lr' ")

    # Shared parameters for kfold, grid train/test
    parser.add_argument("--n_folds",
                        default="10",
                        type=int,
                        help="Number of folds in kfold validation")

    parser.add_argument("--word_ngram_vectorizer",
                        default='cv',
                        choices=['cv', 'tfidf', 'none'],
                        help="Type of vectorizer for word ngrams one of: 'cv', 'tfidf'")

    parser.add_argument("--word_ngram_range",
                        default="1,2",
                        nargs=2,
                        type=str,
                        action='append',
                        help="Range of word ngrams, type two numbers separate with comma")

    parser.add_argument("--char_ngram_vectorizer",
                        default='cv',
                        choices=['cv', 'tfidf', 'none'],
                        help="Type of vectorizer for char ngrams one of: 'cv', 'tfidf'")

    parser.add_argument("--char_ngram_range",
                        default="3,6",
                        nargs=2,
                        type=str,
                        action='append',
                        help="Range of char ngrams, type two numbers separate with comma")

    parser.add_argument("--max_features",
                        default="2600000",
                        type=int,
                        help="Max features for each vectorizer")


    # Other parameters
    parser.add_argument("--silent",
                        default=False,
                        action='store_true',
                        help="If used, logging is set to ERROR level, otherwise INFO is used")

    parser.add_argument("--binary",
                        default=False,
                        action='store_true',
                        help="If used the polarity task is treated as binary classification, i.e., positve/negative"
                             " The neutral examples are dropped")

    parser.add_argument("--max_train_data",
                        default=-1,
                        type=float,
                        help="Amount of data that will be used for training, "
                             "if (-inf, 0> than it is ignored, "
                             "if (0,1> than percentage of the value is used as training data, "
                             "if (1, inf) absolute number of training examples is used as training data")


    args = parser.parse_args()
    if args.silent is True:
        logging.root.setLevel(level=logging.ERROR)
    else:
        logging.root.setLevel(level=logging.INFO)

    logger.info(f"Running baseline with the following parameters:{args}")
    logger.info("-------------------------")
    main(args)
