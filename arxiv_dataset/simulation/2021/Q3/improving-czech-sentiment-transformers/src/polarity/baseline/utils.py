import logging
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger(__name__)


def dataset_get_x_y(my_df, x_col, y_col):
    x = my_df[x_col].astype(str).values.tolist()
    y = my_df[y_col].astype(int).values.tolist()

    return x, y


def get_stats_string(accuracy, macro_f1, micro_f1, precisions, recalls, f1_scores, classes):
    string = '    \t      ' + '\t '.join(classes) + '\n'\
          + 'precision\t' + '\t   '.join(format(x, "2.4f") for x in precisions) + '\n'\
          + 'recall\t   ' + '\t   '.join(format(x, "2.4f") for x in recalls) + '\n' \
          + 'f1 score\t ' + '\t   '.join(format(x, "2.4f") for x in f1_scores) + '\n' \
          + '----Average----' + '\n' \
          + 'accuracy: %2.4f ' % (accuracy) + '\n' \
          + 'f1 macro score: %2.4f ' % (macro_f1) + '\n' \
          + 'f1 micro score: %2.4f ' % (micro_f1)

    return string


def get_table_result_string(config_string, mean_f1_macro, std_macro_f1, mean_acc, std_acc, mean_prec, std_prec, mean_recall,
                            std_recall, train_test_time):
    results = f'{config_string}\t{mean_f1_macro}\t{mean_acc}\t{mean_prec}\t{mean_recall}\t{std_macro_f1}\t{std_acc}\t{std_prec}\t{std_recall}\t{int(train_test_time)} s'
    results_head = '\tF1 Macro\tAccuracy\tPrecision\tRecall\tStd F1 Macro\tStd Accuracy\tStd precision\tStd recall\ttime\n' + results


    return results_head, results


def evaluate_predictions(y_pred, y_test, average='macro'):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)

    return f1, accuracy, precision, recall

def evaluate_baseline_model(model, X_test, y_test, refit_score='f1_macro',
                             average='macro', grid=True, categorical_output=False,
                            prob_output=False):
    y_pred = model.predict(X_test)

    if prob_output is True:
        y_pred = label_probabilities(y_pred)

    if categorical_output is True:
        y_pred = de_vectorize(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)

    grid_string = ''
    grid_string_full = ''
    if grid:
        grid_string, grid_string_full = get_printable_string_grid_search(model, refit_score)

    return f1, accuracy, precision, recall, grid_string, grid_string_full




def get_printable_string_grid_search(model, refit_score):
    cv_results = model.cv_results_
    means_refit = cv_results['mean_test_' + refit_score]
    stds_refit = cv_results['std_test_' + refit_score]
    means_micro = cv_results['mean_test_f1_micro']
    params = cv_results['params']

    str = "Grid Best: %f using %s ," % (model.best_score_, model.best_params_)
    str_full = str
    str_full += '\n-------\n'
    for mean, stdev, micro_mean, param in zip(means_refit, stds_refit, means_micro, params):
        str_full += "%0.3f (+/-%0.03f), micro: %0.3f for %r" % (mean, stdev * 2, micro_mean, param) + '\n'

    return str, str_full

def get_sum_info(X, y, classes):
    ret = 'set has total ' + str(len(X)) + ' entries with \n'
    for i, clazz in enumerate(classes):
        tmp = '{0:.2f}% ({1:d}) - ' + clazz + '\n'
        class_len = len(X[y == clazz])
        tmp = tmp.format((class_len / (len(X) * 1.)) * 100, class_len)
        ret = ret + tmp
    ret = ret + '------------'

    return ret


# predictions from softmax are in array like [[0.01,0.6,0.09,0.3],[...]]
# function select the max probability and replace it with 1
# [[0.01,0.6,0.09,0.3],[...]] => [[0,1,0,0],[...]]
def label_probabilities(prob_vectors):
    list = []
    for vector in prob_vectors:
        max_index = np.argmax(vector)
        new_vector = [0] * len(vector)
        new_vector[max_index] = 1
        list.append(new_vector)

    return list


# because keras works with vectorized labels like [[0,0,1,0],[1,0,0,0]]
# we need to to convert to [2,0]
def de_vectorize(label_vectors):
    list = []
    for vector in label_vectors:
        max_index = np.argmax(vector)
        list.append(max_index)

    return list