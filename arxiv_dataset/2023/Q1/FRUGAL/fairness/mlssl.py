from __future__ import division, print_function
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier, LabelPropagation
from sklearn_cotraining import CoTrainingClassifier
from sklearn.ensemble import RandomForestClassifier


from datetime import timedelta
import pandas as pd
import numpy as np
import sklearn
import random
import pdb
import os
from demos import cmd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from os import listdir
import copy
import collections
try:
    import cPickle as pickle
except:
    import pickle
from learners import Treatment, TM, SVM, RF, DT, NB, LR
import warnings
warnings.filterwarnings('ignore')
from fairway import load_csv
from result.measure import get_counts_german_cla, get_counts_cla
from sklearn.metrics import confusion_matrix
BUDGET = 50
POOL_SIZE = 10000
INIT_POOL_SIZE = 10
np.random.seed(4789)


def getHigherValueCutoffs(data, percentileCutoff):
    '''
	Parameters
	----------
	data : in pandas format
	percentileCutoff : in integer
	class_category : [TODO] not needed

	Returns
	-------
	'''
    # pdb.set_trace()
    abc = data.quantile(float(percentileCutoff) / 100)
    abc = np.array(abc.values)[:-1]
    if abc.shape[0] == 0:
        abc = []
        for c in data.columns[:-1]:
            abc.append(np.percentile(data[c].values, percentileCutoff))
        abc = np.array(abc)
    return abc


def filter_row_by_value(row, cutoffsForHigherValuesOfAttribute):
    '''
	Shortcut to filter by rows in pandas
	sum all the attribute values that is higher than the cutoff
	----------
	row
	cutoffsForHigherValuesOfAttribute

	Returns
	-------
	'''
    rr = row[:-1]
    condition = np.greater(rr, cutoffsForHigherValuesOfAttribute)
    res = np.count_nonzero(condition)
    return res


def getInstancesByCLA(data, percentileCutOff, positiveLabel):
    '''
	- unsupervised clustering by median per attribute
	----------
	data
	percentileCutOff
	positiveLabel

	Returns
	-------

	'''
    # pdb.set_trace()
    # get cutoff per fixed percentile for all the attributes
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(data, percentileCutOff)
    # get K for all the rows
    K = data.apply(lambda row: filter_row_by_value(row, cutoffsForHigherValuesOfAttribute), axis=1)
    # cutoff for the cluster to be partitioned into
    cutoffOfKForTopClusters = np.percentile(K, percentileCutOff)
    instances = [1 if x > cutoffOfKForTopClusters else 0 for x in K]
    data["CLA"] = instances
    data["K"] = K
    return data


def getInstancesByRemovingSpecificAttributes(data, attributeIndices, invertSelection, label="Label"):
    '''
	removing the attributes
	----------
	data
	attributeIndices
	invertSelection

	Returns
	-------
	'''
    # attributeIndices = data.columns[attributeIndices]
    if not invertSelection:
        data_res = data.drop(data.columns[attributeIndices], axis=1)
    else:
        # invertedIndices = np.in1d(range(len(attributeIndices)), attributeIndices)
        # data.drop(data.columns[invertedIndices], axis=1, inplace=True)
        data_res = data[attributeIndices]
        data_res['Label'] = data[label].values
    return data_res


def getInstancesByRemovingSpecificInstances(data, instanceIndices, invertSelection):
    '''
	removing instances
	----------
	data
	instanceIndices
	invertSelection

	Returns
	-------

	'''
    if not invertSelection:
        data.drop(instanceIndices, axis=0, inplace=True)
    else:
        invertedIndices = np.in1d(range(data.shape[0]), instanceIndices)
        data.drop(invertedIndices, axis=0, inplace=True)
    return data


def getSelectedInstances(data, cutoffsForHigherValuesOfAttribute, positiveLabel):
    '''
	select the instances that violate the assumption
	----------
	data
	cutoffsForHigherValuesOfAttribute
	positiveLabel

	Returns
	-------
	'''
    violations = data.apply(lambda r: getViolationScores(r,
                                                         data['Label'],
                                                         cutoffsForHigherValuesOfAttribute),
                            axis=1)
    violations = violations.values
    # get indices of the violated instances
    selectedInstances = (violations > 0).nonzero()[0]
    selectedInstances = data.index.values[selectedInstances]
    # remove randomly 90% of the instances that violate the assumptions
    # selectedInstances = np.random.choice(selectedInstances, int(selectedInstances.shape[0] * 0.9), replace=False)
    # for index in range(data.shape[0]):
    # 	if violations[index] > 0:
    # 		selectedInstances.append(index)
    try:
        tmp = data.loc[selectedInstances]
    except:
        tmp = data.loc[selectedInstances]
    if tmp[tmp["Label"] == 1].shape[0] < 10 or tmp[tmp["Label"] == 0].shape[0] < 10:
        # print("not enough data after removing instances")
        # category = 1 if tmp[tmp["Label"] == 1].shape[0] < 10 else 0
        len_0 = selectedInstances.shape[0]
        # len_0 -= data[data["Label"] == 1].shape[0]
        selectedInstances = np.random.choice(selectedInstances, int(len_0 * 0.9), replace=False)

    return selectedInstances


def getCLAresults(seed=0, input="../new_data/corrected/", output="results/SE_CLAMI_reverse_"):
    treatments = ['chromium', 'firefox', 'eclipse']
    abc = range(5, 50, 5)
    data = {"CLA_%s" % x: [] for x in abc}
    columns = ["Treatment"] + list(sorted(data.keys()))
    result = {}
    data = load_csv(path=input)
    for t in treatments:
        dataset = data[t]
        result[t] = {}
        print(t, dataset["test"].shape)
        for x in abc:
            key = "CLA_%s" % x
            result[t][key] = CLA(dataset["test"], None, x)
            print(result[t][key])
    final_result = {}
    for x in abc:
        key = "CLA_%s" % x
        final_result[key] = []
        for t in treatments:
            final_result[key].append(result[t][key])
    sort_keys = list(final_result.keys())
    sort_keys.sort()
    sort_keys = ["Treatment"] + sort_keys
    print(sort_keys)
    final_result["Treatment"] = treatments

    # Output results to tables
    metrics = final_result[columns[-1]][0].keys()
    pdb.set_trace()
    for metric in metrics:
        df = {key: (final_result[key] if key == "Treatment" else [dict[metric] for dict in final_result[key]]) for key
              in sort_keys}
        pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
                                                 line_terminator="\r\n", index=False)

def getCLAMIresults(seed=0, input="../new_data/corrected/", output="results/CLAMI_FULL_"):
    treatments = ['chromium', 'firefox', 'eclipse']
    abc = range(5, 100, 5)
    data = {"CLA_%s" % x: [] for x in abc}
    columns = ["Treatment"] + list(sorted(data.keys()))
    result = {}
    data = load_csv(path=input)
    for t in treatments:
        dataset = data[t]
        result[t] = {}
        print(t, dataset["test"].shape)
        for x in abc:
            key = "CLA_%s" % x
            result[t][key] = CLAMI(dataset, "test", None, x, label="CLA")
            print(result[t][key])
    final_result = {}
    for x in abc:
        key = "CLA_%s" % x
        final_result[key] = []
        for t in treatments:
            final_result[key].append(result[t][key])
    sort_keys = list(final_result.keys())
    sort_keys.sort()
    sort_keys = ["Treatment"] + sort_keys
    print(sort_keys)
    final_result["Treatment"] = treatments

    # Output results to tables
    metrics = final_result[columns[-1]][0].keys()
    pdb.set_trace()
    for metric in metrics:
        df = {key: (final_result[key] if key == "Treatment" else [dict[metric] for dict in final_result[key]]) for key
              in sort_keys}
        pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
                                                 line_terminator="\r\n", index=False)


def two_step_Jitterbug(data, target, model="RF", est=False, T_rec=0.9, inc=False, seed=0,
                       hybrid=True, CLAfilter=False, CLAlabel=True, toggle=False, early_stop=False):
    np.random.seed(seed)
    jitterbug = CLAHard(data, target, cla=False, thres=55, early_stop=early_stop,
                        CLAfilter=CLAfilter, CLAlabel=CLAlabel, hybrid=hybrid, toggle=toggle)
    # jitterbug.find_patterns()
    jitterbug.test_patterns(include=inc)
    tmp = jitterbug.rest[target]
    print(len(tmp[tmp["label"] == "yes"]))

    jitterbug.ML_hard(model=model, est=est, T_rec=T_rec)
    # stats = None
    return jitterbug


def CLA_SL(data, target, percentileCutoff, model="RF", seed=0, both=False, stats={"tp": 0, "p": 0}):
    try:
        traindata = copy.deepcopy(data["train"])
        testdata = copy.deepcopy(data[target])
    except:
        traindata = copy.deepcopy(data)
        testdata = copy.deepcopy(target)
    final_data = getInstancesByCLA(traindata, percentileCutoff, None)
    final_data["Label"] = final_data["CLA"]
    final_data.drop(["CLA", "K"], axis=1, inplace=True)
    results = training_CLAMI(final_data, testdata, target, model, stats=stats)
    return results


def get_CLASUP(seed=0, input="../new_data/corrected/", output="results/CLA50+SUP_"):
    treatments = ["LR", "DT", "RF", "SVM", "NB"]
    data = load_csv(path=input)
    columns = ["Treatment"] + list(data.keys())

    # Supervised Learning Results
    result = {}
    result["Treatment"] = treatments
    for target in data:
        dataset = data[target]
        result[target] = [CLA_SL(dataset, "test", 50, model=model, seed=seed) for model in treatments]
        # Output results to tables
        metrics = result[target][0].keys()
        print(result[target])
        for metric in metrics:
            df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in
                  result}
            pd.DataFrame(df, columns=columns).to_csv(output + metric + ".csv", line_terminator="\r\n",
                                                     index=False)


def get_MLSSL(seed=0, output="../results/MLSSL_025_"):
    treatments = ['ST', 'LS', 'LP', 'CT']
    data = load_csv()
    columns = ["Treatment"] + list(data.keys())

    # Supervised Learning Results
    result = {}
    result["Treatment"] = treatments
    for target in data:
        dataset = data[target]
        # result[target] = [tuning_frugal(dataset, "test", seed=seed)]
        result[target] = [execute(dataset, target, method="ST", seed=seed)]
        result[target].append(execute(dataset, target, method="LS", seed=seed))
        result[target].append(execute(dataset, target, method="LP", seed=seed))
        result[target].append(execute(dataset, target, method="CT", seed=seed))
        # Output results to tables
        metrics = result[target][0].keys()
        print(result[target])
        for metric in metrics:
            df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in
                  result}
            pd.DataFrame(df, columns=columns).to_csv(output + metric + ".csv", line_terminator="\r\n",
                                                     index=False)


def getJIT_CLAresults(seed=0, input="../new_data/corrected/", output="../results/SE_FALCON_"):
    treatments = ["JITCLA"]
    data = load_csv(path=input)
    columns = ["Treatment"] + list(data.keys())
    result = {}
    keys = list(data.keys())
    keys.sort()
    data_type = output.split("/")[-1]
    print(keys, data_type, "../dump/%sresult.pickle" % data_type)
    total_yes_count = 0
    yes_count_median = []
    stats_satds = {}
    keys = ["apache-jmeter-2.10", "jruby-1.4.0", "hibernate-distribution-3.3.2.GA",
            "emf-2.4.1", "apache-ant-1.7.0", "sql12",
            "columba-1.4-src", "argouml", "jfreechart-1.0.19", "jEdit-4.2"]
    for target in keys:
        # "jEdit-4.2" "emf-2.4.1"
        # target_data_label = data[target]["label"]
        # yes_count = target_data_label[target_data_label == "yes"].shape[0]
        # total_yes_count += yes_count
        # yes_count_median.append(yes_count)
        jitterbug = two_step_Jitterbug(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                       hybrid=True, CLAfilter=False, CLAlabel=False, toggle=False, early_stop=False)
        result[target] = [jitterbug.eval()]
        stats_satds[target] = jitterbug.stats_satd
        print(result[target])

    with open("../dump/%sSATD_stats.pickle" % data_type, "wb") as f:
        pickle.dump(stats_satds, f)
    print(total_yes_count, np.median(yes_count_median), np.median(yes_count_median) / total_yes_count)

    result["Treatment"] = treatments
    # Output results to tables
    metrics = result[columns[-1]][0].keys()
    for metric in metrics:
        df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in keys}
        pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
                                                 line_terminator="\r\n", index=False)

    with open("../dump/%sresult.pickle" % data_type, "wb") as f:
        pickle.dump(result, f)


def CLA(data, positiveLabel, percentileCutoff, suppress=0, experimental=0, both=False):
    try:
        treatment = Treatment({}, "")
    except:
        treatment = Treatment(data, "")
    final_data = getInstancesByCLA(data, percentileCutoff, positiveLabel)
    treatment.y_label = ["yes" if y == 1 else "no" for y in final_data["Label"]]
    treatment.decisions = ["yes" if y == 1 else "no" for y in final_data["CLA"]]
    summary = collections.Counter(treatment.decisions)
    print(summary, summary["yes"] / (summary["yes"] + summary["no"]))
    treatment.probs = final_data["K"]
    results = treatment.eval()
    results["read"] = summary["yes"] / (summary["yes"] + summary["no"])
    return results


def KMEANS(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, both=False):
    treatment = Treatment(data, target)
    treatment.preprocess()
    testdata = treatment.full_test
    data = getInstancesByCLA(testdata, percentileCutoff, positiveLabel)
    treatment.y_label = ["yes" if y == 1 else "no" for y in data["Label"]]
    treatment.decisions = ["yes" if y == 1 else "no" for y in data["CLA"]]
    treatment.probs = data["K"]
    return treatment.eval()


def CLAMI(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, stats={"tp": 0, "p": 0},
          label="Label"):
    '''
	CLAMI - Clustering, Labeling, Metric/Features Selection,
			Instance selection, and Supervised Learning
	----------

	Returns
	-------

	'''
    # pdb.set_trace()
    try:
        traindata = copy.deepcopy(data["train"])
        testdata = copy.deepcopy(data[target])
    except:
        traindata = copy.deepcopy(data)
        testdata = copy.deepcopy(target)
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(traindata, percentileCutoff)
    # print("get cutoffs")
    traindata = getInstancesByCLA(traindata, percentileCutoff, positiveLabel)
    # print("get CLA instances")

    metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(traindata,
                                                                                 cutoffsForHigherValuesOfAttribute,
                                                                                 positiveLabel, label=label)
    # print("get Features and the violation scores")
    # pdb.set_trace()
    keys = list(metricIdxWithTheSameViolationScores.keys())
    # start with the features that have the lowest violation scores
    keys.sort()
    for i in range(len(keys)):
        k = keys[i]
        selectedMetricIndices = metricIdxWithTheSameViolationScores[k]
        # while len(selectedMetricIndices) < 3:
        # 	index = i + 1
        # 	selectedMetricIndices += metricIdxWithTheSameViolationScores[keys[index]]
        # print(selectedMetricIndices)
        # pick those features for both train and test sets
        trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(traindata,
                                                                            selectedMetricIndices, True, label=label)
        newTestInstances = getInstancesByRemovingSpecificAttributes(testdata,
                                                                    selectedMetricIndices, True, label="Label")
        # restart looking for the cutoffs in the train set
        cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
                                                                  percentileCutoff)
        # get instaces that violated the assumption in the train set
        instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
                                                       cutoffsForHigherValuesOfAttribute,
                                                       positiveLabel)
        # remove the violated instances
        trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
                                                                           instIndicesNeedToRemove, False)

        # make sure that there are both classes data in the training set
        zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
        one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
        if zero_count > 0 and one_count > 0:
            break
    # try:
    #     print("Before filtering: ", data["train"].shape[0], "After filtering: ", zero_count + one_count)
    # except:
    #     print("Before filtering: ", data.shape[0], "After filtering: ", zero_count + one_count)
    return CLAMI_eval(trainingInstancesByCLAMI, newTestInstances, target, stats=stats)


def percentile_tuning(func, train, tune):
    percentiles = range(5, 100, 5)
    results = []
    for p in percentiles:
        if func == "CLA":
            res = CLA(tune, None, p)
        elif func == "CLAMI":
            res = CLAMI(train, tune, None, p, label="CLA")
        elif func == "CLASUP":
            res = CLA_SL(train, tune, p, model="RF")
    #     results.append([res[metric], p])
    # results.sort(key = lambda x: x[0])
    # if metric == "fall-out":
    #     return results[-1][1]
    # else:
    #     return results[0][1]
        results.append([res, p])
    return results


def run_method(func, train, test, metric, multi=False, perc=50):
    if func == "CLA":
        res = CLA(test, None, perc)
    elif func == "CLAMI":
        res = CLAMI(train, test, None, perc, label="CLA")
    elif func == "CLASUP":
        res = CLA_SL(train, test, perc, model="RF")
    if not multi:
        return res[metric]
    else:
        return res


def tuning(data, target, method="CLA", seed=0):
    np.random.seed(seed)
    random.seed(seed)
    traindata = copy.deepcopy(data["train"])
    testdata = copy.deepcopy(data[target])
    sss = StratifiedShuffleSplit(n_splits=5, test_size=.05, random_state=seed)
    X, y = traindata[traindata.columns[:-1]], traindata[traindata.columns[-1]]
    # recall, fpr, precision, f-score1
    metrics = ["recall", "fall-out", "precision", "f1"]
    results = {m: [] for m in metrics}
    index = 0
    for train_index, tune_index in sss.split(X, y):
        print("Iteration = ", index)
        train_df = traindata.iloc[train_index]
        tune_df = traindata.iloc[tune_index]
        percentile_res = percentile_tuning(method, train_df, tune_df)
        for m in metrics:
            m_res = [[x[0][m], x[1]] for x in percentile_res]
            m_res.sort(key=lambda x: x[0])
            percentile = m_res[-1][1] if m != "fall-out" else m_res[0][1]
            res = run_method(method, train_df, testdata, m, perc=percentile)
            results[m].append(res)
        index += 1
    for m in metrics:
        results[m] = np.median(results[m])
    print("*"*50)
    print(method)
    print(results)
    print("*"*50)
    return results

def MLSSL(func, train, test, pname, biased_col):
    X_train, y_train = train
    X_test, y_test = test
    y_probabilities = []
    if func != "CT":
        if func == "LS":
            ssl_model = LabelSpreading(n_jobs=-1)
        elif func == "LP":
            ssl_model = LabelPropagation(n_jobs=-1)
        elif func == "ST":
            clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1)
            ssl_model = SelfTrainingClassifier(clf)
        print("finished training")
        # pdb.set_trace()
        ssl_model.fit(X_train, y_train)
        y_predicts = ssl_model.predict(X_test.values)
        for p in ssl_model.predict_proba(X_test.values):
            y_probabilities.append(max(p))
    elif func == "CT":
        N_FEATURES = X_train.shape[1]
        # pdb.set_trace()
        X1 = X_train.values[:, :N_FEATURES // 2]
        X2 = X_train.values[:, N_FEATURES // 2:]

        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1)
        ssl_model = CoTrainingClassifier(clf)
        ssl_model.fit(X1, X2, y_train.values)
        print("finished training")
        y_predicts = ssl_model.predict(X_test.values[:, :N_FEATURES // 2], X_test.values[:, N_FEATURES // 2:])
        for p in ssl_model.predict_proba(X_test.values[:, :N_FEATURES // 2], X_test.values[:, N_FEATURES // 2:]):
            y_probabilities.append(max(p))
    else:
        pdb.set_trace()
        print("wrong method")

    try:
        treatment = Treatment({}, "")
    except:
        treatment = Treatment(train, "")
    treatment.y_label = ["yes" if y == 1 else "no" for y in y_test]
    treatment.decisions = ["yes" if y == 1 else "no" for y in y_predicts]
    summary = collections.Counter(treatment.decisions)
    print(summary, summary["yes"] / (summary["yes"] + summary["no"]))
    # treatment.probs = y_probabilities
    # results = treatment.eval()
    copieddata = copy.deepcopy(X_test)
    copieddata['current_pred_' + biased_col] = y_predicts
    copieddata['Label'] = y_test
    confusion_result = confusion_matrix(y_test, y_predicts).ravel()
    if pname == 'german':
        results = get_counts_german_cla(confusion_result, copieddata, biased_col)
    else:
        results = get_counts_cla(confusion_result, copieddata, biased_col)
    return results

def execute(data, target, method="ST", seed=0):
    np.random.seed(seed)
    random.seed(seed)
    copieddata = copy.deepcopy(data)

    # sss = StratifiedShuffleSplit(n_splits=5, test_size=.5, random_state=seed)
    # X, y = traindata[traindata.columns[:-1]], traindata[traindata.columns[-1]]
    # recall, fpr, precision, f-score1
    # metrics = ["recall", "fall-out", "precision", "f1"]
    metrics = ["recall", "fall-out", "aod", "eod", "spd", "di", "f1", "accuracy", "precision"]
    results = {m: [] for m in metrics}
    pname, biased_column = target.split("_")
    for index in range(3):
        print("Iteration = ", index)
        dataset_orig_train, dataset_orig_vt = train_test_split(copieddata, test_size=0.3)
        dataset_orig_valid, testdata = train_test_split(copieddata, test_size=0.5)
        train_df, tune_df = train_test_split(dataset_orig_train, test_size=0.025)
        train_df['Label'] = [-1] * train_df.shape[0]

        traindf_curr = pd.concat([train_df, tune_df])
        X_train, y_train = traindf_curr.loc[:, traindf_curr.columns != 'Label'], traindf_curr['Label']
        X_test, y_test = testdata.loc[:, testdata.columns != 'Label'], testdata['Label']
        temp_res = MLSSL(method, [X_train, y_train], [X_test, y_test], pname, biased_column)
        for m in metrics:
            results[m].append(temp_res[m])
    for m in metrics:
        results[m] = np.median(results[m])
    print("*"*50)
    print(method)
    print(results)
    print("*"*50)
    return results

def tuning_frugal(data, target, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    traindata = copy.deepcopy(data["train"])
    testdata = copy.deepcopy(data[target])
    sss = StratifiedShuffleSplit(n_splits=5, test_size=.1, random_state=seed)
    X, y = traindata[traindata.columns[:-1]], traindata[traindata.columns[-1]]
    metrics = ["AUC", "fall-out", "recall", "accuracy"]
    results = {m: [] for m in metrics}
    index = 0
    treatments = ["CLA", "CLAMI", "CLASUP"]
    for train_index, tune_index in sss.split(X, y):
        print("Iteration = ", index)
        train_df = traindata.iloc[train_index]
        tune_df = traindata.iloc[tune_index]
        m_res = []
        percentile_res = {}
        for t in treatments:
            percentile_res[t] = percentile_tuning(t, train_df, tune_df)

        for m in metrics:
            for t in treatments:
                tmp_res = percentile_res[t]
                m_res += [[x[0][m], x[1], t] for x in tmp_res]
            m_res.sort(key=lambda x: x[0])
            percentile = m_res[-1][1] if m != "fall-out" else m_res[0][1]
            method = m_res[-1][2] if m != "fall-out" else m_res[0][2]
            res = run_method(method, train_df, testdata, m, perc=percentile)
            results[m].append(res)
        index += 1
    for m in metrics:
        res = np.array(results[m])
        results[m] = res[res!=0].mean()
    print("*"*50)
    print(method)
    print(results)
    print("*"*50)
    return results

def CLAMI_eval(trainingInstancesByCLAMI, newTestInstances, target, stats={"tp": 0, "p": 0}):
    results = []
    # treaments = ["LR", "SVM", "RF", "NB"]
    # treaments = ["RF", "NB"]
    treaments = ["LR"]
    for mlAlg in treaments:
        results.append(training_CLAMI(trainingInstancesByCLAMI, newTestInstances, target, mlAlg, stats=stats))
    return results[-1]


def MI(data, tunedata, selectedMetricIndices, percentileCutoff, positiveLabel, target):
    print(selectedMetricIndices)
    trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(data,
                                                                        selectedMetricIndices, True, label="CLA")
    newTuneInstances = getInstancesByRemovingSpecificAttributes(tunedata,
                                                                selectedMetricIndices, True, label="Label")
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
                                                              percentileCutoff, "Label")
    instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
                                                   cutoffsForHigherValuesOfAttribute,
                                                   positiveLabel)
    trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
                                                                       instIndicesNeedToRemove, False)
    zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
    one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
    if zero_count > 0 and one_count > 0:
        return selectedMetricIndices, training_CLAMI(trainingInstancesByCLAMI, newTuneInstances, target, "RF")
    else:
        return -1, -1


def transform_metric_indices(shape, indices):
    array = np.array([0] * shape)
    array[indices] = 1
    return array


def tune_CLAMI(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, metric="APFD"):
    treatment = Treatment(data, target)
    treatment.preprocess()
    data = treatment.full_train
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=47)
    testdata = treatment.full_test
    X, y = data[data.columns[:-1]], data[data.columns[-1]]
    for train_index, tune_index in sss.split(X, y):
        train_df = data.iloc[train_index]
        tune_df = data.iloc[tune_index]
        train_df.reset_index(drop=True, inplace=True)
        tune_df.reset_index(drop=True, inplace=True)
        cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(train_df, percentileCutoff, "Label")
        print("get cutoffs")
        train_df = getInstancesByCLA(train_df, percentileCutoff, positiveLabel)
        print("get CLA instances")

        metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(train_df,
                                                                                     cutoffsForHigherValuesOfAttribute,
                                                                                     positiveLabel)
        # pdb.set_trace()
        keys = list(metricIdxWithTheSameViolationScores.keys())
        # keys.sort()
        evaluated_configs = random.sample(keys, INIT_POOL_SIZE * 2)
        evaluated_configs = [metricIdxWithTheSameViolationScores[k] for k in evaluated_configs]

        tmp_scores = []
        tmp_configs = []
        for selectedMetricIndices in evaluated_configs:
            selectedMetricIndices, res = MI(train_df, tune_df, selectedMetricIndices,
                                            percentileCutoff, positiveLabel, target)
            if isinstance(res, dict):
                tmp_configs.append(transform_metric_indices(data.shape[1], selectedMetricIndices))
                tmp_scores.append(res)

        ids = np.argsort([x[metric] for x in tmp_scores])[::-1][:1]
        best_res = tmp_scores[ids[0]]
        best_config = np.where(tmp_configs[ids[0]] == 1)[0]

        # number of eval
        this_budget = BUDGET
        eval = 0
        lives = 5
        print("Initial Population: %s" % len(tmp_scores))
        searchspace = [transform_metric_indices(data.shape[1], metricIdxWithTheSameViolationScores[k])
                       for k in keys]
        while this_budget > 0:
            cart_model = DecisionTreeRegressor()
            cart_model.fit(tmp_configs, [x[metric] for x in tmp_scores])

            cart_models = []
            cart_models.append(cart_model)
            next_config_id = acquisition_fn(searchspace, cart_models)
            next_config = metricIdxWithTheSameViolationScores[keys.pop(next_config_id)]
            searchspace.pop(next_config_id)
            next_config, next_res = MI(train_df, tune_df,
                                       next_config, percentileCutoff,
                                       positiveLabel, target)
            if not isinstance(next_res, dict):
                continue

            next_config_normal = transform_metric_indices(data.shape[1], next_config)
            tmp_scores.append(next_res)
            tmp_configs.append(next_config_normal)
            try:
                if abs(next_res[metric] - best_res[metric]) >= 0.03:
                    lives = 5
                else:
                    lives -= 1

                # pdb.set_trace()
                if isBetter(next_res, best_res, metric):
                    best_config = next_config
                    best_res = next_res

                if lives == 0:
                    print("***" * 5)
                    print("EARLY STOPPING!")
                    print("***" * 5)
                    break

                this_budget -= 1
                eval += 1
            except:
                pdb.set_trace()
    _, res = MI(train_df, testdata, best_config, percentileCutoff, positiveLabel, target)
    return res


def training_CLAMI(trainingInstancesByCLAMI, newTestInstances, target, model, all=True, stats={"tp": 0, "p": 0}):
    try:
        newTestInstances.drop(["CLA", "K"], axis=1, inplace=True)
    except:
        pass
    treatments = {"RF": RF, "SVM": SVM, "LR": LR, "NB": NB, "DT": DT, "TM": TM}
    treatment = treatments[model]
    clf = treatment({}, "")
    # print(target, model)
    clf.test_data = newTestInstances[newTestInstances.columns.difference(['Label'])].values
    clf.y_label = np.array(["yes" if x == 1 else "no" for x in newTestInstances["Label"].values])
    try:
        clf.train_data = trainingInstancesByCLAMI.values[:, :-1]
        clf.x_label = np.array(["yes" if x == 1 else "no" for x in trainingInstancesByCLAMI['Label']])
        clf.train()
        clf.stats = stats

        summary = collections.Counter(clf.decisions)
        print(summary, summary["yes"] / (summary["yes"] + summary["no"]))
        results = clf.eval()
        results["read"] = summary["yes"] / (summary["yes"] + summary["no"])
        if all:
            return results
        else:
            return results["APFD"] + results["f1"]
    except:
        # pdb.set_trace()
        results = {"precision": 0, "recall": 0, "f1": 0, "fall-out": 1, "g1": 0}
        return results


def getViolationScores(data, labels, cutoffsForHigherValuesOfAttribute, key=-1):
    '''
	get violation scores
	----------
	data
	labels
	cutoffsForHigherValuesOfAttribute
	key

	Returns
	-------

	'''
    violation_score = 0
    if key not in ["Label", "K", "CLA"]:
        if key != -1:
            # violation score by columns
            categories = labels.values
            cutoff = cutoffsForHigherValuesOfAttribute[key]
            # violation: less than a median and class = 1 or vice-versa
            violation_score += np.count_nonzero(np.logical_and(categories == 0, np.greater(data.values, cutoff)))
            violation_score += np.count_nonzero(np.logical_and(categories == 1, np.less_equal(data.values, cutoff)))
        else:
            # violation score by rows
            row = data.values
            row_data, row_label = row[:-1], row[-1]
            # violation: less than a median and class = 1 or vice-versa

            row_label_0 = np.array(row_label == 0).tolist() * row_data.shape[0]
            # randomness = random.random()
            # if randomness > 0.5:
            violation_score += np.count_nonzero(np.logical_and(row_label_0,
                                                               np.greater(row_data, cutoffsForHigherValuesOfAttribute)))
            row_label_1 = np.array(row_label == 0).tolist() * row_data.shape[0]
            violation_score += np.count_nonzero(np.logical_and(row_label_1,
                                                               np.less_equal(row_data,
                                                                             cutoffsForHigherValuesOfAttribute)))

    # for attrIdx in range(data.shape[1] - 3):
    # 	# if attrIdx not in ["Label", "CLA"]:
    # 	attr_data = data[attrIdx].values
    # 	cutoff = cutoffsForHigherValuesOfAttribute[attrIdx]
    # 	violations.append(getViolationScoreByColumn(attr_data, data["Label"], cutoff))
    return violation_score


def acquisition_fn(search_space, cart_models):
    vals = []
    predicts = []
    ids = []
    ids_only = []
    for cart_model in cart_models:
        predicted = cart_model.predict(search_space)
        predicts.append(predicted)
        ids.append(np.argsort(predicted)[::1][:1])
    for id in ids:
        val = [pred[id[0]] for pred in predicts]
        vals.append(val)
        ids_only.append(id[0])

    return bazza(ids_only, vals)


def bazza(config_ids, vals, N=20):
    dim = len(vals)
    rand_vecs = [[np.random.uniform() for i in range(dim)] for j in range(N)]
    min_val = 9999
    min_id = 0
    for config_id, val in zip(config_ids, vals):
        projection_val = 0
        for vec in rand_vecs:
            projection_val += np.dot(vec, val)
        mean = projection_val / N
        if mean < min_val:
            min_val = mean
            min_id = config_id

    return min_id


def isBetter(new, old, metric):
    if metric == "d2h":
        return new[metric] < old[metric]
    else:
        return new[metric] > old[metric]


def getMetricIndicesWithTheViolationScores(data, cutoffsForHigherValuesOfAttribute, positiveLabel, label="Label"):
    '''
	get all the features that violated the assumption
	----------
	data
	cutoffsForHigherValuesOfAttribute
	positiveLabel

	Returns
	-------

	'''
    # cutoffs for all the columns/features
    # pdb.set_trace()
    # cutoffsForHigherValuesOfAttribute = {i: x for i, x in enumerate(cutoffsForHigherValuesOfAttribute)}
    cutoffsForHigherValuesOfAttribute = {x: y for x, y in zip(data.columns, cutoffsForHigherValuesOfAttribute)}
    # use pandas apply per column to find the violation scores of all the features
    violations = data.apply(
        lambda col: getViolationScores(col, data[label],
                                       cutoffsForHigherValuesOfAttribute,
                                       key=col.name),
        axis=0)
    violations = violations.values
    metricIndicesWithTheSameViolationScores = collections.defaultdict(list)

    # store the violated features that share the same violation scores together
    for attrIdx in range(data.shape[1] - 3):
        key = violations[attrIdx]
        metricIndicesWithTheSameViolationScores[key].append(data.columns[attrIdx])
    return metricIndicesWithTheSameViolationScores


def plot_recall_cost(which="SE_overall"):
    '''
	draw the recall cost curve for all the methods
	----------
	which

	Returns
	-------

	'''
    paths = ["../dump/SE_JITCLA_result.pickle", "../dump/SE_est_result.pickle"]
    temp_results = []
    for p in paths:
        with open(p, "rb") as f:
            temp_results.append(pickle.load(f))
    keys = list(temp_results[0].keys())
    keys.sort()
    print(keys)
    results = {}
    for target in keys:
        if target != "Treatment":
            results[target] = {"CLA+HARD": temp_results[0][target][0],
                               "Jitterbug": temp_results[1][target][0]}

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}

    plt.rcParams.update(paras)

    lines = ['-', ':', '--', (0, (4, 2, 1, 2)), (0, (3, 2)), (0, (2, 1, 1, 1))]

    for project in results:
        fig = plt.figure()
        for i, treatment in enumerate(results[project]):
            plt.plot(results[project][treatment]["CostR"], results[project][treatment]["TPR"], linestyle=lines[i],
                     label=treatment)
        plt.legend()
        plt.ylabel("Recall")
        plt.xlabel("Cost")
        plt.grid()
        plt.savefig("../figures/" + project + ".png")
        plt.close(fig)


def read_stats_results(file="../dump/SE_CLA+HARD_Full_SATD_stats.pickle"):
    result = pickle.load(open(file, "rb"))
    datasets = ["apache-jmeter-2.10", "jruby-1.4.0", "hibernate-distribution-3.3.2.GA",
                "emf-2.4.1", "apache-ant-1.7.0", "sql12",
                "columba-1.4-src", "argouml", "jfreechart-1.0.19", "jEdit-4.2"]
    satds = {"apache-ant-1.7.0": 13, 'apache-jmeter-2.10': 695, 'argouml': 1779, 'columba-1.4-src': 771,
             'emf-2.4.1': 113, 'hibernate-distribution-3.3.2.GA': 277, 'jEdit-4.2': 520,
             'jfreechart-1.0.19': 153, 'jruby-1.4.0': 109, 'sql12': 947}
    # satds = {"apache-jmeter-2.10": 416, "jruby-1.4.0": 665, "hibernate-distribution-3.3.2.GA": 493,
    # 		 "emf-2.4.1": 119, "apache-ant-1.7.0": 135, "sql12": 313, "columba-1.4-src": 220,
    # 		 "jEdit-4.2": 259, "jfreechart-1.0.19": 247, "argouml": 1630}

    for target in datasets:
        actuals, estimates = result[target]['actual_td'], result[target]['estimated_td']
        ratios = []
        actual_ratios = []
        rate_of_changes = []
        index = 0
        for a, e in zip(actuals, estimates):
            if a != 0 and e != 0:
                ratios.append(float(a) / e)
            if index > 0:
                try:
                    rate_of_changes.append((e - estimates[index - 1]) / estimates[index - 1])
                except:
                    rate_of_changes.append(0)
            if e != 0:
                actual_ratios.append(float(e) / satds[target])
            index += 1

        # print(len(ratios))
        print(round(np.percentile(actual_ratios, 50) * 100), end=", ")
    # print(round(np.percentile(ratios, 50) * 100), end=", ")

    # print(round(np.percentile(rate_of_changes, 100) * 100), end=", ")


if __name__ == "__main__":
    eval(cmd())
# if __name__ == '__main__':
#     intrinsic_dim()