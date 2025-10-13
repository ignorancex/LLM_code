import os
import numpy as np
import json

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from mass_model import run_hipmk


# Converts column info into stats for handling different data types
def stats_convert(dataset):
    column_info = load_data_stats(dataset)
    if all(col == "numerical" for col in column_info.values()):
        return None

    stats = {"attribute": []}
    for column_name, column in column_info.items():
        col_dic = {'type': 'Numeric' if column == "numerical" else ''}
        if isinstance(column, dict):
            key = next(iter(column))
            if key in {"ordinal", "nominal"}:
                col_dic['type'] = key.capitalize()
                col_dic['values'] = list(column[key].values())
        stats["attribute"].append(col_dic)
    return stats

# Main function to run models on datasets with missing data
def run(dataset, missing_type, model, missing_rates, y, clustering=False):
    na_path = f"dataset_nan/{dataset}/{missing_type}/"

    if clustering:
        data_stats = load_data_stats(dataset)
    else:
        data_stats = stats_convert(dataset)


    all_results = {}

    if not clustering:
        for rate in missing_rates:
            data_na = np.load(f"{na_path}{rate}.npy")
            skf = StratifiedKFold(n_splits=2)
            results_list = [run_model(model, data_na[trn], data_na[test], y[trn], y[test], data_stats)
                            for trn, test in skf.split(data_na, y)]

            all_results[rate] = aggregate_results(results_list)
        return all_results

    else:
        data_na = np.load(f"dataset/{dataset}/feature.npy")
        results_list = []
        for i in range(1):
            result = run_clustering_model(model, data_na, data_na, y, y, data_stats)
            print(result)
            results_list.append(result)
        return aggregate_results(results_list, clustering=True)

# Loads column info JSON for specific datasets
def load_data_stats(dataset):
    with open(f"dataset/{dataset}/column_info.json", 'r') as file:
        return json.load(file)

# Runs a model for a classification task
def run_model(model, X_train, X_test, y_train, y_test, data_stats):

    train, test = run_hipmk(X_train, X_test, data_stats)
    return SVC_evaluation(train, y_train, test, y_test, kernel="precomputed")


# Evaluates an SVC model and returns accuracy and F1 score
def SVC_evaluation(X_train, y_train, X_test, y_test, kernel="rbf"):
    svc = SVC(C=1, kernel=kernel)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='macro')
    }

# Aggregates the results from multiple folds
def aggregate_results(results_list, clustering=False):
    if not clustering:
        avg_accuracy = np.mean([res['accuracy'] for res in results_list])
        avg_f1_score = np.mean([res['f1_score'] for res in results_list])
        return {
            "avg_accuracy": avg_accuracy,
            "std_accuracy": np.std([res['accuracy'] for res in results_list]),
            "avg_f1_score": avg_f1_score,
            "std_f1_score": np.std([res['f1_score'] for res in results_list])
        }
    else:
        avg_kmeans_nmi = np.mean([res['KMeans_nmi'] for res in results_list])
        avg_kmeans_ari = np.mean([res['KMeans_ari'] for res in results_list])
        return {
            "avg_kmeans_nmi": avg_kmeans_nmi,
            "std_kmeans_nmi": np.std([res['KMeans_nmi'] for res in results_list]),
            "avg_kmeans_ari": avg_kmeans_ari,
            "std_kmeans_ari": np.std([res['KMeans_ari'] for res in results_list])
        }

# Runs a clustering model
def run_clustering_model(model, X_train, X_test, y_train, y_test, data_stats):
    train, test = run_hipmk(X_train, X_test, data_stats)

    result = clustering_evaluation(train, y_train, test, y_test)
    return result


# Evaluates clustering using KMeans and returns NMI and ARI scores
def clustering_evaluation(X_train, y_train, X_test, y_test):
    kmeans = KMeans(n_clusters=len(np.unique(y_train)))
    kmeans.fit(X_train)
    y_pred_kmeans = kmeans.predict(X_test)
    result = {
        "KMeans_nmi": normalized_mutual_info_score(y_test, y_pred_kmeans),
        "KMeans_ari": adjusted_rand_score(y_test, y_pred_kmeans)
    }
    return result
