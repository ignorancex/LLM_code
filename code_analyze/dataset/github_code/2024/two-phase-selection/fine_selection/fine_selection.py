import numpy as np
import pandas as pd
import json
from sklearn import metrics
from sklearn.cluster import KMeans


def create_cluster_model(test_res,val_res,target_model, target_dataset, target_epoch, K=4):
    X_train = []  
    y_train = []  

    for i,dataset in enumerate(val_res[target_model].keys()):
        if dataset == target_dataset:
            continue
        X_train.append(val_res[target_model][dataset][target_epoch])
        y_train.append(test_res[target_model][i])
    X_train = np.array(X_train).reshape(-1,1)

    kmeans = KMeans(n_clusters=K)
    labels = kmeans.fit_predict(X_train)

    cluster_means = {}
    for i in range(K):
        cluster_means[i] = np.mean([y for y, label in zip(y_train, labels) if label == i])

    return kmeans,cluster_means


def filter_models_sh(test_res, val_res, epoch_num = 5, num_models=10, recalled_models = {}):
    results = {}  
    dataset_list = list( val_res[ list( val_res.keys() ) [0] ].keys() )

    for dataset_index,target_dataset in enumerate(dataset_list,recalled_models = {}):
        # Filtering the recalled models
        if target_dataset in recalled_models:
            model_performance = {model: performance[dataset_index] for model, performance in test_res.items() if model in recalled_models[target_dataset]}
        #Filetering the top-k models
        else:
            model_performance = {model: performance[dataset_index] for model, performance in test_res.items()}

        top_models = sorted(model_performance, key=model_performance.get, reverse=True)[:num_models]
        results[target_dataset] = {}
        for epoch in range(epoch_num): 
            cur_num_models = len(top_models)
            while len(top_models) > cur_num_models  // 2:
                worst_model = min(top_models, key=lambda model: val_res[model][target_dataset][epoch])
                top_models.remove(worst_model)

            results[target_dataset][epoch] = {
                "left_num_models": len(top_models),
                "best_test_performance": max(test_res[model][dataset_index] for model in top_models)
            }

            if len(top_models) == 1:
                break

    return results

#
def filter_models_fs(test_res, val_res, epoch_num = 5, num_models=10, threshold=0.0, recalled_models = {}):
    results = {} 
    dataset_list = list( val_res[ list( val_res.keys() ) [0] ].keys() )
    for dataset_index,target_dataset in enumerate(dataset_list):
        # Filtering the recalled models
        if target_dataset in recalled_models:
            model_performance = {model: performance[dataset_index] for model, performance in test_res.items() if model in recalled_models[target_dataset]}
        #Filetering the top-k models
        else:
            model_performance = {model: performance[dataset_index] for model, performance in test_res.items()}

        top_models = sorted(model_performance, key=model_performance.get, reverse=True)[:num_models]
        results[target_dataset] = {}
        for epoch in range(epoch_num):  
            cur_num_models = len(top_models)
            # For each model, creating clustering models and predict its final test performance
            predictions = {}
            for target_model in top_models:
                cluster_model, cluster_means = create_cluster_model(test_res, val_res,target_model, target_dataset,epoch)
                val_accuracy = val_res[target_model][target_dataset][epoch]
                val_accuracy = np.array([val_accuracy]).reshape(-1,1)
                prediction = cluster_model.predict(val_accuracy)
                predictions[target_model] = cluster_means[prediction[0]]

            # Start from the worst performing model.
            # If its predicted performance worse than that of any model with better val performance. Filter it.
            max_per = max(val_res[x][target_dataset][epoch] for x in predictions)
            for i,target_model in enumerate(sorted(predictions, key=lambda x: val_res[x][target_dataset][epoch])):
                if val_res[target_model][target_dataset][epoch]==max_per:
                    continue
                if predictions[target_model] < max(predictions[model] for model in top_models if val_res[model][target_dataset][epoch] > val_res[target_model][target_dataset][epoch])-threshold:
                    top_models.remove(target_model)

#                 # Forcing filtering at least half of the models
            while len(top_models) > cur_num_models  // 2:
                worst_model = min(top_models, key=lambda model: val_res[model][target_dataset][epoch])
                top_models.remove(worst_model)
            
            # Saving results
            results[target_dataset][epoch] = {
                "left_num_models": len(top_models),# How many models left after this filtering
                "best_test_performance": max(test_res[model][dataset_index] for model in top_models)# The best test performance in the remaining models
            }

            # Exit the loop when only one model left
            if len(top_models) == 1:
                break  
    return results


if __name__ == '__main__':
    #1.read history results file (stored in .npy)

    #val_res = {model1: {dataset1:[],dataset2:[]}, model:{} }
    val_path = ""
    val_res = np.load(val_path, allow_pickle = True).item()

    #test_res = {model1:[], model2:[]}
    test_path = ""
    test_res = np.load(test_path, allow_pickle = True).item()

    #2.For every model and every dataset, get the Successive_halving results
    sh_results = filter_models_sh(test_res,val_res)

    #3.For every model and every dataset, get the Fine-Selection results
    fs_results = filter_models_fs(test_res,val_res)