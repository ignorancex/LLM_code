import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tsai.models.FCN import FCN
from tsai.models.InceptionTime import InceptionTime

from utils import pickle_save_to_file, pickle_load_from_file
from utils.data_utils import read_UCR_UEA
from utils.implet_extactor import implet_extractor, implet_cluster_auto, implet_cluster
from utils.visualization import plot_implet_clusters_with_instances
from utils.constants import tasks_new


device = torch.device("cpu")

model_names = ['FCN']
tasks = tasks_new # , "ECG200", "DistalPhalanxOutlineCorrect", "PowerCons", "Earthquakes", "Strawberry"

xai_names = ['Saliency', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion' ] # , , 'Saliency'

k = None
is_clustering = True
verbose = True
for model_name in model_names:
    print(f'======== {model_name} ========')

    for task in tasks:
        print(f'-------- {task} --------')

        # load model
        if model_name == 'FCN':
            model = FCN(c_in=1, c_out=2)
        elif model_name == 'InceptionTime':
            model = InceptionTime(c_in=1, c_out=2)
        else:
            raise ValueError

        model_path = f'models/{model_name}/{task}/weight.pt'
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        def predict(x):
            if len(x.shape) == 1:
                x = x[np.newaxis, np.newaxis, :]
            elif len(x.shape) == 2:
                x = x[np.newaxis]

            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x).to(device)

            logits = model(x).detach().cpu().numpy()
            return np.argmax(logits, axis=-1)
        # load dataset
        _, x_test, _, y_test, _ = read_UCR_UEA(task, None)
        y_test = np.argmax(y_test, axis=1)

        # run prediction on unmodified samples
        y_pred = predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        entry = [model_name, task, None, 'ori', acc]
        print(entry)

        for explainer in xai_names:
            _, test_x, _, test_y, _ = read_UCR_UEA(task, None)
            # load attributions
            with open(f'attributions/{model_name}/{task}/{explainer}/test_exp.pkl', 'rb') as f:
                attr = pickle.load(f)
            attr_test = attr['attributions']

            # compute implets
            implets_class0 = implet_extractor(x_test, y_test, attr_test, target_class=0)
            implets_class1 = implet_extractor(x_test, y_test, attr_test, target_class=1)
            implets = implets_class0 + implets_class1

            implets_list = {
                'implets': implets,
                'implets_class0': implets_class0,
                'implets_class1': implets_class1,
            }
            implets_save_dir = f'./output/{model_name}/{task}/{explainer}'
            pickle_save_to_file(data=implets,
                                file_path=os.path.join(implets_save_dir, 'implets.pkl'))

            # implets_class0 = implets_list['implets_class0']
            # implets_class1 = implets_list['implets_class1']


            for implate_name, implets_class_i in implets_list.items():
                num_implets = len(implets_class_i)
                if is_clustering:
                    # clustering
                    # 2d DTW
                    if verbose:
                        print('computing dependent DTW')
                    implet_with_attr = [np.vstack((imp[1], imp[2])).T for imp in implets_class_i]

                    if k is None:
                        best_k_dep, best_indices_dep, best_centroids_dep = implet_cluster_auto(implet_with_attr, None)
                    else:
                        best_k_dep = k
                        best_indices_dep, best_centroids_dep = implet_cluster(implet_with_attr, k)

                    # 1d DTW
                    if verbose:
                        print('computing 1D DTW')
                    implet_itself = [imp[1] for imp in implets_class_i]
                    # if len(implet_itself) > 100:
                    #     implet_itself = random.sample(implet_itself, 100)
                    if k is None:
                        best_k_1d, best_indices_1d, best_centroids_1d = implet_cluster_auto(implet_itself, None)
                    else:
                        best_k_1d = k
                        best_indices_1d, best_centroids_1d = implet_cluster(implet_itself, k)
                    # if is_plot:
                    #     plot_implet_clusters(implet_with_attr, best_indices_1d, best_centroids_1d,
                    #                          save_path=os.path.join(implets_save_dir, 'cluster_1dDTW.png'))
                    implet_cluster_results = {
                        'implets': implets_class_i,
                        'num_implets': num_implets,
                        'best_k_dep': best_k_dep,
                        'best_indices_dep': best_indices_dep,
                        'best_centroids_dep': best_centroids_dep,
                        'best_k_1d': best_k_1d,
                        'best_indices_1d': best_indices_1d,
                        'best_centroids_1d': best_centroids_1d,
                    }
                    pickle_save_to_file(implet_cluster_results, os.path.join(implets_save_dir, f'{implate_name}_cluster_results.pkl'))
                else:
                    cluster_path = os.path.join(implets_save_dir, f'{implate_name}_cluster_results.pkl')
                    print(cluster_path)
                    implet_cluster_results = pickle_load_from_file(cluster_path)
                    best_indices_dep = implet_cluster_results['best_indices_dep']
                    best_centroids_dep = implet_cluster_results['best_centroids_dep']
                    best_k_dep = implet_cluster_results['best_k_dep']

                    # print(best_indices_dep,best_centroids_dep,best_k_dep)
                    # print(implets_save_dir, implate_name, best_k_dep)
                    # print(implet_cluster_results['best_centroids_dep'])
                if best_k_dep is None:
                    continue

                for clsuster_i in range(best_k_dep):


                    implet_indexes = list(best_indices_dep[clsuster_i])
                    # print(implet_indexes)
                    implet_cluster = [implets_class_i[idx] for idx in implet_indexes]

                    # implet_with_attr = [np.vstack((imp[1], imp[2])).T for imp in implet_cluster]

                    instances_num = list(set([imp[0] for imp in implet_cluster]))
                    instances_num.sort()
                    # print(implet_with_attr, instances_num)
                    save_path = f'./figure/cluster_result/half_cluster/{model_name}/{explainer}/{task}/{implate_name}_clsuter{clsuster_i}.png'
                    plot_implet_clusters_with_instances(implet_cluster, test_x[instances_num], save_path=save_path)