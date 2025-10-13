import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tsai.models.FCN import FCN
from tsai.models.InceptionTime import InceptionTime

from utils import pickle_save_to_file, plot_implet_clusters_with_instances
from utils.data_utils import read_UCR_UEA
from utils.implet_extactor import implet_extractor, implet_cluster_auto, implet_cluster
from utils.insert_shapelet import insert_random, overwrite_shaplet_random
from utils.constants import tasks_new

k = 2
verbose = True
device = torch.device("cpu")

model_names = ['FCN' ]  #, 'FCN', 'InceptionTime'
tasks = [
    'GunPoint']  #tasks_new  #tasks_new #  #['GunPoint'] # , "Strawberry", "ECG200", "DistalPhalanxOutlineCorrect", "PowerCons", "Earthquakes",
xai_names = ['Saliency']  #  'GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion',

# each row is [model_name, task_name, xai_name, method, acc_score]
# method is in ['ori', 'repl_implet', 'repl_random_loc']

is_attr_abs = True
is_vis_implet = False
is_clustering = True
is_global_threshold = False

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
        _, test_x, _, test_y, _ = read_UCR_UEA(task, None)
        test_y = np.argmax(test_y, axis=1)

        for explainer in xai_names:
            # load attributions
            with open(f'attributions/{model_name}/{task}/{explainer}/test_exp.pkl', 'rb') as f:
                attr = pickle.load(f)
            attr_test = attr['attributions']
            if task == 'Chinatown':
                kmin = 2
            else:
                kmin = None
            # compute implets
            implets_class0 = implet_extractor(test_x, test_y, attr_test, target_class=0, is_attr_abs=is_attr_abs,
                                              kmin=kmin, is_global_threshold=is_global_threshold)
            implets_class1 = implet_extractor(test_x, test_y, attr_test, target_class=1, is_attr_abs=is_attr_abs,
                                              kmin=kmin, is_global_threshold=is_global_threshold)
            implets = implets_class0 + implets_class1
            print(f'number of the implets {len(implets)}')
            implets_list = {
                'implets_class0': implets_class0,
                'implets_class1': implets_class1,
            }
            # implets_save_dir = f'./output/{model_name}/{task}/{explainer}' if is_attr_abs else f'./output/no_abs/{model_name}/{task}/{explainer}'
            implets_save_dir = f'./output/{model_name}/{task}/{explainer}' if k is None else f'./output/k{k}/{model_name}/{task}/{explainer}'
            # print(implets_save_dir)
            pickle_save_to_file(data=implets_list,
                                file_path=os.path.join(implets_save_dir, 'implets.pkl'))

            # print(implet_with_attr, instances_num)

            for implate_name, implets_class_i in implets_list.items():
                num_implets = len(implets_class_i)
                if is_vis_implet:
                    if len(implets_class_i) == 0:
                        continue
                    instances_num = list(set([imp[0] for imp in implets_class_i]))
                    instances_num.sort()
                    # print(num_implets, implets_class_i, implets_class_i[0][1].shape, implets_class_i[0][2].shape)
                    save_path = f'./{implets_save_dir}/{implate_name}_vis.png'
                    plot_implet_clusters_with_instances(implets_class_i, test_x[instances_num], save_path=save_path,
                                                        title=f"Number of Implet {len(implets_class_i)}")

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
                    pickle_save_to_file(implet_cluster_results,
                                        os.path.join(implets_save_dir, f'{implate_name}_cluster_results.pkl'))
