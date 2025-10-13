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

k = None
verbose = True
device = torch.device("cpu")

model_names = ['FCN','InceptionTime'] #,
tasks = ['GunPoint', "Strawberry", "ECG200", "DistalPhalanxOutlineCorrect", "PowerCons", "Earthquakes",
         ] #
xai_names = ['GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion', 'Saliency' ] #

np.random.seed(42)
# each row is [model_name, task_name, xai_name, method, acc_score]
# method is in ['ori', 'repl_implet', 'repl_random_loc']
result_path = 'output/blurring_test.csv'
is_attr_abs = True
is_vis_implet = False
is_clustering = True
is_global_threshold = False
if os.path.isfile(result_path):
    result = pd.read_csv(result_path).values.tolist()
else:
    result = []

n_trials = 10

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
        print(test_x.shape)
        indices = np.random.permutation(len(test_x))

        # Apply the same shuffle to both arrays
        test_x = test_x[indices]
        test_y = test_y[indices]
        first_half_x = test_x[:len(test_x) // 2]
        second_half_x = test_x[len(test_x) // 2:]

        first_half_y = test_y[:len(test_y) // 2]
        second_half_y = test_y[len(test_y) // 2:]

        for explainer in xai_names:
            # load attributions
            with open(f'attributions/{model_name}/{task}/{explainer}/test_exp.pkl', 'rb') as f:
                attr = pickle.load(f)
            attr_test = attr['attributions']

            attr_test = attr_test[indices]
            first_half_attr = attr_test[len(attr_test) // 2:]
            second_half_attr = attr_test[:len(attr_test) // 2]
            # print(first_half_y.shape, second_half_y.shape, first_half_x.shape, second_half_x.shape,first_half_attr.shape,second_half_attr.shape,)
        #
            # compute implets
            implets_class0 = implet_extractor(first_half_x, first_half_y, first_half_attr, target_class=0, is_attr_abs=True, is_global_threshold=False)
            implets_class1 = implet_extractor(second_half_x, second_half_y, first_half_attr, target_class=1, is_attr_abs=True, is_global_threshold=False)
            implets = implets_class0 + implets_class1

            implets_list = {
                'implets_class0': implets_class0,
                'implets_class1': implets_class1,
            }
            implets_save_dir = f'./output/half_implet/{model_name}/{task}/{explainer}'
            pickle_save_to_file(data=implets_list,
                                file_path=os.path.join(implets_save_dir, 'implets.pkl'))
        #
        #     # print(implet_with_attr, instances_num)
        #
        #
        #
            for implate_name, implets_class_i in implets_list.items():
                num_implets = len(implets_class_i)
                if is_vis_implet:
                    if len(implets_class_i) == 0:
                        continue
                    instances_num = list(set([imp[0] for imp in implets_class_i]))
                    instances_num.sort()
                    # print(num_implets, implets_class_i, implets_class_i[0][1].shape, implets_class_i[0][2].shape)
                    save_path = f'./{implets_save_dir}/{implate_name}_vis.png'
                    plot_implet_clusters_with_instances(implets_class_i, test_x[instances_num], save_path=save_path, title= f"Number of Implet {len(implets_class_i)}")

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

                        'first_half_x': first_half_x,
                        'second_half_x': second_half_x,

                        'first_half_y': first_half_y,
                        'second_half_y': second_half_y,

                        'first_half_attr': first_half_attr,
                        'second_half_attr': second_half_attr,

                    }
                    pickle_save_to_file(implet_cluster_results, os.path.join(implets_save_dir, f'{implate_name}_cluster_results.pkl'))

