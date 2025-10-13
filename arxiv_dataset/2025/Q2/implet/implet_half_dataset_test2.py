import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tsai.models.FCN import FCN
from tsai.models.InceptionTime import InceptionTime
from dtaidistance import dtw_ndim
from dtaidistance.subsequence.dtw import subsequence_alignment

from utils import pickle_load_from_file
from utils.implet_extactor import implet_extractor
from utils.insert_shapelet import insert_random, overwrite_shaplet_random

device = torch.device("cpu")

model_names = ['FCN', 'InceptionTime']
tasks = ['GunPoint', "ECG200", "DistalPhalanxOutlineCorrect", "PowerCons", "Earthquakes",
         "Strawberry"]
# xai_names = ['GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion', 'Saliency']
xai_names = ['Saliency']
blur_modes = ['poly', 'zero', 'mean', 'gauss', 'slidingwindow', 'lowpass', 'highpass']

# each row is [model_name, task_name, xai_name, method, implet_src, mode, acc_score]
# method is in ['ori', 'repl_implet', 'repl_random_loc']
result_path = f'output/half_dataset_test_blur_modes.csv'
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

        for explainer in xai_names:
            centroids = []  # 1D centroid values
            thresholds = []  # intra-cluster distance thresholds
            xs = None  # samples in the second half of the testing dataset
            ys = None  # labels in the second half of the testing dataset
            attrs = None  # attr of the samples in the second half of the tesing dataset

            for cls in range(2):
                data = pickle_load_from_file(
                    f'./output/half_implet/{model_name}/{task}/{explainer}/implets_class{cls}_cluster_results.pkl')

                if xs is None:
                    xs = data['second_half_x']
                    xs = xs.squeeze()

                    ys = data['second_half_y']
                    ys = ys.squeeze()

                    attrs = data['first_half_attr']
                    attrs = attrs.squeeze()

                # 1D centroid values
                _centroids = data['best_centroids_dep']
                if _centroids:
                    _centroids = [c[:, 0] for c in _centroids]
                else:
                    _centroids = []

                centroids += _centroids

                # intra-cluster distances & thresholds
                for j in range(len(_centroids)):
                    c = _centroids[j]
                    dists = []
                    for i in data['best_indices_dep'][j]:
                        implet = data['implets'][i][1]
                        dist = dtw_ndim.distance(c, implet)
                        dists.append(dist)

                    # set thresholds
                    thresh = np.mean(dists) + 2 * np.std(dists)
                    thresholds.append(thresh)

            # compute real implets
            implets_real = []
            for cls in range(2):
                implets_real += implet_extractor(xs, ys, attrs, target_class=cls)

            print(np.mean([len(implet[1].flatten()) for implet in implets_real]))
