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
from utils.constants import tasks_new,tasks

device = torch.device("cpu")

model_names = ['FCN', 'InceptionTime']
tasks_new.remove('Chinatown')
tasks = tasks_new+tasks #tasks_new #['GunPoint', "ECG200", "DistalPhalanxOutlineCorrect", "PowerCons", "Earthquakes",
#"Strawberry"]
# xai_names = ['GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion', 'Saliency']
xai_names = ['Saliency']

# each row is [model_name, task_name, xai_name, method, implet_src, mode, acc_score]
# method is in ['ori', 'repl_implet', 'repl_random_loc']
result_path = f'output/half_dataset_test_new.csv'
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
                x = x[:, np.newaxis]

            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x).to(device)

            logits = model(x).detach().cpu().numpy()
            return np.argmax(logits, axis=-1)


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

                    attrs = data['second_half_attr'] # Ziwen: data['first_half_attr'] 03/12/2025 this has been corrected
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
                print(xs.shape, ys.shape, attrs.shape)
                implets_real += implet_extractor(xs, ys, attrs, target_class=cls)

            # identify "implets" in the second half of the dataset
            # they should be similar to cluster centroids of the first half
            identified_implets_dict = {j: [] for j in range(len(centroids))}

            for j in range(len(centroids)):
                for i, x in enumerate(xs):
                    min_len = 3
                    max_len = x.shape[0] // 2
                    subseq_alignment = subsequence_alignment(centroids[j], x)
                    match = subseq_alignment.best_match()
                    l, r = match.segment
                    subseq_len = r - l + 1
                    if match.distance <= thresholds[j] and min_len <= subseq_len <= max_len:
                        identified_implets_dict[j].append((
                            i,  # instance index in the second half
                            x[l:r + 1],  # subsequence
                            None,  # attr, placeholder
                            match.distance,  # score, the distance between the centroid and the subsequence
                            l,  # staring loc
                            r  # ending loc
                        ))

            implets_ident = []
            for v in identified_implets_dict.values():
                implets_ident += v

            # blur and eval
            for implets, implet_src in zip([implets_real, implets_ident], ['real', 'ident']):
                for mode in ['single', 'all']:
                    for _ in range(n_trials):
                        # modify samples
                        if mode == 'single':
                            x_test_overwritten = []
                            x_test_rand_insert = []

                            for implet in implets:
                                i, _, _, _, start_loc, end_loc = implet
                                implet_len = end_loc - start_loc + 1
                                sample = xs[i].flatten()

                                sample_overwritten = overwrite_shaplet_random(sample, start_loc, implet_len)
                                sample_overwritten = sample_overwritten[np.newaxis]
                                x_test_overwritten.append(sample_overwritten)

                                sample_rand_insert = insert_random(sample, implet_len)
                                sample_rand_insert = sample_rand_insert[np.newaxis]
                                x_test_rand_insert.append(sample_rand_insert)

                            x_test_overwritten = np.array(x_test_overwritten)
                            x_test_rand_insert = np.array(x_test_rand_insert)

                            y_true = np.array([ys[i] for i, _, _, _, _, _ in implets])

                        elif mode == 'all':
                            x_test_overwritten = xs.copy()
                            x_test_rand_insert = xs.copy()

                            for implet in implets:
                                i, _, _, _, start_loc, end_loc = implet
                                implet_len = end_loc - start_loc + 1

                                x_test_overwritten[i] = overwrite_shaplet_random(
                                    x_test_overwritten[i].flatten(), start_loc, implet_len)[np.newaxis]
                                x_test_rand_insert[i] = insert_random(
                                    x_test_rand_insert[i].flatten(), implet_len)[np.newaxis]

                            y_true = ys
                        else:
                            raise ValueError

                        # compute accuracies
                        for method, x in zip(['repl_implet', 'repl_random_loc'],
                                             [x_test_overwritten, x_test_rand_insert]):
                            if len(x):
                                y_pred = predict(x)
                                acc = accuracy_score(y_true, y_pred)
                                entry = [model_name, task, explainer, method, implet_src, mode, acc]
                                print(entry)
                                result.append(entry)
                            else:
                                entry = [model_name, task, explainer, method, implet_src, mode, None]
                                print(entry)
                                result.append(entry)

        # get acc for baseline (without modifications)
        y_pred = predict(xs)
        acc = accuracy_score(ys, y_pred)
        entry = [model_name, task, None, 'ori', None, None, acc]
        print(entry)
        result.append(entry)

        df = pd.DataFrame(result,
                          columns=['model_name', 'task_name', 'xai_name',
                                   'method', 'implet_src', 'mode', 'acc_score'])
        df.to_csv(result_path, index=False)
