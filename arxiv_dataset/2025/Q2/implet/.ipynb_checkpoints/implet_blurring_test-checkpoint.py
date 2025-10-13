import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tsai.models.FCN import FCN
from tsai.models.InceptionTime import InceptionTime

from utils import pickle_save_to_file
from utils.constants import tasks, tasks_new, xai_names
from utils.data_utils import read_UCR_UEA
from utils.implet_extactor import implet_extractor
from utils.insert_shapelet import insert_random, overwrite_shaplet_random

device = torch.device("cpu")

# if there's multiple implet in a sample
# if 'single', replace each implet individually
# if 'all', replace all implets on the same sample
# modes = ['single', 'all', 'single_pos_only', 'all_pos_only']
mode = 'all'
insert_mode = 'mean'
# tasks = tasks + tasks_new
tasks = ['Earthquakes', 'FordA']
model_names = ['FCN', 'InceptionTime']

# each row is [model_name, task_name, xai_name, method, acc_score]
# method is in ['ori', 'repl_implet', 'repl_random_loc']
result_path = f'output/blurring_test_all_mean_repl.csv'
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
        _, x_test, _, y_test, _ = read_UCR_UEA(task, None)
        y_test = np.argmax(y_test, axis=1)

        # run prediction on unmodified samples
        y_pred = predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        entry = [model_name, task, None, 'ori', acc]
        print(entry)
        result.append(entry)

        for explainer in xai_names:
            if model_name == 'InceptionTime' and explainer == 'DeepLift':
                continue 
            
            # load attributions
            with open(f'attributions/{model_name}/{task}/{explainer}/test_exp.pkl', 'rb') as f:
                attr = pickle.load(f)
            attr_test = attr['attributions']

            # compute implets
            if task == 'Chinatown':
                kmin = 2
            else:
                kmin = None
            implets_class0 = implet_extractor(x_test, y_test, attr_test, target_class=0, kmin=kmin)
            implets_class1 = implet_extractor(x_test, y_test, attr_test, target_class=1, kmin=kmin)
            implets = implets_class0 + implets_class1

            implets_list = {
                'implets': implets,
                'implets_class0': implets_class0,
                'implets_class1': implets_class1,
            }
            implets_save_dir = f'./output/{model_name}/{task}/{explainer}'
            pickle_save_to_file(data=implets,
                                file_path=os.path.join(implets_save_dir, 'implets.pkl'))

            for _ in range(n_trials):
                # modify samples
                if mode == 'single':
                    x_test_overwritten = []
                    x_test_rand_insert = []

                    for implet in implets:
                        i, _, _, _, start_loc, end_loc = implet
                        implet_len = end_loc - start_loc + 1
                        sample = x_test[i].flatten()

                        sample_overwritten = overwrite_shaplet_random(sample, start_loc, implet_len, mode=insert_mode)
                        sample_overwritten = sample_overwritten[np.newaxis]
                        x_test_overwritten.append(sample_overwritten)

                        sample_rand_insert = insert_random(sample, implet_len, mode=insert_mode)
                        sample_rand_insert = sample_rand_insert[np.newaxis]
                        x_test_rand_insert.append(sample_rand_insert)

                    x_test_overwritten = np.array(x_test_overwritten)
                    x_test_rand_insert = np.array(x_test_rand_insert)

                    y_true = np.array([y_test[i] for i, _, _, _, _, _ in implets])

                elif mode == 'all':
                    x_test_overwritten = x_test.copy()
                    x_test_rand_insert = x_test.copy()

                    for implet in implets:
                        i, _, _, _, start_loc, end_loc = implet
                        implet_len = end_loc - start_loc + 1

                        x_test_overwritten[i] = overwrite_shaplet_random(
                            x_test_overwritten[i].flatten(), start_loc, implet_len, mode=insert_mode)[np.newaxis]
                        x_test_rand_insert[i] = insert_random(
                            x_test_rand_insert[i].flatten(), implet_len, mode=insert_mode)[np.newaxis]

                    y_true = y_test
                else:
                    raise ValueError

                if implets:  # if there's at least one implet
                    # compute accuracies
                    for method, x in zip(['repl_implet', 'repl_random_loc'],
                                         [x_test_overwritten, x_test_rand_insert]):
                        y_pred = predict(x)
                        acc = accuracy_score(y_true, y_pred)
                        entry = [model_name, task, explainer, method, acc]
                        print(entry)
                        result.append(entry)

        df = pd.DataFrame(result, columns=['model_name', 'task_name', 'xai_name', 'method', 'acc_score'])
        df.to_csv(result_path, index=False)
