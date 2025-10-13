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
from utils.implet_extactor import implet_extractor
from utils.insert_shapelet import insert_random, overwrite_shaplet_random
from utils.constants import tasks,tasks_new
device = torch.device("cpu")

model_names = ['FCN', 'InceptionTime']
tasks = tasks_new + tasks #['GunPoint', "ECG200", "DistalPhalanxOutlineCorrect", "PowerCons", "Earthquakes", "Strawberry"] #

xai_names = ['Saliency','GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion'] #

# each row is [model_name, task_name, xai_name, method, acc_score]
# method is in ['ori', 'repl_implet', 'repl_random_loc']
result_path = f'output/implet_coverage_new.csv'


result = []
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

        for explainer in xai_names:

            implets_save_dir = f'./output/half_implet/{model_name}/{task}/{explainer}'
            implets_list = pickle_load_from_file(os.path.join(implets_save_dir, 'implets.pkl'))

            implets_class0 = implets_list['implets_class0']
            implets_class1 = implets_list['implets_class1']

            instance_count_class0 = len(set([imp[0] for imp in implets_class0]))
            instance_count_class1 = len(set([imp[0] for imp in implets_class1]))

            coverage_class0 =  instance_count_class0 / len(y_pred[y_pred==0])
            coverage_class1 = instance_count_class1 / len(y_pred[y_pred==1])
            total_coverage = (instance_count_class0+instance_count_class1) / len(y_pred)
            # print(instance_count_class0, instance_count_class1, len(y_pred[y_pred == 0]), len(y_pred[y_pred == 1]), coverage_class0,coverage_class1)
            entry = [model_name,task, explainer, total_coverage, coverage_class0, coverage_class1]

            result.append(entry)

df = pd.DataFrame(result, columns=['model_name', 'task_name', 'xai_name', 'total_coverage', 'coverage_class0','coverage_class1'])
df.to_csv(result_path, index=False)