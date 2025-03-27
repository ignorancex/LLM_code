import pickle as pkl
import numpy as np
import os

project_dir = '.'
checkpoints_dir = 'checkpoint_archive'
dataset_category = 'realworld'
mae_pickle = 'test_mae.pkl'
mae_report = 'test_mae_report.log'
mae_e_pickle = 'test_mae_e.pkl'
mae_e_report = 'test_mae_e_report.log'
checkpoint_batch_indexs = ['1', '2', '3']

dict_dataset_names = {
    'synthetic': ['hawkes_1_v2', 'hawkes_2_v2', 'poisson_v2', 'self_correct_v2', 'stationary_renewal_v2'],
    'realworld': ['bookorder', 'mooc', 'retweet', 'stackoverflow']
}

for checkpoint_batch_index in checkpoint_batch_indexs:
    selected_checkpoint_batch_dir = os.path.join(project_dir, checkpoints_dir, dataset_category, checkpoint_batch_index, 'output')
    dataset_names = dict_dataset_names[dataset_category]
    for dataset_name in dataset_names:
        result_of_one_dataset = os.path.join(selected_checkpoint_batch_dir, dataset_name)
        for result in os.listdir(result_of_one_dataset):
            results_dir = os.path.join(result_of_one_dataset, result)

            f_mae = open(os.path.join(results_dir, mae_pickle), 'rb')
            mae = pkl.load(f_mae)
            f_mae.close()

            # We want the mean, var, Q1, Q2, and Q3 of MAE.
            mae_array = mae.cpu().numpy()
            mean, var = mae_array.mean(), mae_array.var()
            Q1, Q2, Q3 = np.percentile(mae_array, [25, 50, 75]).tolist()
            
            f_report = open(os.path.join(results_dir, mae_report), 'w')
            f_report.write(f'We announce the mean is {mean} with variance is {var}.\n Q1: {Q1}, Q2: {Q2}, Q3: {Q3}.')
            f_report.close()

            # As some MTPP models do not have MAE-E evaluation part.
            # We will try this part.
            try:
                f_mae_e = open(os.path.join(results_dir, mae_e_pickle), 'rb')
                mae_e_array = pkl.load(f_mae_e)
                f_mae_e.close()
    
                # We want the mean, var, Q1, Q2, and Q3 of MAE.
                mean, var = mae_e_array.mean(), mae_e_array.var()
                Q1, Q2, Q3 = np.percentile(mae_e_array, [25, 50, 75]).tolist()
                
                f_report = open(os.path.join(results_dir, mae_e_report), 'w')
                f_report.write(f'We announce the mean is {mean} with variance is {var}.\n Q1: {Q1}, Q2: {Q2}, Q3: {Q3}.')
                f_report.close()
            except:
                continue
