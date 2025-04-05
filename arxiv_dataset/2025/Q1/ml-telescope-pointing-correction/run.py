import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython import embed
import importlib
import model
importlib.reload(model)

from model import XGBoostRegressor
from dataset import PrepareData

random_seed = 412069413
np.random.seed(random_seed)


def get_split_indices(l, n):
    """
    Returns a list of cumulative indices to split an array of length l into n sub-arrays.
    The sub-arrays contain approximately the same number of rows.

    Parameters:
    l (int): The length of the array.
    n (int): The number of sub-arrays to split the array into.

    Returns:
    list: A list of cumulative indices indicating where to split the array.
    """
    
    indices = [0]
    size = l // n
    remainder = l % n
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + size + 1
        else:
            end = start + size
        indices.append(end)
        start = end

    return indices




def get_top_features_with_mutual_information(df, k, target_key, path_results, name):
    """
    Calculate the mutual information between features and a target variable,
    and return a list of the top k features with the highest mutual information.
    
    Parameters:
    - df: DataFrame containing the dataset to calculate mutual information on.
    - k: Number of top features to select.
    - target_key: Key indicating the target variable ('az' or 'el').
    - path_results: Path to the directory where the results will be saved.
    - name: Name of the file to save the selected features (this is also the model name).
    
    Returns:
    - selected_features: List of the top k features with the highest mutual information.
    """


    path_features = path_results + 'SelectedFeatures/'
    if not os.path.exists(path_features):
        os.makedirs(path_features)
        
    target       = 'Off_Az' if target_key == 'az' else 'Off_El'
    other_target = 'Off_El' if target_key == 'az' else 'Off_Az'
    
    X = df.loc[ : , ~df.columns.isin([target, other_target, 'date', 'day', 'rx'])]
    y = df[target]
    
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].to_list()
    df_feature_list = pd.DataFrame(data={'features':selected_features})
    df_feature_list.to_csv(os.path.join(path_features, name + '.csv'), index=False)
    return selected_features


def XGB_experiment_CV_exp1(process_number = 99, run_number = 99):
    
    path_results = f'./Results/Experiment1/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)
        
    df_model_params = pd.DataFrame(columns=['dataset', 'fold','target','num_features', 'Model Name', 'Model Parameters', 'Val RMS Model', 'Val RMS Current', 'Val RMS Compared', 'Test RMS Model', 'Test RMS Current', 'Test RMS Compared'])

    constant_features = ['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']
    for dataset_key,dataset in datasets:
        df = pd.read_csv(os.path.join(path_datasets, f'{dataset}/features_offsets.csv'))
        df['date'] = pd.to_datetime(df['date'])
        df.dropna(axis=1, thresh=len(df)-67, inplace=True)
        df.dropna(inplace=True)

        n_folds = 6
        test_size = 1/n_folds
        df_folds = np.array_split(df, n_folds)
        
        for i,df_tmp in enumerate(df_folds):
            n = len(df_tmp)
            df_trainval = df_tmp.iloc[:int((1-test_size)*n)]
            df_test = df_tmp.iloc[int((1-test_size)*n):]
            for target_key in ['az', 'el']:
                for k in [2,5,10,20,30,40,50]:      
                    model_name = f'XGB_ds{dataset_key}_tp{i}_k{k}_{target_key}'
                    feature_list = get_top_features_with_mutual_information(df_trainval, k, target_key, path_results, model_name)
                    feature_list += constant_features
                    feature_list = sorted(list(set(feature_list)))
                    
                    if len(df_tmp['rx'].unique()) > 1:
                        feature_list.append('rx')

                    ds = PrepareData(
                        df = df_trainval.copy(),
                        parameter_keys = (process_number, dataset_key),
                        feature_list = feature_list,
                        target_key = target_key,
                        run_number = run_number,
                        path_results = path_results,
                    )

                    model = XGBoostRegressor(ds, name = model_name)
                    model_parameters = model.train()
                    RMS_val_model, RMS_val_current = model.plot_sorted_prediction()
                    RMS_val_compared = RMS_val_model / RMS_val_current
                    
                    #Test model on df_test
                    X = df_test[feature_list]
                    if 'rx' in X.columns:
                        le = LabelEncoder()
                        X['rx'] = le.fit_transform(X['rx'])
                        
                    X = X.values
                    y = df_test['Off_Az' if target_key == 'az' else 'Off_El'].values

                    y_pred = model.model.predict(X)
                    model.plot_histogram(X, y, '_test')
                    RMS_test_model = np.sqrt(mean_squared_error(y, y_pred))
                    RMS_test_current = np.sqrt(mean_squared_error(y, np.zeros_like(y)))
                    RMS_test_compared = RMS_test_model / RMS_test_current

                    df_model_params = df_model_params.append({
                        'dataset': dataset,
                        'fold': i,
                        'target': target_key,
                        'num_features': k,
                        'Model Name': model_name,
                        'Model Parameters': model_parameters,
                        'Val RMS Model': RMS_val_model,
                        'Val RMS Current': RMS_val_current,
                        'Val RMS Compared': RMS_val_compared,
                        'Test RMS Model': RMS_test_model,
                        'Test RMS Current': RMS_test_current,
                        'Test RMS Compared': RMS_test_compared,
                    }, ignore_index=True)
                    
                    df_model_params.to_csv(path_results + f'model_parameters_short.csv', index=False)



def XGB_experiment_CV_exp2(process_number = 99, run_number = 99):
    

    path_results = f'./Results/Experiment2/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    
    path_datasets = './Datasets/'
    datasets = [(1,'tmp2022_clean_clf'), (2,'tmp2022_clean_clf_nflash230')]

    constant_features = ['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']

    # create an empty DataFrame
    df_model_params = pd.DataFrame(columns=['dataset', 'fold','target','num_features', 'Model Name', 'Model Parameters', 'Val RMS Model', 'Val RMS Current', 'Val RMS Compared', 'Test RMS Model', 'Test RMS Current', 'Test RMS Compared'])

    n_folds = 6

    for dataset_key,dataset in datasets:
        df = pd.read_csv(os.path.join(path_datasets, f'{dataset}/features_offsets.csv'))
        df['date'] = pd.to_datetime(df['date'])
        df.dropna(axis=1, thresh=len(df)-67, inplace=True)
        df.dropna(inplace=True)

        n = len(df)
        split_indices = get_split_indices(n, n_folds)

        for i in range(n_folds):
            df_test = df.iloc[split_indices[i] : split_indices[i+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[i]], df.iloc[split_indices[i+1]:]])

            for target_key in ['az', 'el']:
                for k in [2,5,10,20,30,40,50]:      
                    model_name = f'XGB_ds{dataset_key}_tp{i}_k{k}_{target_key}'

                    
                    feature_list = get_top_features_with_mutual_information(df_trainval.copy(), k, target_key, path_results, model_name)
                    feature_list += constant_features
                    feature_list = sorted(list(set(feature_list)))
    
                    if len(df_trainval['rx'].unique()) > 1:
                        feature_list.append('rx')

                    ds = PrepareData(
                        df = df_trainval.copy(),
                        parameter_keys = (process_number, dataset_key),
                        feature_list = feature_list,
                        target_key = target_key,
                        run_number = run_number,
                        path_results = path_results,
                    )
                    
                    model = XGBoostRegressor(ds, name = model_name)
                    model_parameters = model.train()

                    RMS_val_model, RMS_val_current = model.plot_sorted_prediction()
                    RMS_val_compared = RMS_val_model / RMS_val_current
                    
                    #Test model on df_test
                    X = df_test[feature_list]
                    if 'rx' in X.columns:
                        le = LabelEncoder()
                        X['rx'] = le.fit_transform(X['rx'])
                    X = X.values
                    y = df_test['Off_Az' if target_key == 'az' else 'Off_El'].values

                    y_pred = model.model.predict(X)
                    model.plot_histogram(X, y, '_test')
                    
                    RMS_test_model = np.sqrt(mean_squared_error(y, y_pred))
                    RMS_test_current = np.sqrt(mean_squared_error(y, np.zeros_like(y)))
                    RMS_test_compared = RMS_test_model / RMS_test_current

        
                    df_model_params = df_model_params.append({
                        'dataset': dataset,
                        'fold': i,
                        'target': target_key,
                        'num_features': k,
                        'Model Name': model_name,
                        'Model Parameters': model_parameters,
                        'Val RMS Model': RMS_val_model,
                        'Val RMS Current': RMS_val_current,
                        'Val RMS Compared': RMS_val_compared,
                        'Test RMS Model': RMS_test_model,
                        'Test RMS Current': RMS_test_current,
                        'Test RMS Compared': RMS_test_compared,
                    }, ignore_index=True)
                

                    df_model_params.to_csv(path_results + f'model_parameters_long.csv', index=False)

    


if __name__ == '__main__':
            
    path_datasets = './Datasets/'
    datasets = [(1,'tmp2022_clean_clf'), (2,'tmp2022_clean_clf_nflash230')]
    
    XGB_experiment_CV_exp1()
    XGB_experiment_CV_exp2()