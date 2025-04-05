import pandas as pd
import numpy as np
import os
import random
from sklearn.utils.class_weight import compute_class_weight
import torch
import sys

from torchmetrics import ConfusionMatrix

# A 2 Dimenstional Numpy Array Of Efficacy/Resistance:
if os.path.isdir("../data"):
    project_folder = "../"
else:
    shared_folder = "OoD + KBNN"
    project_folder = "/content/gdrive/Shareddrives/"+shared_folder
efficacy_table = pd.read_csv(os.path.join(project_folder,'data/raw/HIV_resistance_score.csv'), delimiter=';', index_col=0) #

# Params
row_keys = list(efficacy_table.index)

col_keys = list(efficacy_table.columns)

all_keys = set(row_keys).union(set(col_keys))


def get_subset_rows(df, subset_cols, opposite=True):
    if len(subset_cols)==0:
        return np.array([True]*len(df)),np.array([False]*len(df))
    if opposite:
        app = df[subset_cols].sum(axis=1)
    else:
        other_cols = [col for col in df.columns if (col not in subset_cols and col!='label')]
        app = df[other_cols].sum(axis=1)

    return app==0, app!=0

def subset_to_rows(df, subset_cols, opposite=True):
    train_ids, test_ids = get_subset_rows(df, subset_cols, opposite)

    df_train = df.loc[train_ids].reset_index(drop=True)
    df_test = df.loc[test_ids].reset_index(drop=True)

    return df_train, df_test

def subset_cols_to_str(subset_cols, opposite=True):
    app = set(subset_cols)
    if opposite:
        app = all_keys.difference(app)
    return str(sorted(list(app))).replace(",","").replace("(","").replace(")","").replace("[","").replace("]","").replace("{","").replace("}","").replace(" ","_")


def save_results(results_folder, experiment_name, df_res, df_res_div):
    folder_results = os.path.join(results_folder,experiment_name)
    if not os.path.isdir(folder_results):
        os.makedirs(folder_results)

    old_threshold = np.get_printoptions()["threshold"]
    np.set_printoptions(threshold=sys.maxsize)

    df_res.to_csv(os.path.join(folder_results,'conf_matrix_all.csv'),index=False)
    df_res_div.to_csv(os.path.join(folder_results,'conf_matrix_div.csv'),index=False)

    np.set_printoptions(threshold=old_threshold)

def load_results(results_folder,experiment_name):
    folder_results = os.path.join(results_folder,experiment_name)

    #try:
    df_res = pd.read_csv(os.path.join(folder_results,'conf_matrix_all.csv'))
    #except FileNotFoundError:
    #    df_res = pd.read_csv(os.path.join(folder_results,'conf_matrix_all_FC.csv'))
    #try:
    df_res_div = pd.read_csv(os.path.join(folder_results,'conf_matrix_div.csv'))
    #except FileNotFoundError:
    #    df_res_div = pd.read_csv(os.path.join(folder_results,'conf_matrix_div_FC.csv'))
    #   save_results(results_folder,experiment_name,df_res,df_res_div)

    return df_res, df_res_div

def get_all_data(raw_data_folder, filename='../data/raw/HIV_database_try.csv', remove_not_in_efficacy_table=True):
    app = pd.read_csv(filename)#+["No",""][use_history]+'HistoryMuts.csv')
    #last column is target
    #app.rename(columns={"label":"Label"},inplace=True)

    if remove_not_in_efficacy_table:
        cols_to_remove = set(app.columns[:-1]).difference(row_keys).difference(col_keys)
        #print(cols_to_remove)
        #last column is target

        app.drop(cols_to_remove,axis=1,inplace=True)

    return app

def load_results(results_folder,experiment_name):
    folder_results = os.path.join(results_folder,experiment_name)

    #try:
    df_res = pd.read_csv(os.path.join(folder_results,'conf_matrix_all.csv'))
    #except FileNotFoundError:
    #    df_res = pd.read_csv(os.path.join(folder_results,'conf_matrix_all_FC.csv'))
    #try:
    df_res_div = pd.read_csv(os.path.join(folder_results,'conf_matrix_div.csv'))
    #except FileNotFoundError:
    #    df_res_div = pd.read_csv(os.path.join(folder_results,'conf_matrix_div_FC.csv'))
    #   save_results(results_folder,experiment_name,df_res,df_res_div)

    return df_res, df_res_div
    

###FC
def load_data_FC(raw_data_folder, filename, subset_cols, opposite=True):
    data = {}
    #subsets_dir = os.path.join(raw_data_folder, 'all_battles'+['','_no_draws'][keep_draws]+'_subsets')

    #file_loc = os.path.join(raw_data_folder,'all_battles'+['_no_draws',''][keep_draws]+".csv")
    df = get_all_data(raw_data_folder, filename)

    data["train"],data["test"] = subset_to_rows(df, subset_cols, opposite)
    #print(dataset[sub_set].shape)

    return data


def split_data_FC(data, val_split=0.2, random_seed=42):
    random.seed(random_seed)

    if len(data["test"])==0: #test not already divided
        train_num_samples = len(data['train'])
        num_test = int(train_num_samples * 0.2)
        data['train'] = data['train'].iloc[:-num_test]
        data["test"] = data['train'].iloc[-num_test:]

    train_num_samples = len(data['train'])
    index_traindata = np.random.choice(range(train_num_samples),size=train_num_samples,replace=False)

    num_val = int(train_num_samples * val_split)

    data['val'] = data['train'].iloc[index_traindata[:num_val]]
    data['train'] = data['train'].iloc[index_traindata[num_val:]]

    for split in ["train","val","test"]:
        data["x_"+split] = np.array(data[split].iloc[:, :-1])
        data["y_"+split] = data[split].iloc[:, -1]
        data["y_"+split] = np.array(data["y_"+split]) #.reshape(data["y_"+split].shape[0],1)

    return data

def compute_class_weights(y, as_tensor=True):
    app = compute_class_weight("balanced", classes = np.unique(y), y = y)
    if as_tensor:
        return torch.Tensor(app)
    return app



def compute_metrics(num_classes,model,data,loaders,trainer,opposite=True):
    confmat = ConfusionMatrix(num_classes=num_classes, task='BINARY')

    dict_res = {}

    dict_res_div = {'type': list(all_keys.intersection(data["train"].columns)),
                    'Confusion Matrix_train':[],
                    'Confusion Matrix_val':[],
                    'Confusion Matrix_test':[],
                    'probs_train':[],
                    'probs_val':[],
                    'probs_test':[],
                    'pred_train':[],
                    'pred_val':[],
                    'pred_test':[],
                    'true_train':[],
                    'true_val':[],
                    'true_test':[]
                    }

    for split in ["train","val","test"]:
        y_probs = torch.cat(trainer.predict(model,loaders[split]))
        if len(y_probs.shape)==1 or y_probs.shape[-1]==1:
            y_hat = y_probs >= 0.5
        else:
            y_hat = y_probs.argmax(-1)
        y_hat = torch.tensor(y_hat, dtype=torch.int32)
        true_y = torch.tensor(data["y_"+split])

        conf_matr = confmat(true_y.squeeze(), y_hat.squeeze()).detach().cpu().numpy()
        loss = trainer.test(model,loaders[split])[0]['loss/test']

        dict_res['loss_'+split] = [loss]
        #print(conf_matr)
        #dict_res['TP_'+str_matr] = [int(conf_matr[0][0])]
        #dict_res['FN_'+str_matr] = [int(conf_matr[0][1])]
        #dict_res['FP_'+str_matr] = [int(conf_matr[1][0])]
        #dict_res['TN_'+str_matr] = [int(conf_matr[1][1])]
        dict_res['Confusion Matrix_'+split] = [conf_matr]
        dict_res['probs_'+split] = [y_probs.detach().cpu().numpy()]
        dict_res['pred_'+split] = [y_hat.detach().cpu().numpy()]
        dict_res['true_'+split] = [true_y.detach().cpu().numpy()]

        for key in dict_res_div["type"]:
            if key in data[split].columns:
                _, select_ids = get_subset_rows(data[split], [key], opposite)
                select_ids = np.where(select_ids)[0]
                #select_ids = np.array(select_ids[select_ids==True].index)
                #select_ids = np.where(np.in1d(data[split], select_ids))[0]
                
                if len(select_ids) == 0:
                    dict_res_div['Confusion Matrix_'+split].append(np.nan)
                    dict_res_div['probs_'+split].append(np.nan)
                    dict_res_div['pred_'+split].append(np.nan)
                    dict_res_div['true_'+split].append(np.nan)
                    #dict_res_div['TP_'+str_matr].append(np.nan)
                    #dict_res_div['FN_'+str_matr].append(np.nan)
                    #dict_res_div['FP_'+str_matr].append(np.nan)
                    #dict_res_div['TN_'+str_matr].append(np.nan)
                else:
                    y_probs_div = y_probs[select_ids]
                    y_hat_div = y_hat[select_ids]
                    true_y_div = true_y[select_ids]
                    conf_matr_div = confmat(true_y_div.squeeze(), y_hat_div.squeeze() ).detach().cpu().numpy()
                    dict_res_div['Confusion Matrix_'+split].append(conf_matr_div)
                    dict_res_div['probs_'+split].append(y_probs_div.detach().cpu().numpy())
                    dict_res_div['pred_'+split].append(y_hat_div.detach().cpu().numpy())
                    dict_res_div['true_'+split].append(true_y_div.detach().cpu().numpy())
                    #dict_res_div['TP_'+str_matr].append(int(conf_matr_div[0][0]))
                    #dict_res_div['FN_'+str_matr].append(int(conf_matr_div[0][1]))
                    #dict_res_div['FP_'+str_matr].append(int(conf_matr_div[1][0]))
                    #dict_res_div['TN_'+str_matr].append(int(conf_matr_div[1][1]))
    df_res = pd.DataFrame.from_dict(dict_res)
    df_res_div = pd.DataFrame.from_dict(dict_res_div)
    return df_res,df_res_div