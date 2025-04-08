import os, glob
import pandas as pd
import numpy as np

train_dir = "./data/train"
test_dir = "./data/test"
truth_dir = "golden_truth"

def load(set_name:str='train', with_labels:bool=True) -> pd.DataFrame:
    """
    Load the data from the csv files
    """
    if set_name == 'train':
        path = train_dir
    elif set_name == 'test':
        path = test_dir
    else:
        raise ValueError("set_name must be either 'train' or 'test'")
    if not os.path.exists(path):
        if set_name=="train":
            df = get_train(with_labels=with_labels)
        else:
            df = get_test(with_labels=with_labels)
    else:
        data_files = glob.glob(os.path.join(path, '*.json'))
        if with_labels:
            truth_path = os.path.join(path, truth_dir, 'task2_gold_d.txt')
            truth_df = pd.read_csv(truth_path).rename(
                columns=lambda s: 'd_' + s if s != 'Subject' else 'subject_id'
            )
        else:
            truth_df = None
        df = load_from_files(data_files, truth=truth_df)
        abc_labels_df = make_task_labels_from_d(df.filter(regex='^d_').values.astype(float))
        df = pd.concat([df, abc_labels_df], axis=1)
    return df

def concat_messages(df:pd.DataFrame, sep:str=' | ') -> pd.DataFrame:
    """
    Concatenate all the messages of a subject into a single message
    """
    df = (
        df
        .assign(date=lambda x: pd.to_datetime(x['date']))
        .sort_values(['subject_id', 'date'], ascending=[True, True])
        .groupby('subject_id')
        .agg({
            'message': lambda x: sep.join(x),
            'round': 'last',
            **{c: 'first' for c in df.columns.drop(['subject_id', 'message', 'round'])}
         }).sort_index()
         .reset_index()
    )
    return df

def load_from_files(files, truth=None):
    """load all the data into a dataframe"""
    import os, json
    data = []
    for f in files:
        with open(f) as file:
            msgs = json.load(file)
            for msg in msgs:
                data.append([
                    msg.get('nick',os.path.basename(f).split('.')[0]),
                    msg.get('round', -1),
                    msg['id_message'], 
                    msg['date'], 
                    msg['message']])
    df = pd.DataFrame(data, columns=['subject_id', 'round', 'id_message', 'date', 'message'])
    if truth is not None:
        df = df.merge(truth, on='subject_id')
    return df

def get_train(hf_token:str):
    from datasets import load_dataset, Dataset
    ds = load_dataset('nlpUc3mStudents/mental-risk-d')
    train_df = ds['train'].to_pandas()
    return train_df

def get_test(hf_token:str):
    raise NotImplementedError("Test data is not available")


task_d_cols = ['suffer+in favour', 'suffer+against', 'suffer+other', 'control']

def make_task_labels_from_d(d_data:np.ndarray, include_d:bool=False) -> pd.DataFrame:
    """
    Get the labels of all other tasks from the labels of the d task
    """
    if isinstance(d_data, pd.DataFrame):
        d_df = d_data.astype(float)
    else:
        d_df = pd.DataFrame(d_data, columns=task_d_cols).astype(float)
    df = d_df.assign(
        c_label = lambda df: df.iloc[:,:-1].apply(
            lambda x: df.columns[np.argmax(x)] if sum(x)>=0.5 else 'control', axis=1
        ),
        a_label=lambda df: (df.c_label!='control').astype(int),
        b_label = lambda df: df[task_d_cols[:-1]].sum(axis=1).round(2)
    )
    if not include_d:
        df = df[['a_label', 'b_label', 'c_label']]
    return df