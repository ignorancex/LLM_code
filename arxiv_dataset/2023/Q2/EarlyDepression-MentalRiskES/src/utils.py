from typing import List, Tuple
import numpy as np
import pandas as pd

def print_messages(msgs:List[dict]):
    """
    Print the messages of a subject
    
    Messages are a list of dictionaries of the form: [{'id_message': {int_id}, 'message': '{str_message}', 'date': '{str_date}'}, ...]
    and are attached to an specific subject.
    """
    for message in msgs:
        print(f"{message['date']} - {message['message']}")

def load_data(files, truth):
    """load all the data into a dataframe"""
    import os, json
    data = []
    for f in files:
        with open(f) as file:
            msgs = json.load(file)
            for msg in msgs:
                data.append([os.path.basename(f).split('.')[0], msg['id_message'], msg['date'], msg['message']])
    df = pd.DataFrame(data, columns=['subject_id', 'id_message', 'date', 'message'])
    df = df.merge(truth, on='subject_id')
    return df


def normalize(x, prob=True):
    """
    Normalize a vector to [0,1] and sum 1 if prob=True
    """
    x = x.reshape(-1,4)
    # normalize to [0,1]
    x = ((x - x.min(axis=1)[...,None])/(x.max(axis=1)[...,None] - x.min(axis=1)[...,None])).round(4)
    if prob:
        # normalize to sum 1
        x = x/x.sum(axis=1)[...,None]
    return x.round(4)

def label_metrics(score_fun, y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(score_fun(y_true[:,i],y_pred[:,i]))
    return scores

def make_predict(predict_fn, **kwargs):
    def predict(msg):
        pred = predict_fn(msg, **kwargs)
        return pred
    return predict


def comp_score(y_true:np.ndarray,y_pred:np.ndarray)->float:
    """
    Metric for simple and multiclass regression. Computes the average of the RMSE scores for each label.
    """
    from sklearn.metrics import mean_squared_error
    rmse_scores = []
    for i in range(y_true.shape[1]):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    return np.mean(rmse_scores)