import numpy as np 
import pandas as pd
# Embeddings
from sentence_transformers import SentenceTransformer

# train a classifier on the embeddings for multiclass regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error,  # regression metrics
    accuracy_score, f1_score, precision_score, recall_score # classification metrics
)
from sklearn.multioutput import MultiOutputRegressor, RegressorChain # for multiclass regression

# Estimators
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier
)
from sklearn.linear_model import (
    LinearRegression, 
    LogisticRegression,
    Ridge,
    Lasso
)
# other regressors
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor, LGBMClassifier

# type hinting
import os, json
from typing import List, Callable, Dict, Tuple, Any

# local imports
from src import data, utils
from src.embeddings import EmbeddingsRegressor

def comp_score(y_true:np.ndarray,y_pred:np.ndarray)->float:
    """
    Metric for multiclass regression. Computes the average of the RMSE scores for each label.
    """
    rmse_scores = []
    for i in range(y_true.shape[1]):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    return np.mean(rmse_scores)


def estimators_eval(estimators:List[Tuple[str,Any]], score_func:Callable[[np.ndarray, np.ndarray], float]):
    def fit_eval_estimators(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray) -> dict:
        estimator_scores = {}
        for name, estimator in estimators:
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            score = score_func(y_test, y_pred)#*(1.4*((y_train>th).sum()/(len(y_train)-1))) # weighted for class imbalance
            print(f"\"{name}\" estimator score: {score:.4f}")
            estimator_scores[name] = score
        return estimator_scores
    return fit_eval_estimators


def get_data():
    # load the train and test data
    train_data = data.load('train')
    test_df = data.load('test')
    # concat messages by subject id
    train_data = data.concat_messages(train_data)
    test_df = data.concat_messages(test_df)

    # split into 15% of subject ids for validation 
    # get the classes as the argmax of the label probabilities to use them for stratification
    subj_classes = train_data.set_index('subject_id').filter(regex='^d_')\
        .apply(lambda x: x.argmax() if x[:-1].sum()<0.5 else x[:-1].argmax(), axis=1)\
            .replace(dict(enumerate(train_data.filter(regex='^d_').columns)))
    tr_subj_ids, val_subj_ids = train_test_split(subj_classes.index, test_size=0.15, random_state=42, stratify=subj_classes.values)
    # split the train data into train and validation sets
    val_df = train_data[train_data['subject_id'].isin(val_subj_ids)]
    train_df = train_data[train_data['subject_id'].isin(tr_subj_ids)]

    # augment the train data by taking only the first half of the messages
    half_messages_df_train = train_df.assign(
        message=lambda df: df['message'].apply(lambda x: ' | '.join(x.split(' | ')[:len(x.split(' | '))//2])),
        # num_messages=lambda df: df['message'].apply(lambda x: len(x.split(' | ')))
    )
    train_df = pd.concat([train_df, half_messages_df_train], axis=0).sort_values('subject_id').reset_index(drop=True)
    return train_df, val_df, test_df