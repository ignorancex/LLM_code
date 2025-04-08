import os, glob, json
import logging
from pathlib import Path
from typing import List, Callable, Dict, Tuple, Any
from functools import partial, lru_cache

import pandas as pd
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, DatasetDict, Dataset, NamedSplit

from sentence_transformers import SentenceTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor # for multiclass regression

# from utils import make_evaluator, comp_score


def comp_score(y_true:np.ndarray,y_pred:np.ndarray)->float:
    """
    Metric for multiclass regression. Computes the average of the RMSE scores for each label.
    """
    rmse_scores = []
    print(y_true.shape, y_pred.shape)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        for i in range(y_true.shape[1]):
            rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    else:
        rmse_scores.append(np.sqrt(mean_squared_error(y_true.ravel(),y_pred.ravel())))
    return np.mean(rmse_scores)

def label_metrics(score_fun, y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        scores = []
        for i in range(y_true.shape[1]):
            scores.append(score_fun(y_true[:,i],y_pred[:,i]))
        return scores
    return score_fun(y_true.ravel(),y_pred.ravel())

def make_evaluator(X_test:np.ndarray, y_test:np.ndarray):
    def eval_estimators(estimators:List[Tuple[str, Any]], score_func:Callable[[np.ndarray, np.ndarray], float]=comp_score) -> Dict[str, float]:
        estimator_scores = {}
        for name, estimator in estimators:
            y_pred = estimator.predict(X_test)
            metric_scores = label_metrics(score_func, y_test, y_pred)
            estimator_scores[name] = metric_scores
        return estimator_scores
    return eval_estimators

rmse = partial(mean_squared_error, squared=False)

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_embeddings.log"),
        logging.StreamHandler()
    ]
)


@lru_cache(maxsize=1)
def load_data_for_task(task:str='d') -> pd.DataFrame:
    ds = load_dataset(f'nlpUc3mStudents/mental-risk-{task}')
    # to pandas
    train_df:pd.DataFrame = ds['train'].to_pandas()
    test_df:pd.DataFrame = ds['test'].to_pandas()
    label_names = train_df.iloc[:,4:].columns.tolist()
    # concat messages by subject id
    train_by_subjectid = (
        train_df.groupby('subject_id')
        .agg({'message': lambda x: ' | '.join(x), **{col: 'first' for col in label_names}})
        .reset_index()
    )
    test_by_subjectid = (
        test_df.groupby('subject_id')
        .agg({'message': lambda x: ' | '.join(x), **{col: 'first' for col in label_names}})
        .reset_index()
    )
    data = pd.concat([train_by_subjectid, test_by_subjectid], axis=0)
    return data

def split_data(data, embeddings):
    Y = data.iloc[:, 2:].values.astype(np.float32)
    classes = data.iloc[:,2:].apply(lambda x: x.argmax(),axis=1).replace(dict(enumerate(data.iloc[:,2:].columns)))
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, Y, 
        test_size=0.15, 
        random_state=42, 
        stratify=classes
    )
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logging.info(pd.DataFrame(y_test).apply(lambda x: x.argmax(),axis=1).replace(dict(enumerate(data.iloc[:,2:].columns))).value_counts())
    return X_train, X_test, y_train, y_test

@lru_cache(maxsize=1)
def make_embeddings_for_task(task:str, embeddings_model:str):
    data = load_data_for_task(task)
    text = data['message'].values.tolist()
    return make_embeddings(text, embeddings_model)

def make_embeddings(text:List[str], embeddings_model:str) -> np.array:
    if "m2m" in embeddings_model:
        # m2m is handled differently
        model = pipeline("feature-extraction", model=embeddings_model, device=0)
    model = SentenceTransformer(embeddings_model) # note: SentenceTransformer from BERTforSequenceClassification (will throw a warning)
    embeddings = model.encode(text)
    return embeddings

def evaluate_embeddings(train_df, val_df, task:str, embedding_model:str, estimators:List[Any], metrics:List[Callable]):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logging.info(f"Making embeddings with model: {embedding_model}...")
    tr_embeddings = make_embeddings(train_df['message'].values.tolist(), embedding_model)
    val_embeddings = make_embeddings(val_df['message'].values.tolist(), embedding_model)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    X_train, y_train = tr_embeddings, train_df.filter(regex=f'^{task}_').assign(
        label=lambda df: df.apply(lambda x: x.tolist() if len(x)>1 else x, axis=1)
    ).label.values
    X_test, y_test = val_embeddings, val_df.filter(regex=f'^{task}_').assign(
        label=lambda df: df.apply(lambda x: x.tolist() if len(x)>1 else x, axis=1)
    ).label.values
    labels = train_df.filter(regex=f'^{task}_').columns.tolist()
    # evaluate
    logging.info(f"Evaluating task: {task}, Embedding: {embedding_model}")
    for name, estimator in estimators:
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = comp_score(y_test, y_pred)
        logging.info(f"- Estimator: {name}, score={score}")
    evaluate_estimators = make_evaluator(X_test, y_test)
    metric_dfs = []
    for metric in metrics:
        metric_scores = evaluate_estimators(estimators, metric)
        metrics_df = pd.DataFrame(metric_scores, index=labels).T\
            .rename_axis(metric.__name__, axis=1)\
            .rename(columns=lambda x: metric.__name__ + '_' + x)
        logging.info(f"Metric: {metric.__name__} -----------------")
        logging.info('\n'+str(metrics_df))
        metric_dfs.append(metrics_df)
        # metrics_df.to_csv(f"reports/{task}_{embedding_model.split('/')[-1].replace(' ','_')}_{metric.__name__}.csv")
    metric_df = pd.concat(metric_dfs, axis=1).assign(
        mean=lambda df: df.mean(axis=1),
        std=lambda df: df.std(axis=1),
        task=task,
        embedding_model=embedding_model
    )
    # metric_df.to_csv(f"reports/embeddings/task_{task}_{embedding_model.split('/')[-1].replace(' ','_')}.csv")
    return metric_df


def load_data(task) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    load load the train and test data

    returns:
    --------
    train_df: pd.DataFrame
        dataframe with the train data
    val_df: pd.DataFrame
        dataframe with the validation data
    """
    from src import data
    from sklearn.model_selection import train_test_split
    train_data = data.load('train')
    # concat messages by subject id
    train_data = data.concat_messages(train_data)

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
    return train_df, val_df

if __name__ == "__main__":
    # Estimators
    from sklearn.multioutput import MultiOutputRegressor, RegressorChain
    from sklearn.ensemble import (
        RandomForestRegressor, 
        GradientBoostingRegressor,
        AdaBoostRegressor)
    from sklearn.linear_model import (
        LinearRegression, 
        Ridge,
        Lasso)
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from lightgbm import LGBMRegressor
    # Metrics
    from sklearn.metrics import r2_score, mean_squared_error
    from src.multiregression import RegChainWithPCA
    from functools import partial

    task = 'd'
    multioutput_strategy = partial(
        RegChainWithPCA,
        num_components=0.55,
        pca_exclude_first=True,
    )
    # multioutput_strategy = MultiOutputRegressor
    # multioutput_strategy = RegressorChain

    embedding_models = [
        "PlanTL-GOB-ES/roberta-base-bne",
        "hackathon-somos-nlp-2023/roberta-base-bne-finetuned-suicide-es", # roberta fine-tuned for classifying sucide
        # "bert-base-uncased",
        # "bert-base-multilingual-uncased",
        "dccuchile/bert-base-spanish-wwm-cased",
        # "guidecare/all-mpnet-base-v2-feature-extraction"
    ]

    estimators = [
        LinearRegression(),
        RandomForestRegressor(n_jobs=-1),
        LGBMRegressor(n_jobs=-1),
        GradientBoostingRegressor(),
        AdaBoostRegressor(),
        SVR(),
        # KNeighborsRegressor(),
        MLPRegressor(),
        # DecisionTreeRegressor(),
        Lasso(),
        Ridge()
    ]
    make_multreg = lambda reg: multioutput_strategy(reg) if task=='d' else reg
    estimators = [(reg.__class__.__name__, make_multreg(reg)) for reg in estimators]
    metrics = [r2_score, mean_squared_error]
    logging.info(f"Loading data for task {task}...")
    train_df, val_df = load_data(task)
    results = []
    for embedding_model in embedding_models:
        logging.info(f"Evaluating embeddings with model: {embedding_model}...")
        logging.info('='*50)
        embedding_model_result = evaluate_embeddings(train_df, val_df, task, embedding_model, estimators, metrics)
        results.append(embedding_model_result)
    results_df = pd.concat(results, axis=0)
    results_df.to_csv(f"reports/embeddings/task_{task}_results.csv")





