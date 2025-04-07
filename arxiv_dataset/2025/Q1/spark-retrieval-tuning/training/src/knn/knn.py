import json
from operator import ge
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .embeddings import get_embeddings
from .local_vector_storage import LocalVectorDBClient
from ..splits import get_train_test, get_data

def load_logical_plans():

    dataset = get_data("query_plans", "json")

    def json_to_pandas(json_file):
        df = pd.json_normalize(json_file).explode(column='queryPlans').reset_index().drop(columns='index')
        with_query_plans = pd.json_normalize(df['queryPlans']).drop(columns=["applicationName", "applicationStartTime"])
        result = df.join(with_query_plans, how='inner').drop(columns='queryPlans')
        per_workload = result.groupby('applicationName')['logicalPlan'].last().reset_index()
        return per_workload

    dataset = json_to_pandas(dataset).reset_index().drop(columns="index")

    embeddings = get_embeddings(dataset['logicalPlan'].tolist())
    dataset['embeddings'] = embeddings
    return dataset.reset_index().drop(columns="index")

def get_best_trials():
    trials = get_data("trials", "csv")
    trials = trials.dropna()
    best_trials = trials.groupby("applicationName")['value'].idxmin()
    best_trials = trials.loc[best_trials]
    return best_trials

def get_top_k(embedding, vector_db: LocalVectorDBClient, k):
    similar_documents = vector_db.search(embedding, k)
    return similar_documents

def populate_vector_db(data, vector_db: LocalVectorDBClient):
    for _, row in tqdm(data.iterrows(), desc="Populating vector db"):
        embedding = row['embeddings']

        v = row.to_dict()
        v.pop('logicalPlan', None)
        v.pop('embeddings', None)    
        v.pop('applicationId', None)

        vector_db.create(
            row['applicationName'],
            v,
            embedding=embedding
        )

def weighted_sum(top_k):
    # use linear decay
    k = len(top_k)
    weights = np.linspace(1, 0, num=k)
    tot = np.sum(weights)
    return int(np.rint(np.sum(top_k * weights) / tot))



def compute_configs(top_k):
    return {
        'params_spark.driver.cores': weighted_sum(np.array([t['params_spark.driver.cores'] for t in top_k])),
        'params_spark.executor.cores': weighted_sum(np.array([t['params_spark.executor.cores'] for t in top_k])),
        'params_spark.executor.memory': weighted_sum(np.array([t['params_spark.executor.memory'] for t in top_k])),
        'params_spark.sql.shuffle.partitions': weighted_sum(np.array([t['params_spark.sql.shuffle.partitions'] for t in top_k])),
        'params_spark.executor.instances': weighted_sum(np.array([t['params_spark.executor.instances'] for t in top_k])),
        'params_spark.driver.memory': weighted_sum(np.array([t['params_spark.driver.memory'] for t in top_k])),
    }


if __name__ == "__main__":
    data = pd.merge(load_logical_plans(), get_best_trials(), on='applicationName', how='inner')
    train, test = get_train_test(data)

    vector_db = LocalVectorDBClient()
    populate_vector_db(train, vector_db)

    k = 5
    results = [] 
    for _, row in tqdm(test.iterrows(), desc="Computing KNN on test results"):
        result = {'applicationName': row['applicationName']}
        top_k = get_top_k(row['embeddings'], vector_db, k)
        top_k_params = [tk[0] for tk in top_k]
        result.update(**compute_configs(top_k_params))
        result['neighbours'] = [{'applicationName': t[0]['applicationName'], 'score': t[1]} for t in top_k]
        results.append(result)
    
    result_df = pd.DataFrame(results).drop(columns='neighbours')
    result_df.to_csv(f'src/knn/knn_{k}_results.csv', index=False)

    with open(f'src/knn/knn_{k}.json', 'w') as json_file:
        json.dump(results, json_file)

