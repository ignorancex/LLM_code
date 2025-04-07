"""
Number Feature name
0 Input Metrics: Bytes Read
1 Executor Deserialize Time
2 Executor Deserialize CPU Time
3 Executor Run Time
4 Executor CPU Time
5 Result Size
6 JVM GC Time
7 Result Serialization Time
8 Memory Bytes Spilled
9 Disk Bytes Spilled
10 Shuffle Read Metrics: Remote Blocks Fetched
11 Shuffle Read Metrics: Local Blocks Fetched
12 Shuffle Read Metrics: Fetch Wait Time
13 Shuffle Read Metrics: Remote Bytes Read
14 Shuffle Read Metrics: Remote Bytes Read To Disk
15 Shuffle Read Metrics: Local Bytes Read
16 Shuffle Read Metrics: Total Records Read
17 bytesRead_sum: dataset size

Compute min, max, mean, std for each = 68 similarity features

"""
import os
import json
import pandas as pd
from ..splits import get_train_test, get_data
from tqdm import tqdm
import numpy as np

task_dataset = get_data("task_metrics", "json")
stage_dataset = get_data("stage_metrics", "json")


def json_to_pandas(json_obj):    
    df = pd.json_normalize(json_obj).explode(column='stageMetricsData').reset_index().drop(columns='index')
    with_metrics = pd.json_normalize(df['stageMetricsData'])
    result = df.join(with_metrics, how='inner').drop(columns='stageMetricsData')
    return result


def get_features_from_df(df):
    per_workload = df.groupby('uuid')

    #Add task number mean min max std from stage metrics
    features = per_workload.agg({
        'applicationName': 'first',
        'applicationId': 'first',
        'bytesRead': ['mean', 'min', 'max', 'std', 'sum'],
        'executorDeserializeTime': ['mean', 'min', 'max', 'std'],
        'executorDeserializeCpuTime': ['mean', 'min', 'max', 'std'],
        'executorRunTime': ['mean', 'min', 'max', 'std'],
        'executorCpuTime': ['mean', 'min', 'max', 'std'],
        'resultSize': ['mean', 'min', 'max', 'std'],
        'jvmGCTime': ['mean', 'min', 'max', 'std'],
        'resultSerializationTime': ['mean', 'min', 'max', 'std'],
        'memoryBytesSpilled': ['mean', 'min', 'max', 'std'],
        'diskBytesSpilled': ['mean', 'min', 'max', 'std'],
        'shuffleRemoteBlocksFetched': ['mean', 'min', 'max', 'std'],
        'shuffleLocalBlocksFetched': ['mean', 'min', 'max', 'std'],
        'shuffleFetchWaitTime': ['mean', 'min', 'max', 'std'],
        'shuffleRemoteBytesRead': ['mean', 'min', 'max', 'std'],
        'shuffleRemoteBytesReadToDisk': ['mean', 'min', 'max', 'std'],
        'shuffleLocalBytesRead': ['mean', 'min', 'max', 'std'],
        'shuffleRecordsRead': ['mean', 'min', 'max', 'std'],
    })
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.rename(columns={'applicationName_first': 'applicationName', 'applicationId_first': 'applicationId'})
    return features

print("Computing task dataset")
task_dataset = get_features_from_df(json_to_pandas(task_dataset))

print("Getting runtime configurations and info")
# replace inf trials with the max value in the application, and drop errored data
trials = get_data("trials", "csv")
max_values = trials.replace(float('inf'), np.nan).groupby('applicationName')['value'].transform('max')
trials['value'] = trials.apply(
    lambda row: max_values[row.name] if row['value'] == float('inf') else row['value'], axis=1
)
trials = trials[trials['value'].notnull()]

task_features = pd.merge(
    task_dataset, trials, 
    how='inner', 
    on=['applicationName', 'applicationId']
)

print("Processing stage data..")
# Process stage data
# from paper: https://upcommons.upc.edu/bitstream/handle/2117/340271/YORO_IEEE_TNSM_final_version_UPCversion-3.pdf?sequence=1
STAGE_DESCRIPTORS=["collect", "count", "countByKey", "first", "flatMap", "map", "reduce", "reduceByKey", "run-Job", "takeSample","treeAggregate"]

def get_stage_features_from_df(df):
    # attach stage descriptor count
    for descriptor in STAGE_DESCRIPTORS:
        df[f"{descriptor}_count"] = df['name'].str.count(descriptor)
    
    aggregations = {f"{d}_count": 'sum' for d in STAGE_DESCRIPTORS}

    per_workload = df.groupby('uuid')
    features = per_workload.agg({
        'applicationName': 'first',
        'applicationId': 'first',
        **aggregations
    })
    features = features.rename(columns={'applicationName_first': 'applicationName', 'applicationId_first': 'applicationId'})
    return features

stage_dataset = get_stage_features_from_df(json_to_pandas(stage_dataset))

all_features = pd.merge(
    task_features, stage_dataset, 
    how='inner', 
    on=['applicationName', 'applicationId']
)

all_features = all_features.drop(columns=['dataset','size','query_id'])


print("Spliting data")
train, test = get_train_test(all_features)

print("Writing results")
train.to_csv('src/sbo/features_train.csv', index=False)
test.to_csv('src/sbo/features_test.csv', index=False)