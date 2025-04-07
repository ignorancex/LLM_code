import os
import pandas as pd
from ..splits import get_data, get_train_test

from sklearn.preprocessing import MaxAbsScaler

dataset = get_data("stage_metrics", "csv")

print(dataset.info())

# Get runtimes and filter out runs which did not succeed (runtime of inf) and errored runs (runtime NaN)
runtimes = get_data("trials", "csv")
runtimes = runtimes[runtimes['value'].notnull()]
runtimes = runtimes[['applicationName', 'applicationId', 'value']]
runtimes = runtimes.rename(columns={'value': 'runtime'})
runtimes = runtimes[runtimes['runtime'] != float('inf')]

print(runtimes.info())

dataset = dataset.merge(runtimes, on=['applicationName', 'applicationId'], how='inner')
print(dataset.info())

# The following features were not gathered:
"""
    'avgOutSize',

"""
features = [
    'avgExecutorRuntime',
    'avgExecutorCputime',
    'bytesRead',
    'avgShuffleReadBytes',
    'avgShuffleWriteBytes',
    'avgMemoryBytesSpilled',
    'avgDiskBytesSpilled',
    'avgExecutorDeserializeTime',
    'avgExecutorDeserializeCpuTime',
    'avgResultSize',
    'numStages',
    'avgGCTime',
    'avgResultSerTime',
    'avgInputRecordsRead',
    'avgOutputRecordsWritten',
    'avgShuffleRemoteBlocksFetched',
    'avgShuffleLocalBlocksFetched',
    'avgShuffleFetchWaitTime',
    'avgShuffleRemoteBytesRead',
    'avgShuffleTotalBlocksFetched',
    'avgShuffleWriteTime',
    'avgShuffleRecordsWritten',
    'avgShuffleRecordsRead',
    'runtime'
]

def compute_mean(mean_col, sum_col):
    dataset[mean_col] = dataset[sum_col].astype(float) / dataset['numTasks'].astype(float)

compute_mean('avgExecutorRuntime', 'executorRunTime')
compute_mean('avgExecutorCputime', 'executorCpuTime')
compute_mean('avgShuffleReadBytes', 'shuffleTotalBytesRead')
compute_mean('avgShuffleWriteBytes', 'shuffleBytesWritten')
compute_mean('avgMemoryBytesSpilled', 'memoryBytesSpilled')
compute_mean('avgDiskBytesSpilled', 'diskBytesSpilled')
compute_mean('avgExecutorDeserializeTime', 'executorDeserializeTime')
compute_mean('avgExecutorDeserializeCpuTime', 'executorDeserializeCpuTime')
compute_mean('avgResultSize', 'resultSize')
compute_mean('avgGCTime', 'jvmGCTime')
compute_mean('avgResultSerTime', 'resultSerializationTime')
compute_mean('avgInputRecordsRead', 'recordsRead')
compute_mean('avgOutputRecordsWritten', 'recordsWritten')
compute_mean('avgShuffleRemoteBlocksFetched', 'shuffleRemoteBlocksFetched')
compute_mean('avgShuffleLocalBlocksFetched', 'shuffleLocalBlocksFetched')
compute_mean('avgShuffleFetchWaitTime', 'shuffleFetchWaitTime')
compute_mean('avgShuffleRemoteBytesRead', 'shuffleRemoteBytesRead')
compute_mean('avgShuffleTotalBlocksFetched', 'shuffleTotalBlocksFetched')
compute_mean('avgShuffleWriteTime', 'shuffleWriteTime')
compute_mean('avgShuffleRecordsWritten', 'shuffleRecordsWritten')
compute_mean('avgShuffleRecordsRead', 'shuffleRecordsRead')

# TODO: split by train/test/val

simtune_train, simtune_test = get_train_test(dataset)

train_queries = simtune_train[['applicationName', 'applicationId']]
test_queries = simtune_test[['applicationName', 'applicationId']]

simtune_train = simtune_train[[*features]]
simtune_test = simtune_test[[*features]]
scaler = MaxAbsScaler()

train = scaler.fit_transform(simtune_train)
test = scaler.transform(simtune_test)

pd.DataFrame(train).to_csv("src/simtune/features_train.csv", index=False, header=False)
pd.DataFrame(train_queries).to_csv("src/simtune/labels_train.csv", index=False, header=False)
pd.DataFrame(test).to_csv("src/simtune/features_test.csv", index=False, header=False)
pd.DataFrame(test_queries).to_csv("src/simtune/labels_test.csv", index=False, header=False)