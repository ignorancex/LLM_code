from functools import partial
import pickle
import pandas as pd
import logging
import optuna
import numpy as np
from optuna.samplers import GPSampler
from torch import mode

from ..utils import SPARK_DEFAULTS, SPARK_PARAMETER_RANGES, suggest_spark_configurations

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.addHandler(logging.FileHandler(filename='logs/sbo_optimization.log'))

STUDY_LOCATION = "src/sbo/trials"
NUMBER_OF_TRIALS = 100

# Load model
with open("src/sbo/model.pkl", 'rb') as pkl_file:
    model = pickle.load(pkl_file)


test_features = pd.read_csv("src/sbo/features_test.csv")

test_features = test_features.sort_values('applicationId').groupby("applicationName").first().reset_index()
test_workloads = test_features.sort_values('applicationId').to_dict('records')


def predict_runtime(trial, initial_execution_features):
    confs = suggest_spark_configurations(trial)

    feature_dict = initial_execution_features.copy()

    # modify the execution commands when predicting runtime
    feature_dict['params_spark.driver.cores'] = confs['driver_cores']
    feature_dict['params_spark.executor.memory'] = confs['executor_memory']
    feature_dict['params_spark.executor.cores'] = confs['executor_cores']
    feature_dict['params_spark.sql.shuffle.partitions'] = confs['shuffle_partitions']
    feature_dict['params_spark.executor.instances'] = confs['executor_instances']
    feature_dict['params_spark.driver.memory'] = confs['driver_memory']

    feature = np.array(list(feature_dict.values())).reshape(1, -1)

    return model.predict(feature)


def optimize_workload(workload_name, initial_features, initial_runtime):
    """Main function that runs optimization"""

    # create and save study
    study_name = f"sbo_100_{workload_name}"
    logger.info(f"Executing optimization of {study_name}")
    db_location = f"sqlite:///{STUDY_LOCATION}/{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=db_location, sampler=GPSampler(seed=0))
    
    distributions = { 
        k: optuna.distributions.IntDistribution(low=v['low'], high=v['high'], step=v['step']) for k, v in SPARK_PARAMETER_RANGES.items()
    }
    initial_trial = optuna.create_trial(
        params=SPARK_DEFAULTS, value=initial_runtime, distributions=distributions
    )
    # set initial state
    study.add_trial(initial_trial)
    
    # run
    study.optimize(partial(predict_runtime, initial_execution_features=initial_features), n_trials = NUMBER_OF_TRIALS)
    logger.info(f"Finished executing optimization of {study_name}")

    # Print the best trial and its result
    best_trial = study.best_trial
    logger.info(f"Best Trial - Params: {best_trial.params}") 
    logger.info(f"Best execution time: {best_trial.value}")
    
    return best_trial


results = []

for workload in test_workloads:
    name = workload['applicationName']
    initial_runtime = workload['value']
    # keep only features
    initial_features =  {key: value for key, value in workload.items() if key not in ['applicationName', 'applicationId', 'value']}
    best_result = optimize_workload(name, initial_features, initial_runtime)
    results.append({
        'query': name,
        'runtime': best_result.value,
        **best_result.params
    })

pd.DataFrame(results).to_csv("src/sbo/sbo_results.csv", index=False)