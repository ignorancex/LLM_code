from utils import *
from time import sleep
from functools import partial

import numpy as np 
from skopt import gp_minimize
from skopt.space import Space
from skopt.utils import use_named_args
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.base import clone
from skopt.space import Integer, Real, Categorical

from monitoring import SQLiteStorage

import argparse
# import optuna
import logging
import sys

# Example usage
TPCH_CONTAINER_ID = "tpch-spark35_tpch_1"
TPCDS_CONTAINER_ID = "tpcds-spark35-tpcds-1"
logger: logging.Logger = None

QUERIES = ["{:02d}".format(n) for n in range(1, 23)]


class BayesianOptimizationStudy:
    def __init__(self, objective_function, parameter_ranges, storage=None, n_calls=10, random_state=None):
        self.objective_function = objective_function
        self.space = self.create_space(parameter_ranges)
        self.n_calls = n_calls
        self.random_state = random_state
        self.storage = storage
        self._trials = []
        self._best_trial = None
        self.surrogate_model = None

    def suggest(self, params=None):
        if params is None:
            params = self.space.rvs(random_state=self.random_state)
        return dict(zip(self.space.dimensions, params))

    def create_space(self, parameter_ranges):
        space = []
        for param_name, param_range in parameter_ranges.items():
            if isinstance(param_range["low"], int) and isinstance(param_range["high"], int):
                space.append(Integer(param_range["low"], param_range["high"], name=param_name))
            elif isinstance(param_range["low"], float) and isinstance(param_range["high"], float):
                space.append(Real(param_range["low"], param_range["high"], name=param_name))
            else:
                raise ValueError(f"Unsupported parameter range for {param_name}")
        return Space(space)

    def evaluate_objective(self, params):
        param_dict = {dim.name: value for dim, value in zip(self.space.dimensions, params)}
        trial = {'params': param_dict, 'value': self.objective_function(param_dict, trial_number=len(self._trials)+1)}
        self._trials.append(trial)
        if self._best_trial is None or trial['value'] < self._best_trial['value']:
            self._best_trial = trial

        logger.info(f"Iteration {len(self._trials)} - Current Time: {trial['value']} - Best objective value: {self._best_trial['value']}")

        if self.storage:
            self.storage.create_trial(param_dict, trial['value'])

        return trial['value']

    def optimize(self, initial_trial=None):
        logger.info(f"Starting optimization for {self.n_calls} trials.")

        # Perform Bayesian optimization using Gaussian process
        optimization_result = gp_minimize(
            self.evaluate_objective,
            dimensions=self.space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            x0=initial_trial,
            n_initial_points=0,
        )
        self.surrogate_model = optimization_result.models[-1]


    def get_surrogate_model(self):
        if self.surrogate_model is None:
            logger.warning("Surrogate model is not available. Please run optimization first.")
            return None
        return clone(self.surrogate_model)

    @property
    def trials(self):
        if self.storage:
            return self.storage.get_all_trials()
        else:
            return self._trials

    @property
    def best_trial(self):
        return self._best_trial


def create_study(objective_function, space, n_calls=10, storage=None, random_state=None):
    return BayesianOptimizationStudy(objective_function, space, n_calls=n_calls, storage=storage, random_state=random_state)

def parse_execution_time(execution_data):
    """
    Parses the execution time out from the execution data collected from running tpch
    """
    return float(execution_data.split("\n")[-2].split("\t")[-1])

def run_tpch_execution(id, size, query, confs):
    """
    Run query, get execution time, and copy over data
    """
    
    updated_confs = generate_confs_for_spark_submit(confs)

    directory_name = f"/data_tpch_{size}_{query}_{id}"
    docker_exec(TPCH_CONTAINER_ID, ['mkdir', directory_name])
    
    # run query but mark as inf if the execution fails for whatever reason
    try:
        docker_exec(TPCH_CONTAINER_ID, ['/scripts/run_tpch.sh', id, size, query, updated_confs])
    except Exception as e:
        logger.exception(e)
        return float('inf')

    sleep(5) # race condition where the listeners may not have finished writing the files after the spark app succeeded and before we copy it

    # get execution time
    execution_data = docker_exec(TPCH_CONTAINER_ID, ["cat", f"{directory_name}/tpch_execution_times.txt"])

    
    # copy over metrics and delete them in hdfs
    copy_training_data_to_local(TPCH_CONTAINER_ID, directory_name, f"{TRAINING_DATA_LOCAL_DIR}/tpch")
    docker_exec(TPCH_CONTAINER_ID, ['rm', '-r', directory_name])

    return parse_execution_time(execution_data)

def run_tpcds_execution(id, size, query, confs):
    """
    Run query, get execution time, and copy over data
    """
    updated_confs = generate_confs_for_spark_submit(confs)

    logger.info(f"\n\nStarting trial: {id}, \n   Size: {size}, \n   q: {query} \n   Conf: {confs}")

    directory_name = f"/tpcds-spark/data_tpcds_{size}_{query}_{id}"
    docker_exec(TPCDS_CONTAINER_ID, ['mkdir', directory_name])
    
    # run query but mark as inf if the execution fails for whatever reason
    try:
        docker_exec(TPCDS_CONTAINER_ID, ['/scripts/run_tpcds.sh', id, size, query, updated_confs])
    except Exception as e:
        logger.exception(e)
        return float('inf')

    sleep(5) # race condition where the listeners may not have finished writing the files after the spark app succeeded and before we copy it

    # get execution time
    execution_data = docker_exec(TPCDS_CONTAINER_ID, ["cat", f"{directory_name}/../output_result/TIMES.txt"])

    
    # copy over metrics and delete them in hdfs
    copy_training_data_to_local(TPCDS_CONTAINER_ID, directory_name, f"{TRAINING_DATA_LOCAL_DIR}/tpcds")
    docker_exec(TPCDS_CONTAINER_ID, ['rm', '-r', directory_name])

    return parse_execution_time(execution_data)


def evaluate_process(trial_confs, id, size, query, dataset, trial_number=0):
    """Objective function that runs tpch on container and retrieves execution time"""
    # confs = suggest_spark_configurations(trial)
    # Rename some of the keys to match previous method 
    trial_confs = {
        "executor_instances": trial_confs["spark.executor.instances"],
        "executor_memory": trial_confs["spark.executor.memory"],
        "executor_cores": trial_confs["spark.executor.cores"],
        "driver_memory": trial_confs["spark.driver.memory"],
        "driver_cores": trial_confs["spark.driver.cores"],
        "shuffle_partitions": trial_confs["spark.sql.shuffle.partitions"]
    }

    if dataset == "tpch":
        return run_tpch_execution(f"{id}_{trial_number}", size, query, trial_confs)
    elif dataset == "tpcds":
        return run_tpcds_execution(f"{id}_{trial_number}", size, query, trial_confs)


def optimize_queries(id, size, query, dataset):
    """Main function that runs optimization per query"""

    # create and save study
    study_name = f"{dataset}_{size}_{query}_{id}"
    logger.info(f"Executing optimization of {study_name}")
    db_location = f"{TRAINING_DATA_LOCAL_DIR}/{dataset}/{study_name}.db"
    # Create study 
    study = create_study(partial(evaluate_process, id=id, size=size, query=query, dataset=dataset), SPARK_PARAMETER_RANGES, storage=SQLiteStorage(db_location), n_calls=NUMBER_OF_TRIALS, random_state=42)
    
    
    # run
    initial_trial = [SPARK_DEFAULTS[param.name] for param in study.space]
    study.optimize(initial_trial=initial_trial)
    logger.info(f"Finished executing optimization of {study_name}")
    
    # print results
    initial_trial = study._trials[0]
    logger.info(f"Initial execution time: {initial_trial['value']}")

    # Print the best trial and its result
    best_trial = study.best_trial
    logger.info(f"Best Trial - Params: {best_trial['params']}") 
    logger.info(f"Best execution time: {best_trial['value']}")
    
    print("Surrogate model: ", study.get_surrogate_model())

    return


def run(id, size, queries, dataset):
    """Runs and tries to optimize all the queries"""
    logger.info(f"Starting runs for {dataset} with size {size} and id {id}")
    for query in queries:
        optimize_queries(id, size, query, dataset)
    
    logger.info(f"Done running {dataset} for size {size}. Results have an id of {id}")


parser = argparse.ArgumentParser(description='Arguments for generating training data with a specific dataset.')
parser.add_argument('--size', type=str, help='The size parameter')
parser.add_argument('--id', type=str, help='The id parameter')
parser.add_argument('--queries', type=str, nargs='+', default=QUERIES)
parser.add_argument('--dataset', type=str, default="tpch", choices=["tpch", "tpcds"])

# Parse the command-line arguments
args = parser.parse_args()

# Run
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(f"logs/{args.dataset}_{args.size}_{args.id}.log"))
logger.addHandler(logging.StreamHandler(sys.stdout))

# optuna.logging.enable_propagation()  # Propagate logs to the root logger.
# optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr.

logger.info(args.queries)

run(args.id, args.size, args.queries, args.dataset)
