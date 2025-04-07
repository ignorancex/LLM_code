import os
import pandas as pd
import json 
import numpy as np

from splits import TPCH_TRAIN, TPCH_TEST, TPCDS_TRAIN, TPCDS_TEST, get_train_test
import numpy as np
from local_vector_storage import LocalVectorDBClient
from embeddings import get_embeddings
from utils import SPARK_PARAMETER_RANGES

def preprocess_all_queries(trials_path='trials.csv', model=None):
    """
    Preprocesses all queries by loading logical plans and separating them into train and test sets.

    Args:
        minimum_dataset_size (int): The minimum dataset size for a query to be included in the processed queries. Default is 0.
        directory_path (str): The directory path where the logical plan files are located. Default is "data/processed_data/query_plans/".
        datasets (list): The list of datasets to process. Default is [TPCDS, TPCH].
        dataset_sizes (list): The list of dataset sizes to process. Default is DATASET_SIZES.

    Returns:
        tuple: A tuple containing the train and test trials for TPCDS and TPCH datasets.

    """
    LOGICAL_PLAN_FILES = []

    for task_dataset in ['tpch', 'tpcds']:
        for size in ['250g', '750g', '100g', '500g']:
            path = f"spark-tuning/training/data/{task_dataset}/{task_dataset}_{size}_query_plans.json"
            if os.path.exists(path):
                LOGICAL_PLAN_FILES.append(path)
            else:
                print(f"File not found: {path}")

    trials = pd.read_csv(trials_path)
    logical_plans_dict = load_all_logical_plans(LOGICAL_PLAN_FILES)

    best_trials = get_best_trials(trials)
    first_trials = trials.groupby("applicationName").first()


    # Logical plans for each query in the best trials
    best_logical_plans = {}
    for _, trial in best_trials.iterrows():
        # if len(best_logical_plans) == 50:
        #     break
        plans = get_logical_plan(logical_plans_dict, trial['applicationName'], trial['applicationId'])
        # get the first logical plan by replacing applicationId with 0 after the very last _
        first_application_id = trial['applicationId'].rsplit('_', 1)[0] + '_0'
        first_plan = get_logical_plan(logical_plans_dict, trial['applicationName'], first_application_id)
        if first_plan is None:
            print(f"Could not find FIRST logical plan for application name {trial['applicationName']} - {first_application_id}")
            print('getting for second')
            first_application_id = trial['applicationId'].rsplit('_', 1)[0] + '_1'
            first_plan = get_logical_plan(logical_plans_dict, trial['applicationName'], first_application_id)
        if plans and first_plan:
            best_logical_plans[trial['applicationName']] = {
                'best': plans,
                'default': first_plan
            }
        else:
            # print(f"Could not find logical plan for application name {trial['applicationName']}")
            # remove the trial from the best trials
            best_trials = best_trials[best_trials["applicationName"] != trial['applicationName']]
            # first_trials = first_trials[first_trials["applicationName"] != trial['applicationName']]

    print(f"Number of best logical plans: {len(best_logical_plans)}")
    print("Sample best logical plans:")
    # print(best_logical_plans)

    # Load embeddings for logical plans
    embeddings = get_embeddings_for_logical_plan(best_logical_plans, model)

    return embeddings, best_trials
def get_best_trials(trials):
    """
    Get the best trials for each application name.

    Args:
        trials (pd.DataFrame): A DataFrame containing the trials.

    Returns:
        pd.DataFrame: A DataFrame containing the best trials for each application name.

    """
    trials = trials.dropna()
    best_trials = trials.groupby("applicationName")['value'].idxmin()
    best_trials = trials.loc[best_trials]

    print(f"Number of best trials: {len(best_trials)}")
    print("Best trials:")
    print(best_trials.head())

    return best_trials


def load_all_logical_plans(logical_plan_file_paths):
    """
    Load all logical plans from the given file paths. The files are large and loading the files multiple times can be slow.

    Args:
        logical_plan_file_paths (list): A list of file paths containing logical plans.

    Returns:
        dict: A dictionary where the keys are the file paths and the values are the loaded logical plans.
    """

    loaded_logical_plan_files = {}

    for file in logical_plan_file_paths:
        print("Loading", file)
        with open(file, "r") as f:
            data = json.load(f)
            loaded_logical_plan_files[file] = data

    return loaded_logical_plan_files

def get_application_id(dataset_name, dataset_size, query_num):
    """
    Generates an application ID based on the dataset name, dataset size, and query number.

    Args:
        dataset_name (str): The name of the dataset.
        dataset_size (int): The size of the dataset in gigabytes.
        query_num (str): The query number formatted as a string.

    Returns:
        str: The generated application ID.

    """
    return f"{dataset_name}_{dataset_size}_{query_num}"

def get_embeddings_for_logical_plan(queries_dict, model):
    """
    Retrieves embeddings for logical plans in the given queries.

    Args:
        queries (list): A list of queries containing logical plans.

    Returns:
        None

    Modifies:
        Each query in the input list is modified to include embeddings for default and best logical plans.
    """

    default_logical_plans = []
    best_logical_plans = []
    for query in queries_dict.values():
        default_logical_plans.append(query["default"])
        best_logical_plans.append(query["best"])

    print("Getting embeddings for default logical plans Num:", len(default_logical_plans))

    default_embeddings = get_embeddings(
        default_logical_plans, model_name_or_path=model#"thenlper/gte-base" #jinaai/jina-embeddings-v3"
    )

    print("Getting embeddings for best logical plans Num:", len(best_logical_plans))
    best_embeddings = ["" for _ in range(len(best_logical_plans))]
    #get_embeddings(
        # best_logical_plans, model_name_or_path="jinaai/jina-embeddings-v3"
    # )

    for i, query in enumerate(queries_dict.values()):
        query["embeddings"] = {}
        query["embeddings"]["default"] = default_embeddings[i]
        query["embeddings"]["best"] = best_embeddings[i]

    return queries_dict

def get_logical_plan(
    logical_plans_dict,
    application_name,
    application_id
):
    """
    Load the logical plan for a given query and trial from a JSON file.

    Args:

    Returns:
        dict: The logical plan or optimized plan for the specified query and trial.

    Raises:
        AssertionError: If the specified query and trial cannot be found in the JSON file.

    """

    found_plan = None

    # Go through the logical plans and see if the application name matches one in the file
    for file, data in logical_plans_dict.items():
        for plan in data:
            if plan["applicationName"] == application_name:
                if plan['applicationId'] == application_id:
                    found_plan = plan
                   

    if found_plan is None:
        print(f"Could not find logical plan for application name {application_name} - {application_id}")
        return None
    
    # go into queryPlans and get the last one
    found_plan = found_plan["queryPlans"][-1]['logicalPlan']                

    return found_plan

    # trial_num -= 1  # Since the data is saved as 0 indexed

    # # Find all trials related to the current query
    # trials = [
    #     trial for trial in data if trial["applicationName"].endswith("_" + query_num)
    # ]

    # trial = [
    #     trial
    #     for trial in trials
    #     if trial["applicationId"].endswith("_" + str(trial_num))
    # ]

    # assert (
    #     len(trial) >= 1
    # ), f"Found {len(trials)} trials for query {query_num} but could not find trial {trial_num} in {path}"

    # logical_plan = trial[0]["queryPlans"][-1]

    # if optimized:
    #     return logical_plan["optimizedPlan"]

    # return logical_plan["logicalPlan"]

def add_to_vector_db(vectorDB, best_trials, embeddings):
    """
    Adds trials to the vectorDB if the dataset size meets the minimum requirement.

    Args:
        vectorDB (VectorDB): The vector database to add the trials to.
        trials (list): A list of trial metadata.
        minimum_dataset_size (int, optional): The minimum dataset size required for a trial to be added. Defaults to 0.
    """

    for _,trial in best_trials.iterrows():
        # print(trial)

        id = trial["applicationName"]
        # print("ID:", id)
        vectorDB.create(trial['applicationId'], 
                        trial, 
                        embedding=embeddings[trial['applicationName']]["embeddings"]["default"])

    print("Added to vector db: ", len(vectorDB))

# List of relevant columns
columns_to_normalize = [
    'params_spark.driver.cores',
    'params_spark.executor.memory',
    'params_spark.executor.cores',
    'params_spark.sql.shuffle.partitions',
    'params_spark.executor.instances',
    'params_spark.driver.memory'
]
# Function to calculate normalized distance
def calculate_normalized_value(param_name, param_value):
    param_range = SPARK_PARAMETER_RANGES.get(param_name.replace('params_', ''), None)
    if param_range:
        return (param_value - param_range['low']) / (param_range['high'] - param_range['low'])
    return None

def harmonic_mean(values):
    if len(values) == 0:
        return 0  # or raise an error if preferred
    if any(v == 0 for v in values):
        raise ValueError("Cannot calculate harmonic mean for values containing zero.")
    
    n = len(values)
    reciprocal_sum = sum(1 / x for x in values)
    return n / reciprocal_sum
import math

def exponential_decay_average(values, decay_constant=0.5):
    """
    Calculate the exponential decay average of a list of values.

    :param values: List of values to calculate the average for.
    :param decay_constant: The decay constant (lambda) to control the decay rate.
    :return: The exponential decay average.
    """
    if not values:
        return 0  # or raise an error if preferred

    decay_average = 0  # Initialize the decay average
    total_weight = 0  # Total weight to normalize the average

    for t, value in enumerate(values):
        weight = math.exp(-decay_constant * t)  # Calculate weight based on time step
        decay_average += weight * value  # Weighted value
        total_weight += weight  # Accumulate total weight

    return decay_average / total_weight  # Normalize by total weight
def calc_distance(vectorDB, trials, embeddings, k=1, method='mean'):
    distances = []
    all_rmse = []
    for _, trial in trials.iterrows():
        embedding = embeddings[trial['applicationName']]["embeddings"]["default"]
        sim_queries = vectorDB.search(embedding, k)

        # print("SIM QUERIES: ", sim_queries)

        # average out all the values for columns_to_normalize
        avg_values = {
            'params_spark.driver.cores':0,
            'params_spark.executor.memory':0,
            'params_spark.executor.cores':0,
            'params_spark.sql.shuffle.partitions':0,
            'params_spark.executor.instances':0,
            'params_spark.driver.memory':0
        }
        for query in sim_queries:
            query = query[0]
            for col in columns_to_normalize:
                # print(query)
                # print("COL", col)
                avg_values[col] += query[col]
    
        # divide by k
        for col in columns_to_normalize:
            if method == 'mean':
                avg_values[col] /= k

        
        ground_truth_values = {
            'params_spark.driver.cores': trial['params_spark.driver.cores'],
            'params_spark.executor.memory': trial['params_spark.executor.memory'],
            'params_spark.executor.cores': trial['params_spark.executor.cores'],
            'params_spark.sql.shuffle.partitions': trial['params_spark.sql.shuffle.partitions'],
            'params_spark.executor.instances': trial['params_spark.executor.instances'],
            'params_spark.driver.memory': trial['params_spark.driver.memory']
        }

        # Calculate RMSE between ground truth and average values
        squared_differences = []

        for key in ground_truth_values:
            ground_truth_normalized = calculate_normalized_value(key, ground_truth_values[key])
            avg_normalized = calculate_normalized_value(key, avg_values[key])
            
            if ground_truth_normalized is not None and avg_normalized is not None:
                # Calculate squared difference
                squared_difference = (ground_truth_normalized - avg_normalized) ** 2
                squared_differences.append(squared_difference)

        # Calculate RMSE
        rmse = np.sqrt(np.mean(squared_differences))
        all_rmse.append(rmse)

    return np.sqrt(np.mean(all_rmse)), 0

def get_folds(dataset, n_splits=10):
    # Define train and test sets based on application names
    train = dataset[dataset['applicationName'].apply(lambda appName: any([appName.startswith("tpch") and appName.endswith(q) for q in TPCH_TRAIN] + [appName.startswith("tpcds") and appName.endswith(q) for q in TPCDS_TRAIN]))]
    test = dataset[dataset['applicationName'].apply(lambda appName: any([appName.startswith("tpch") and appName.endswith(q) for q in TPCH_TEST] + [appName.startswith("tpcds") and appName.endswith(q) for q in TPCDS_TEST]))]

    # Extract the queries (q) as groups from the 'applicationName'
    groups = train['applicationName'].apply(lambda appName: next(q for q in TPCH_TRAIN + TPCDS_TRAIN if appName.endswith(q)))
    
    # Initialize GroupKFold to ensure each group (q) is entirely in one fold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Create a list of training and validation sets from the group-wise k-fold split
    folds = []
    for train_index, val_index in gkf.split(train, groups=groups):
        train_fold = train.iloc[train_index]
        val_fold = train.iloc[val_index]
        folds.append((train_fold, val_fold))
    
    return folds, test

if __name__ == "__main__":
    
    models = ['jinaai/jina-embeddings-v3', 'WhereIsAI/UAE-Large-V1', 'dunzhang/stella_en_400M_v5', 'Alibaba-NLP/gte-large-en-v1.5', 'BAAI/bge-large-en-v1.5' ]


    mean_rmse_values_per_model = {}

    for model in models:
        
        embeddings, best_trials = preprocess_all_queries(
            trials_path = 'spark-tuning/training/data/trials.csv', model=model)

        vectorDB = LocalVectorDBClient()


        
        from sklearn.model_selection import GroupKFold

        train, test = get_train_test(best_trials)

        folds, test = get_folds(best_trials, n_splits=10)

        average_per_fold = []

        k_test_values = range(1, 100, 2)

        rmse_values_mean = [0] * len(k_test_values)

        for train, val in folds:

            add_to_vector_db(vectorDB, train, embeddings)



            for i, k in enumerate(k_test_values):
                rmse, _ = calc_distance(vectorDB, val, embeddings, k=k, method='mean')
                rmse_values_mean[i] += rmse/len(folds)

        mean_rmse_values_per_model[model] = rmse_values_mean


    # Convert the dictionary to a DataFrame
    
    # Update the DataFrame's columns with the custom header
    df = pd.DataFrame.from_dict(mean_rmse_values_per_model, orient='index', columns=k_test_values)

    df.index.name = 'Model'
    # Reset the index to move the model names into the DataFrame as a column
    df.reset_index(inplace=True)
    # Save to CSV
    df.to_csv('k_values_embedding_models.csv', index=False)

    import matplotlib.pyplot as plt
    # plot all the model from the dict
    for model in models:
        plt.plot(k_test_values, mean_rmse_values_per_model[model], label=model)

    # plt.plot(k_values, rmse_values_harmonic, label='harmonic')
    # plt.plot(k_values, rmse_values_exponential, label='exponential')
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('k vs RMSE')
    plt.show()
    plt.savefig('k_vs_rmse.png')

    print("Finished, saved to k_values_embedding_models.csv")
    print("Plotted k vs RMSE")
    print("Saved plot to k_vs_rmse.png")

