import optuna
import pandas as pd

from optuna.importance import MeanDecreaseImpurityImportanceEvaluator, FanovaImportanceEvaluator, get_param_importances

# Define a function to load study and extract required information

def load_study(dataset, number):
    study_name = f"{dataset}_1g_{number}_vanilla"
    database_file = f"data/{dataset}/{study_name}.db"
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{database_file}")


def extract_study_results(dataset, query_number):
    
    study = load_study(dataset, query_number)

    # Extract required information
    initial_trial_result = study.trials[0].value
    best_trial = study.best_trial
    best_trial_result = best_trial.value
    best_trial_index = best_trial.number
    best_trial_params = best_trial.params

    # Create a dictionary with the extracted information
    study_info = {
        'query': query_number,
        'initial_performance': initial_trial_result,
        'best_performance': best_trial_result,
        'best_trial_index': best_trial_index,
        'best_trial_params': best_trial_params
    }

    return study_info

def extract_parameter_importance(dataset, query_number):
    study = load_study(dataset, query_number)
    importance_evaluator = FanovaImportanceEvaluator()
    importances = get_param_importances(study, evaluator=importance_evaluator)
    importance_info =  {
        'query': query_number
    }
    importance_info.update(importances)
    return importance_info


# Create an empty list to store study information for all numbers
study_info_list = []
importance_info_list = []

# Iterate over numbers from 1 to 22
for query in range(1, 23):
    study_info = extract_study_results("{:02d}".format(query))
    study_info_list.append(study_info)

    importance_info = extract_parameter_importance("tpch" , "{:02d}".format(query))
    importance_info_list.append(importance_info)

# Create a DataFrame from the list of study information
overall_results = pd.DataFrame(study_info_list)
parameter_importances = pd.DataFrame(importance_info_list)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Print the DataFrame
#overall_results.to_csv('data/tpch_1g_vanilla_overall_results.csv', index=False)
#parameter_importances.to_csv('data/tpch_1g_vanilla_parameter_importances.csv', index=False)

fig = optuna.visualization.plot_optimization_history(load_study("tpch", "05"))
fig.show()
print(overall_results)
print("\n\n\n")
print(parameter_importances)