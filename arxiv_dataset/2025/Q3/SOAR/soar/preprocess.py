import copy 
import json
# rename to get_dataset.py

def merge_GT(data,data_GT):
    """merge ground truth data to data (correct output for the test example)"""
    data_merged = copy.deepcopy(data)
    for puzzle in data_GT:
        assert len(data_GT[puzzle]) == len(data_merged[puzzle]['test'])
        for i in range(len(data_GT[puzzle])):
            data_merged[puzzle]['test'][i]['output']=data_GT[puzzle][i] 
        # data_merged[puzzle]['test'][0]['output']=data_GT[puzzle][0]
    return data_merged


def get_dataset(data_path="/home/flowers/work/arc/aces_arc/",arc_2=False):
    """
    get train, val, test dataset:
       - arc-agi_training-challenges.json: contains input/output pairs that demonstrate reasoning pattern to be applied to the "test" input for each task. This file and the corresponding solutions file can be used as training for your models.
       - arc-agi_training-solutions.json: contains the corresponding task "test" outputs (ground truth).
       - arc-agi_evaluation-challenges.json: contains input/output pairs that demonstrate reasoning pattern to be applied to the "test" input for each task. This file and the corresponding solutions file can be used as validation data for your models.
       - arc-agi_evaluation-solutions.json: contains the corresponding task "test" outputs (ground truth).
       - arc-agi_test-challenges.json: this file contains the tasks that will be used for the leaderboard evaluation, and contains "train" input/output pairs as well as the "test" input for each task. Your task is to predict the "test" output. Note: The file shown on this page is a placeholder using tasks from arc-agi_evaluation-challenges.json. When you submit your notebook to be rerun, this file is swapped with the actual test challenges.
       - sample_submission.json: a submission file in the correct format
    """
    folder_arc_1 = "arc-prize-2024/"
    folder_arc_2 = "arc-prize-2025/"
    if arc_2:
        data_path = data_path + folder_arc_2
    else:
        data_path = data_path + folder_arc_1
    
    path_trainset = data_path+"arc-agi_training_challenges.json"
    path_trainset_GT = data_path+"arc-agi_training_solutions.json"
    path_valset = data_path+"arc-agi_evaluation_challenges.json"
    path_valset_GT = data_path+"arc-agi_evaluation_solutions.json"
    path_testset = data_path+"arc-agi_test_challenges.json"
    # path_sample_submission = data_path+"sample_submission.json"
    print(data_path)
    print(path_trainset)
    with open(path_trainset) as f:
        data_train = json.load(f)
    with open(path_trainset_GT) as f:
        data_train_GT = json.load(f)
    with open(path_valset) as f:
        data_val = json.load(f)
    with open(path_valset_GT) as f:
        data_val_GT = json.load(f)
    with open(path_testset) as f:
        data_test = json.load(f)
    train_data = merge_GT(data_train,data_train_GT)
    val_data = merge_GT(data_val,data_val_GT)
    test_data = data_test
    return train_data, val_data, test_data


def convert_to_list(arrays):
    """convert a array of array of array of int64 to a list of square list of int """
    result = []
    for array in arrays:
        converted_array = []
        for sub_array in array:
            converted_array.append(sub_array.astype(int).tolist())
        result.append(converted_array)
    return result

def reformat_solutions_parquet(df):
    """Reformat solutions from a DataFrame to a dictionary of task_id -> list of solutions."""
    dict_solutions = {}
    for task_id, group in df.groupby('task_id'):
        dict_solutions[task_id] = group.to_dict(orient='records')
    for task_id, solutions in dict_solutions.items():
        for solution in solutions:
            solution['predicted_train_output'] = convert_to_list(solution['predicted_train_output'])
            solution['predicted_test_output'] = convert_to_list(solution['predicted_test_output'])
    return dict_solutions