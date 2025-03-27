# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:26:05 2024

@author: dansc
"""
import random
random.seed(56)
import json
import re
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel
    )

# =============================================================================
# 
# =============================================================================

def simple_True_False_f1(true_false_list):
    label_prediction_list = [] 
    for item in true_false_list:
        if item.upper() == "TRUE":
            label_prediction_list.append(1)
        if item.upper() == "FALSE":
            label_prediction_list.append(0)
    return label_prediction_list

# =============================================================================
# 
# =============================================================================
    
def get_results(file, model, data_dict):
    '''
    Takes in a file path and the name of the model used for predictions.

    Parameters:
    - file (str): The file path to the results file containing model predictions.
    - model (str): The name of the model for which results are being analyzed.

    Reads the results from the specified file, extracts predictions, and computes evaluation metrics.

    Returns:
    None

    Prints:
    - Model name (str): The name of the model.
    - Accuracy (float): The accuracy of the model predictions.
    - Macro F1 Score (float): The macro-averaged F1 score.
    - Binary F1 Score (float): The binary F1 score.

    Example:
    get_results('/path/to/results.txt', 'MyModel')
    '''
    
    results_list = []
    with open(file,'r') as infile:
        results = infile.read()
    results = results.strip().split('\n')
    for line in results:
        results_list.append(json.loads(line)['output'])
    
    predictions_list = []
    for item in results_list:
        if item.upper() == 'TRUE':
            predictions_list.append(1)
        if item.upper() == 'FALSE':
            predictions_list.append(0)
                
    data_dict['prediction'] = predictions_list
    
    print(f'{model.upper()}:')
    acc = accuracy_score(data_dict['label'], predictions_list)
    print('Accuracy:', acc)
    f1_macro = f1_score(data_dict['label'], predictions_list, average = 'macro')
    print('Macro F1 Score:', f1_macro)
    
    f1_binary = f1_score(data_dict['label'], predictions_list)
    print('Binary F1 Score:', f1_binary)

# =============================================================================
# 
# =============================================================================
    
def scores(ground_truth, predictions):
    '''
    Calculates and prints accuracy, macro F1 score, and binary F1 score based on ground truth and predictions.

    Parameters:
    - ground_truth (array-like): True labels.
    - predictions (array-like): Predicted labels.

    Prints:
    - Accuracy (float): The accuracy of the predictions.
    - Macro F1 Score (float): The macro-averaged F1 score.
    - Binary F1 Score (float): The binary F1 score.

    Notes:
    - If there are values in the input arrays not equal to 1 or 0 they will be replaced as 0s.

    Example:
    scores(ground_truth=[1, 0, 1, 1], predictions=[1, 0, 4, 1])
    '''
    for i, pred in enumerate(predictions):
        if pred not in {0, 1}:
            predictions[i] = 0
            
    f1_binary = f1_score(ground_truth, predictions, average='binary', pos_label=1)
    f1_macro = f1_score(ground_truth, predictions, average='macro')
    acc = accuracy_score(ground_truth, predictions)
    print('Accuracy:', acc)
    print('Macro F1 Score:', f1_macro)
    print('Binary F1 Score:', f1_binary)
        
# =============================================================================
# 
# =============================================================================

def extract_output_values_from_json_file(file):
    '''
    Reads a file containing JSON-formatted results and returns a list of output values.

    Parameters:
    - file (str): The file path to the JSON results file.

    Returns:
    list: A list containing the 'output' values extracted from each line of the JSON file.

    Example:
    results_list = extract_output_values_from_json_file('/path/to/results.json')
    '''
    results_list = []  
    with open(file, 'r') as infile:
        results = infile.read()
        results = results.strip().split('\n')
        for line in results:
            results_list.append(json.loads(line)['output'])
    return results_list

# =============================================================================
# 
# =============================================================================

def read_json_lines_and_extract_output(file):
    '''
    Reads a file containing JSON-formatted results line by line, extracts 'output' values, and returns a list.

    Parameters:
    - file (str): The file path to the JSON results file.

    Returns:
    list: A list containing the 'output' values extracted from each valid JSON line in the file.
          If a line is not a valid JSON or lacks the 'output' key, it is skipped.

    Example:
    output_values = read_json_lines_and_extract_output('/path/to/results.json')

    Note:
    This function reads the specified file line by line, attempting to extract 'output' values from each JSON line.
    Lines that are not valid JSON or lack the 'output' key are skipped.

    Example Usage:
    >>> output_values = read_json_lines_and_extract_output('/path/to/results.json')
    >>> print(output_values)
    [value1, value2, ..., valueN]
    '''
    results_list = []  # Initialize the list before the loop
    with open(file, 'r') as infile:
        for line in infile:
            try:
                result_data = json.loads(line)
                output_value = result_data['output']
                results_list.append(output_value)
            except json.JSONDecodeError as e:
                pass
    return results_list

# =============================================================================
# 
# =============================================================================
         
# r'THE ANSWER CANDIDATE IS (TRUE|FALSE)\b'
def re_scan(pattern, lst):
    '''
    Scans a list of strings and extracts binary labels based on patterns.

    Parameters:
    - lst (list): A list of strings to be scanned.

    Returns:
    list: A list of binary labels (1 for 'TRUE', 0 for 'FALSE') extracted from each string.
          Returns None if no match is found or 99 if the match group is neither 'TRUE' nor 'FALSE'.

    Example:
    labels = re_scan(['The answer candidate is TRUE.', 'No relevant information.'])
    '''
    filtered = []
    for line in lst:
        line = line.upper()
        match = re.search(pattern, line)
        if match:
            if match.group(1) == 'TRUE':
                filtered.append(1)
            elif match.group(1) == 'FALSE':
                filtered.append(0)
            else:
                filtered.appen(99)
        else:
            filtered.append(None)
    return filtered

# =============================================================================
# 
# =============================================================================

def count_true_false(lst):
    '''
    Counts the occurrences of 'True' (1), 'False' (0), and 'None' values in a list.

    Parameters:
    - lst (list): A list containing binary labels (1 for 'True', 0 for 'False') and/or None values.

    Prints:
    - Total count (int): The total number of elements in the list.
    - True count (int): The number of 'True' (1) occurrences in the list.
    - False count (int): The number of 'False' (0) occurrences in the list.
    - None count (int): The number of 'None' occurrences in the list.

    Example:
    count_true_false([1, 0, None, 1, 0, None, None])

    Output:
    Total: 7
    True: 2
    False: 2
    None: 3
    '''
    total_count = 0
    true_count = 0
    false_count = 0
    none_count = 0
    for item in lst:
        total_count += 1
        if item == 1:
            true_count += 1
        elif item == 0:
            false_count += 1
        else:
            none_count += 1
    print('Total:',
            total_count,
            '\nTrue:',
            true_count,
            '\nFalse:',
            false_count,
            '\nNone:',
            none_count
            )
    
# =============================================================================
#     
# =============================================================================


def cot_process_output(output_list, falsify_or_steal = 'falsify', steal_source = './json_results/pe_gpt-4-1106-preview.out'):
    """
    Process a list of model outputs and convert them to binary values (0 or 1) based on specific patterns.

    Parameters:
    - output_list (list): A list of model output strings to be processed.
    - falsify_or_steal (str, optional): The strategy to handle unmatched patterns.
        - 'falsify': Force unmatched patterns to be False (0).
        - 'steal': Steal values from a pre-existing model (Simple_GPT4's Output).
        - 'none': Keep the value as None for unmatched patterns. Default is 'falsify'.

    Returns:
    list: A list of binary values (0 or 1) corresponding to the processed model outputs.

    This function processes each model output in the input list and converts it to a binary value
    based on specific patterns. It supports matching patterns related to labels, answer candidates,
    likelihood, and support. Unmatched patterns are handled according to the specified strategy.

    Example:
    >>> output_list = ["Label: TRUE", "The answer candidate is TRUE", "Likely to be FALSE", "Supports the answer candidate"]
    >>> cot_process_output(output_list, falsify_or_steal='steal')

    Output:
    [1, 1, 0, 1]
    """
    ones_zeros_list = []

    for i, out in enumerate(output_list):
        match_label = re.search(r'LABEL:\s*(.*)', out.upper())
        match_answer_candidate = re.search(r'THE ANSWER CANDIDATE IS (\w+)', out.upper())
        match_likely = re.search(r'(?:\bLIKELY TO BE\s+)?\b(TRUE|FALSE)\b', out.upper())
        match_support = re.search(r'SUPPORTS THE ANSWER CANDIDATE', out.upper())

        if match_label:
            label_value = match_label.group(1)
            ones_zeros_list.append(1 if label_value in {'TRUE', 'VALID', 'CORRECT'} else 0)
            continue
        elif match_answer_candidate:
            answer_candidate_value = match_answer_candidate.group(1)
            ones_zeros_list.append(1 if answer_candidate_value in {'TRUE', 'VALID'} else 0)
            continue
        elif match_likely:
            likely = match_likely.group(1)
            ones_zeros_list.append(1 if likely in {'TRUE', 'VALID'} else 0)
            continue
        elif match_support:
            ones_zeros_list.append(1)
            continue
        else:  # if everything else fails, force it false or steal Simple_GPT4's Output
            if falsify_or_steal == 'falsify':
                ones_zeros_list.append(0)
            if falsify_or_steal == 'steal':
                gpt4_tf = extract_output_values_from_json_file(steal_source)
                tf_pattern = '(TRUE|FALSE)'
                gpt4_tf = re_scan(tf_pattern, gpt4_tf)
                ones_zeros_list.append(gpt4_tf[i])
            if falsify_or_steal == 'none':
                ones_zeros_list.append(None)

    return ones_zeros_list

# =============================================================================
# 
# =============================================================================

def look_at_nones(lst,output_text, prnt=False):
    """
    Look at None values in a list and print corresponding information from another list.

    Parameters:
    - lst (list): The input list containing values, including None.
    - output_text (list): The list containing additional information corresponding to the input list.

    Returns:
    None

    This function identifies the indices of None values in the input list (lst) and prints the
    corresponding information from the output_text list. It uses pandas DataFrame to find the
    indices of None values and prints the information along with a message for each identified index.

    Example:
    >>> lst = [1, None, 3, None, 5]
    >>> output_text = ['Info1', 'Info2', 'Info3', 'Info4', 'Info5']
    >>> look_at_nones(lst, output_text)

    Output:
    LOOK AT ME LOOK AT ME 1-1-1-1

    Info2

    LOOK AT ME LOOK AT ME 3-3-3-3

    Info4
    """
    none_output_list = []
    pd_df = pd.DataFrame(enumerate(lst))
    none_index_lsit = pd_df[pd_df[1].isna()][0].tolist()
    print(none_output_list)

    for i in none_index_lsit:
        none_output_list.append(output_text[i])
    if prnt:
        print(f'LOOK AT ME LOOK AT ME {i}-{i}-{i}-{i}\n\n{output_text[i]}\n\n\n\n')
    else:
        return none_output_list
    
# =============================================================================
#     
# =============================================================================
    
def list_to_csv(path, lst):
    """
    Write a list of values to a CSV file.

    Parameters:
    - path (str): The file path where the CSV file will be created or overwritten.
    - lst (list): The list of values to be written to the CSV file.

    Returns:
    None

    CSV Format:
    - The CSV file will have two columns: 'idx' and 'baseline'.
    - 'idx' represents the index of each element in the list.
    - 'baseline' contains the corresponding values from the input list.

    Example:
    ```python
    predictions = [0.6, 0.3, 0.8, 0.2]
    list_to_csv('output.csv', predictions)
    ```
    """
    with open(path, 'w') as out_file:
        w = csv.writer(out_file)
        w.writerow(['idx', 'baseline'])
        for i, pred in enumerate(lst):
            w.writerow([i, pred])
            
# =============================================================================
# 
# =============================================================================

def ensemble(list_of_lists, round_down=True):
    """
    Combine multiple prediction lists and produce an ensemble prediction.

    Parameters:
    - list_of_lists (list of lists): Each list represents a set of predictions with the same length.
    - round_down (bool, optional): If True (default), round predictions to 0 or 1 based on the threshold 0.5.
      If False, round predictions to 0 or 1 based on the threshold 0.5 (excluding 0.5 itself).

    Returns:
    - preds_list (list): Ensemble predictions, rounded to 0 or 1 based on the specified threshold.

    Note:
    - Assumes all lists in list_of_lists have the same length.

    Example:
    ```python
    gpt4Rag = [0.6, 0.3, 0.8]
    gpt4 = [0.4, 0.7, 0.2]
    ensemble_list = ensemble([gpt4Rag, gpt4])
    print(ensemble_list)
    ```
    """
    
    
    # instantiate a numpy array of zeros of length of the first list (lists should be same size)
    added_array = np.zeros(len(list_of_lists[0]))
    
    # Add the arrays together
    for lst in list_of_lists:
        lst = np.array(lst)
        added_array += lst
    
    # Find the average by dividing the sum by number of lists added
    added_array /= len(list_of_lists)
    
    # Predict output a list of 1s or 0s.
    preds_list = [] 
    for item in added_array:
        if round_down:
            if item <= 0.5:
                preds_list.append(0)
            else:
                preds_list.append(1)
        else:
            if item < 0.5:
                preds_list.append(0)
            else:
                preds_list.append(1)
    
    return preds_list

# =============================================================================
# 
# =============================================================================

def ensemble_w_cutoff(list_of_lists, cutoff=.5):
    """
    Combines predictions from multiple models using ensemble averaging
    and applies a cutoff threshold to generate final binary predictions.

    Parameters:
    - list_of_lists (list): A list containing lists of predictions from individual models.
    - cutoff (float, optional): Threshold value for binary classification (default is 0.5).

    Returns:
    - preds_list (list): Final binary predictions based on ensemble averaging and cutoff thresholding.

    Example:
    ```python
    # Usage example
    model1_preds = csv_to_list('model1_predictions.csv')
    model2_preds = csv_to_list('model2_predictions.csv')
    ensemble_preds = ensemble_w_cutoff([model1_preds, model2_preds], cutoff=0.6)
    ```
    """
    # instantiate a numpy array of zeros of length of the first list (lists should be same size)
    added_array = np.zeros(len(list_of_lists[0]))
    
    # Add the arrays together
    for lst in list_of_lists:
        lst = np.array(lst)
        added_array += lst
    
    # Find the average by dividing the sum by number of lists added
    added_array /= len(list_of_lists)
    
    # Predict output a list of 1s or 0s.
    preds_list = [] 
    for item in added_array:
        
        if item <= cutoff:
            preds_list.append(0)
        else:
            preds_list.append(1)
    
    return preds_list

def csv_to_list(csv_file_path):
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        return_list = []
        header=True
        for row in csv_reader:
            if header:
                header=False 
                next
            else:
                try:
                    return_list.append(int(row[1]))
                except:
                    return_list.append(row[1])
    return return_list

# =============================================================================
# 
# =============================================================================

def create_embeddings(checkpoint, dataset, batch_size=1):
    """
    Generate sentence embeddings using a pre-trained transformer model.

    Parameters:
    - checkpoint (str): The identifier of the pre-trained transformer model.
    - dataset (Dataset object): The DataFrame containing the data for which embeddings are to be generated.
    - batch_size (int, optional): The batch size for processing sentences in parallel. Default is 1.

    Returns:
    - np.ndarray: An array containing the sentence embeddings.

    The function utilizes the Hugging Face Transformers library to tokenize and encode sentences using a pre-trained
    transformer model. It processes the sentences in batches to efficiently use the available resources.

    Example:
    ```
    python
    checkpoint = 'nlpaueb/legal-bert-base-uncased'
    df_train = Dataset.from_dict(df_train)
    embeddings = create_embeddings(checkpoint, df_train, batch_size=4)
    ```
    """
    checkpoint = checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        truncation=True,
        padding='max_length',
        max_length=512
    )
    model = AutoModel.from_pretrained(checkpoint)
    sentence_vectors = []
    cqas = dataset['cqa']

    for batch_start in range(0, len(cqas), batch_size):
        batch_cqas = cqas[batch_start:batch_start + batch_size]
        input_ids_list = []

        for cqa in batch_cqas:
            input_ids = tokenizer.encode(
                cqa,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                padding='max_length'
            )
            input_ids_list.append(input_ids)

        input_ids_tensor = torch.tensor(input_ids_list)
        outputs = model(input_ids_tensor)
        last_hidden_states = outputs[0]
        sentence_vectors.extend(last_hidden_states.mean(dim=1).tolist())

    sentence_vectors = np.array(sentence_vectors)
    return sentence_vectors

# =============================================================================
# 
# =============================================================================

def ragtime(checkpoint, dataset, batch_size=1):
    """
    Generate sentence embeddings using a pre-trained transformer model.

    Parameters:
    - checkpoint (str): The identifier of the pre-trained transformer model.
    - dataset (Dataset object): The DataFrame containing the data for which embeddings are to be generated.
    - batch_size (int, optional): The batch size for processing sentences in parallel. Default is 1.

    Returns:
    - np.ndarray: An array containing the sentence embeddings.

    The function utilizes the Hugging Face Transformers library to tokenize and encode sentences using a pre-trained
    transformer model. It processes the sentences in batches to efficiently use the available resources.

    Example:
    ```
    python
    checkpoint = 'nlpaueb/legal-bert-base-uncased'
    df_train = Dataset.from_dict(df_train)
    embeddings = create_embeddings(checkpoint, df_train, batch_size=4)
    ```
    """
    checkpoint = checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        truncation=True,
        padding='max_length',
        max_length=512
    )
    model = AutoModel.from_pretrained(checkpoint)
    sentence_vectors = []
    cqas = dataset['cqa']

    for batch_start in range(0, len(cqas), batch_size):
        batch_cqas = cqas[batch_start:batch_start + batch_size]
        input_ids_list = []

        for cqa in batch_cqas:
            input_ids = tokenizer.encode(
                cqa,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                padding='max_length'
            )
            input_ids_list.append(input_ids)

        input_ids_tensor = torch.tensor(input_ids_list)
        outputs = model(input_ids_tensor)
        last_hidden_states = outputs[0]
        sentence_vectors.extend(last_hidden_states.mean(dim=1).tolist())
    # Set up dictionary to return
    labels = dataset['label']
    sentence_vectors = np.array(sentence_vectors)
    return_dict = {}
    return_dict['label'] = labels
    return_dict['sentence_vectors'] = sentence_vectors
    return return_dict

# =============================================================================
# 
# =============================================================================

def generate_prompt_head_list(test_dict, test_embeddings, true_train_dev, false_train_dev, n):
    """
    Generate a list of prompts by pairing a test observation with randomly selected
    true and false observations based on cosine similarity.

    Parameters:
    - test_dict (dict): A dictionary containing test data with keys 'cqa' and others.
    - test_embeddings (numpy.ndarray): Embeddings corresponding to test data.
    - true_train_dev (dict): A dictionary containing true training and development data
                            with keys 'cqaal', 'label', and 'sentence_vectors'.
    - false_train_dev (dict): A dictionary containing false training and development data
                             with keys 'cqaal', 'label', and 'sentence_vectors'.
    - n (int): Number of most similar true and false observations to consider.

    Returns:
    - prompt_head_list (list): A list of prompts generated by pairing true and false observations.

    Example:
    ```python
    n_value = 5
    prompt_list = generate_prompt_head_list(test_dict, test_embeddings, true_train_dev, false_train_dev, n_value)
    ```
    """
    
    prompt_head_list = []
    
    # iterate over each of our test obsver. 
    for test_text, test_emb in zip(test_dict['cqa'], test_embeddings):
        # reshape
        test_emb = test_emb.reshape(1, -1)
        
        # Find n most similar TRUEs
        true_cosine_similarities = cosine_similarity(test_emb, true_train_dev['sentence_vectors'])
        true_indexes = true_cosine_similarities[0].argsort()[::-1]
        true_indexes = true_indexes[0:n]
        true_indexes.tolist()
        random_true = random.choice(true_indexes)    
        
        # Find n most similar FALSEs
        false_cosine_similarities = cosine_similarity(test_emb, false_train_dev['sentence_vectors'])
        false_indexes = false_cosine_similarities[0].argsort()[::-1]
        false_indexes = false_indexes[0:n]
        false_indexes.tolist()
        random_false = random.choice(false_indexes)    

        prompt_head_list.append(
            f"{true_train_dev['cqaal'][random_true]}\n\n{false_train_dev['cqaal'][random_false]}\n\n"
        )

    return prompt_head_list