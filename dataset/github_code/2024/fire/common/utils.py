import copy, io, json, os, random, re, string, types, termcolor
from typing import Any
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


################################################################################
#                              Cost Calculation                                #
################################################################################

def calculate_cost_o1(model_type, input_tokens, completion_tokens):
    # Normalize model type to lowercase to handle both upper and lower case inputs
    model_type = model_type.lower()
    
    # Define pricing based on model type
    if model_type == 'o1-preview' or model_type == 'o1-preview-2024-09-12':
        input_token_price_per_million = 15.00
        output_token_price_per_million = 60.00
    elif model_type == 'o1-mini' or model_type == 'o1-mini-2024-09-12':
        input_token_price_per_million = 3.00
        output_token_price_per_million = 12.00
    else:
        raise ValueError("Invalid model type. Choose from 'o1-preview', 'o1-preview-2024-09-12', 'o1-mini', or 'o1-mini-2024-09-12'.")
    
    # Calculate the cost for the given model
    input_cost = (input_tokens / 1_000_000) * input_token_price_per_million
    completion_cost = (completion_tokens / 1_000_000) * output_token_price_per_million
    
    total_cost = input_cost + completion_cost
    
    return {
        "Model Type": model_type,
        "Input Cost": input_cost,
        "Completion Cost": completion_cost,
        "Total Cost": total_cost
    }

def calculate_cost_claude(model_type, input_tokens, completion_tokens):
    # Normalize model type to lowercase to handle both upper and lower case inputs
    model_type = model_type.lower()
    
    # Define pricing based on model type
    if model_type == 'sonnet':
        input_token_price_per_million = 3.00
        output_token_price_per_million = 15.00
    elif model_type == 'opus':
        input_token_price_per_million = 15.00
        output_token_price_per_million = 75.00
    elif model_type == 'haiku':
        input_token_price_per_million = 0.25
        output_token_price_per_million = 1.25
    else:
        raise ValueError("Invalid model type. Choose from 'Sonnet', 'Opus', or 'Haiku'.")
    
    # Calculate the cost for the given model
    input_cost = (input_tokens / 1_000_000) * input_token_price_per_million
    completion_cost = (completion_tokens / 1_000_000) * output_token_price_per_million
    
    total_cost = input_cost + completion_cost
    
    return {
        "Model Type": model_type.capitalize(),
        "Input Cost": input_cost,
        "Completion Cost": completion_cost,
        "Total Cost": total_cost
    }

################################################################################
#                              Metrics Calculation                             #
################################################################################


def calculate_metrics(tp, tn, fp, fn):
    # Positive class metrics
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)) if (precision_pos + recall_pos) > 0 else 0

    # Negative class metrics
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_neg = (2 * precision_neg * recall_neg / (precision_neg + recall_neg)) if (precision_neg + recall_neg) > 0 else 0
    
    print('For positive class & For negative class')
    print(f'{round(precision_pos, 2)} & {round(recall_pos, 2)} & {round(f1_pos, 2)}', end=' & ')
    print(f'{round(precision_neg, 2)} & {round(recall_neg, 2)} & {round(f1_neg, 2)}')

def evaluate_file(file_name):
    tp, tn, fp, fn = 0, 0, 0, 0
    search_nums = []

    with open(file_name) as file:
        print(f'The evaluated file is: {file_name}')
        for line in file.readlines():
            data = json.loads(line)
            label = data['label']
            generated = data['result']
            searches = data['searches']['google_searches']
            predicted = generated['answer'].lower()
            if label == 'true' and predicted == 'true':
                tp += 1
            elif label == 'false' and predicted == 'false':
                tn += 1
            elif label == 'false' and predicted == 'true':
                fp += 1
            elif label == 'true' and predicted == 'false':
                fn += 1

    calculate_metrics(tp, tn, fp, fn)

def count_searches_and_plot(filenames, models, output_file='search_numbers.pdf'):
    search_nums = [[] for _ in range(len(filenames))]

    # Count the number of searches for each file
    for i, file_name in enumerate(filenames):
        with open(file_name) as file:
            for line in file.readlines():
                data = json.loads(line)
                searches = data['searches']['google_searches']
                num_searches = len(searches)
                search_nums[i].append(num_searches)

        # print(f'Search counts for {models[i]}: {search_nums[i]}')
        print(f'Total searches for {models[i]}: {sum(search_nums[i])}')

    # Count frequencies using Counter
    freq_list = [Counter(search_nums[i]) for i in range(len(filenames))]

    # Get the unique numbers from all lists
    all_numbers = sorted(set().union(*[set(freq.keys()) for freq in freq_list]))

    # Create lists of frequencies for each number in all lists
    freq_matrix = [[freq.get(num, 0) for num in all_numbers] for freq in freq_list]

    # Define the positions of the bars on the x-axis
    bar_width = 0.8 / len(models)  # Adjust bar width based on the number of models
    indices = np.arange(len(all_numbers))

    # Create the bars for each model
    for i, (freqs, model) in enumerate(zip(freq_matrix, models)):
        bars = plt.bar(indices + i * bar_width, freqs, width=bar_width, label=model)

        # Add frequency labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    # Add labels, title, and legend
    plt.xlabel('Number of Google Searches')
    plt.ylabel('Frequency')
    plt.xticks(indices + (bar_width * len(models)) / 2, all_numbers)
    plt.legend()

    # Save the plot to a file
    plt.savefig(output_file)
    plt.show()




################################################################################
#                             STRING MANIPULATION                              #
################################################################################
def join_segments(*args: str | list[str], separator: str = '\n\n\n') -> str:
  """Joins an unspecified number of strings using the separator."""
  all_segments = []

  for arg in args:
    if isinstance(arg, list):
      all_segments.extend(arg)
    else:
      all_segments.append(strip_string(str(arg)))

  return strip_string(separator.join(all_segments))


def strip_string(s: str) -> str:
  """Strips a string of newlines and spaces."""
  return s.strip(' \n')

def extract_json_from_output(model_output) -> dict | None:
    """Extracts JSON from model output string.

    Args:
        model_output (str): The output string from the model that may contain JSON.

    Returns:
        dict or None: Parsed JSON as a dictionary if found; None if not found or parsing fails.
    """
    # Step 1: Use regex to find the JSON block in the output
    json_match = re.search(r'{.*}', model_output, re.DOTALL)

    # Step 2: If JSON is found, attempt to parse it
    if json_match:
        try:
            # Extract JSON string
            json_str = json_match.group(0)
            # Parse the JSON string
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            # print("Error parsing JSON")
            return None
    else:
        print("No JSON found in the output")
        return None

