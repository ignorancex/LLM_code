'''
Script to visualize the averaged weights for one given checkpoint of router applied on differnt QA datasets to show that it effectively adopts different preference of granularities on different QA datasets.

Zijie May 15 2024
'''

# LIBRARIES
import sys
root_path = "***"
sys.path.insert(0, f"{root_path}/MoG/src")
# from moe import Router

import torch
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import numpy as np

# medmcqa
def load_medmcqa_training_data(medmcqa_path, retrieval_result_path):
    data = []
    labels = []
    medmcqa_path = os.path.join(medmcqa_path, "data", "dev.json")
    # medmcqa is stored in fact as jsonl format
    for line in open(medmcqa_path):
        line = json.loads(line)
        data.append(line["question"])

        exp = line["exp"] if line["exp"] is not None else ""
        if line["cop"] == 1:
            labels.append(exp + line["opa"])
        elif line["cop"] == 2:
            labels.append(exp + line["opb"])
        elif line["cop"] == 3:
            labels.append(exp + line["opc"])
        elif line["cop"] == 4:
            labels.append(exp + line["opd"])

    retrieval_results = []
    scores = []
    with open(
        os.path.join(retrieval_result_path, "medmcqa_retrieval_result.jsonl"), "r"
    ) as file:
        for line in file:
            json_obj = json.loads(line)
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])
    print("medmcqa training data loaded. ")
    return data, labels, retrieval_results, scores

# bioasq
def load_bioasq_training_data(bioasq_path, retrieval_result_path):
    data = []
    labels = []
    bioasq_path = os.path.join(bioasq_path, "Task7BGoldenEnriched")
    # get all the json file names under bioasq_path
    file_names = [f for f in os.listdir(bioasq_path) if f.endswith(".json")]
    for file_name in file_names:
        file_path = os.path.join(bioasq_path, file_name)
        with open(file_path, "r") as json_file:
            json_data = json.load(json_file)
            json_data_questions = json_data["questions"]
            for question in json_data_questions:
                data.append(question["body"])
                context_label = []
                # context_label should be a list of the strings under snippets.text
                for snippet in question["snippets"]:
                    context_label.append(snippet["text"])
                context_label = " ".join(context_label)
                labels.append(context_label)

    retrieval_results = []
    scores = []
    with open(
        os.path.join(retrieval_result_path, "bioasq_retrieval_result.jsonl"), "r"
    ) as file:
        for line in file:
            json_obj = json.loads(line)
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])
    print("bioasq training data loaded. ")

    return data, labels, retrieval_results, scores

# pubmedqa
def load_pubmedqa_training_data(pubmedqa_path, retrieval_result_path):
    data = []
    labels = []
    pubmedqa_path = os.path.join(pubmedqa_path, "data", "test_set.json")

    with open(pubmedqa_path, "r") as json_file:
        json_data = json.load(json_file)
        for question_id, question_data in json_data.items():
            data.append(question_data["QUESTION"])
            labels.append(question_data["CONTEXTS"][0])

    retrieval_results = []
    scores = []
    with open(
        os.path.join(retrieval_result_path, "pubmedqa_retrieval_result.jsonl"), "r"
    ) as file:
        for line in file:
            json_obj = json.loads(line)
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])

    print("pubmedqa training data loaded. ")
    return data, labels, retrieval_results, scores

# medqa
def load_medqa_training_data(medqa_path, retrieval_result_path):
    data = []
    labels = []
    medqa_path = os.path.join(
        medqa_path,
        "data_clean",
        "questions",
        "US",
        "4_options",
        "phrases_no_exclude_test.jsonl",
    )
    # medqa is stored in fact as jsonl format
    for line in open(medqa_path):
        line = json.loads(line)
        data.append(line["question"])
        labels.append("Q:" + line["question"] + "; A:" + line["answer"])

    retrieval_results = []
    scores = []
    with open(
        os.path.join(retrieval_result_path, "medqa_retrieval_result.jsonl"), "r"
    ) as file:
        for line in file:
            json_obj = json.loads(line)
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])

    print("medqa training data loaded. ")
    return data, labels, retrieval_results, scores

# mmlu
def load_mmlu_training_data(mmlu_path, retrieval_result_path):
    data = []
    labels = []
    mmlu_path = os.path.join(mmlu_path, "data", "dev.json")
    # Initialize lists to hold questions and answers
    data = []
    labels = []

    # Open and read the entire JSON file
    with open(mmlu_path, 'r') as file:
        try:
            json_data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            json_data = {}

    # Iterate over each item in the JSON data
    for item_key, item_value in json_data.items():
        try:
            data.append(item_value["question"])
            labels.append(item_value["answer"])
        except KeyError as e:
            print(f"Missing key {e} in item: {item_key}")

    retrieval_results = []
    scores = []
    with open(
        os.path.join(retrieval_result_path, "mmlu_retrieval_result.jsonl"), "r"
    ) as file:
        for line in file:
            json_obj = json.loads(line)
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])

    print("mmlu training data loaded. ")
    return data, labels, retrieval_results, scores
    

def visualize_weights(checkpoint_path, dataset_paths, retrieval_result_path):
    print("Visualization starts")
    # # Init model
    # model = Router(
    #     query="sample query",
    #     output_dim=5,
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # )
    # # Load checkpoint
    # model.load_state_dict(torch.load(checkpoint_path))
    # model.eval()
    
    # # Store the avg weights to plot the aggregated plot
    # avg_weight_list = {}

    # for i1, dataset_path in enumerate(dataset_paths, start=1):
    #     # Get dataset_name
    #     dataset_name = dataset_path.split("/")[-1]
        
    #     # Load dataset
    #     if dataset_name == 'mmlu':
    #         dataset, _, _, _ = load_mmlu_training_data(dataset_path, retrieval_result_path)
    #     elif dataset_name == 'medmcqa':
    #         dataset, _, _, _ = load_medmcqa_training_data(dataset_path, retrieval_result_path)
    #     elif dataset_name == 'medqa':
    #         dataset, _, _, _ = load_medqa_training_data(dataset_path, retrieval_result_path)
    #     elif dataset_name == 'bioasq':
    #         dataset, _, _, _ = load_bioasq_training_data(dataset_path, retrieval_result_path)
    #     elif dataset_name == 'pubmedqa':
    #         dataset, _, _, _ = load_pubmedqa_training_data(dataset_path, retrieval_result_path)
            
    #     print(f"\n[Done] dataset {dataset_name} loaded.")

    #     # Initialize list to store averaged weights
    #     averaged_weights = [0] * 5

    #     # Iterate through dataset
    #     for i in tqdm(range(len(dataset)), leave=False, desc='samples'):
    #         question = dataset[i]
    #         # Run inference on question
    #         output = model.run(question)  # Assuming model accepts questions directly

    #         # Accumulate weights
    #         for j in range(5):
    #             averaged_weights[j] += output[j].item()

    #     # Calculate average
    #     total_items = len(dataset)
    #     averaged_weights = [weight / total_items for weight in averaged_weights]
    #     print(f"\n[Done] total sample number: {total_items}, averaged_weights: {averaged_weights}")
    #     # Store the results
    #     avg_weight_list[dataset_name] = averaged_weights
        
    avg_weight_list = {
        'mmlu':[0.09757632307346915, 0.14459752268291892, 0.21715837503863514, 0.25880381365137817, 0.28186396605509223], 
        'medmcqa': [0.12622399205387805, 0.16259407277756782, 0.23210075774451136, 0.23365332095254757, 0.24542785792157262],
        'bioasq': [0.12303687038482167, 0.1643851921198657, 0.23036468829016668, 0.23354870268143713, 0.24866454602137675],
        'pubmedqa': [0.31357217134477094, 0.2558766730745771, 0.19032370683066585, 0.13531029886600224, 0.10491715491079713],
        'medqa': [0.03872429975862034, 0.12054489952563663, 0.2383591352029126, 0.27483769482091647, 0.3275339710324065]
    }
    
    # Plot the aggregated plot
    # Extract keys and values from the dictionary
    keys = list(avg_weight_list.keys())
    values = list(avg_weight_list.values())

    # Number of datasets and number of bars in each group
    num_datasets = len(keys)
    num_bars = len(values[0])

    # Generate positions for the groups of bars with spacing between groups
    bar_width = 0.2
    group_width = num_datasets * bar_width
    group_spacing = 0.5
    x = np.arange(num_bars) * (group_width + group_spacing)

    # Define colors for the bars
    # colors = ['#9cc3e6', '#f4b184', '#e2f0d9', '#fbd967', '#bfbfbf']
    colors = ['#8FABDB', '#F4B184', '#A8D08F', '#FFE699', '#BFBFBF']

    # Plotting each dataset with adjusted figure size for single-column A4 article
    fig, ax = plt.subplots(figsize=(10, 3))

    for i, (dataset, value) in enumerate(avg_weight_list.items()):
        ax.bar(x + i * bar_width, value, width=bar_width, label=dataset, color=colors[i])


    # Adding labels and legend
    ax.set_xlabel('Granularity level')
    ax.set_ylabel('Avg. weights')
    ax.set_xticks(x + bar_width * (num_datasets - 1) / 2)
    ax.set_xticklabels([f'Level {i+1}' for i in range(num_bars)])
    ax.legend()
    
    plt.tight_layout()
    
    save_path = f'granularity_distribution_plot.png'
    plt.savefig(save_path)
    plt.close()
    print(f"\n[Done] Aggregated plot saved in {save_path}")

# PARAMETERS

medmcqa_path = f"{root_path}/MoG/qa_datasets_rawdata/medmcqa"
bioasq_path = f"{root_path}/MoG/qa_datasets_rawdata/bioasq"
pubmedqa_path = f"{root_path}/MoG/qa_datasets_rawdata/pubmedqa"
medqa_path = f"{root_path}/MoG/qa_datasets_rawdata/medqa"
mmlu_path = f"{root_path}/MoG/qa_datasets_rawdata/mmlu"
retrieval_result_path = f"{root_path}/MoG/retrieval_results_textbooks/rag_3_corpus_mog_BM25/"
checkpoint_path = f'{root_path}/MoG/router_checkpoint/2001_20240519/Epoch_999_Loss_0.2049.pt'

dataset_paths = [mmlu_path, medmcqa_path, bioasq_path, pubmedqa_path, medqa_path]

visualize_weights(checkpoint_path, dataset_paths, retrieval_result_path)