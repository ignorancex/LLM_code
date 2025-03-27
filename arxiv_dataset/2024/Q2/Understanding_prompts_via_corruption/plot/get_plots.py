# %%
# import    
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from get_component_len import compute_segment_lengths_anytask
from get_attention_matrix import get_attention_matrix_gptj

parser = argparse.ArgumentParser(description="Calculate average attention values from JSON data")

# Add command-line arguments for file_path and corruption_name
parser.add_argument("--file_path", type=str, help="Path to the JSON data file")
parser.add_argument("--corruption_name", type=str, help="Name of the corruption")
parser.add_argument("--start_from", type=int, default=0, help="sample number to start")

# Parse the command-line arguments
args = parser.parse_args()

file_path = args.file_path
corruption_name = args.corruption_name
start_from = args.start_from

# which model
# Replace with the actual model name and checkpoint you're using
# model_name = "EleutherAI/gpt-j-6b"
model_name = "facebook/opt-30b"

# get 5 samples where prediction is correct for all tasks so maximum ~50 samples, make a list of these datapoints

# Define the names of the tasks you're interested in.
# removed as it was acting weird "task284_imdb_classification"

tasks_of_interest = [
    #"task828_copa_commonsense_cause_effect",
    "task1346_glue_cola_grammatical_correctness_classification",
    "task1564_triviaqa_answer_generation",
    # "task116_com2sense_commonsense_reasoning",
    # "task1645_medical_question_pair_dataset_text_classification",
    # "task843_financial_phrasebank_classification",
    "task512_twitter_emotion_classification",
    # "task835_mathdataset_answer_generation",
    # "task379_agnews_topic_classification",
    # "task1344_glue_entailment_classification",
] 

# get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    return_tensors="pt"
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # for batch inferencecle

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    output_attentions=True,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    # device_map={"": "cpu"}
)

model_name_to_save = model_name.split("/")[-1]

# Create a dictionary to store samples for each task.
task_samples = {task: [] for task in tasks_of_interest}

if corruption_name not in ["labels_randomwords", "randomwords_inline_instr_in_0_demo"]:
    correct = "correct"
    # get correct pred samples
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)

                if data.get("Task") in tasks_of_interest:
                    target = data.get("Target")
                    prediction = data.get("Prediction")
                    
                    # Check if "Target" and "Prediction" match.
                    if target == prediction:
                        if len(tokenizer.tokenize(data["Instance"]["input"])) <= 405: # supported prompt length is 450
                            task_samples[data["Task"]].append(data)
                        
                        # If we have collected 10 samples for this task, move on to the next task.
                        if len(task_samples[data["Task"]]) >= 10:
                            tasks_of_interest.remove(data["Task"])
                            if not tasks_of_interest:
                                break
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")

else:
    correct = "incorrect"
    # get first 5 incorrect pred samples from each task
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                # Check if the "Task" is in the tasks of interest.
                if data.get("Task") in tasks_of_interest:
                    if len(tokenizer.tokenize(data["Instance"]["input"])) <= 405: # supported prompt length is 405
                        task_samples[data["Task"]].append(data)
                    
                    # If we have collected 5 samples for this task, move on to the next task.
                    if len(task_samples[data["Task"]]) >= 10:
                        tasks_of_interest.remove(data["Task"])
                        if not tasks_of_interest:
                            break
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")

# Flatten the dictionary to get a list of  samples.
result_samples = [sample for samples in task_samples.values() for sample in samples]
result_samples = result_samples[start_from:]

print(f"Total samples collected: {len(result_samples)}")

normalized_attention_list = []
# Iterate over each sample in result_samples
m = 0
with open(f"plot_results/{model_name_to_save}_{corruption_name}_{correct}_predicting.jsonl", 'w') as f:
    for sample in result_samples:
        print(m)
        # Get the "input" text from the sample
        input_text = sample["Instance"]["input"]

        # get attentions
        attention =  get_attention_matrix_gptj(sample, model, tokenizer) # [layers, sequence]
        attention = attention[:, -1, :].squeeze() # last query and it will look at all keys [layers, 1, s] -> [layers, s]
        print("Attention norms shape: ", attention.shape)
        print("Remember that we are looking at the attention norms of the last query token. The model is currently trying to predict the next token after the token")

        # get components length
        compname, comptoken_lens, compcolors = compute_segment_lengths_anytask(sample, model_name)

        # computer average attention per components
        comp_avg_attention = []
        for i, length in enumerate(comptoken_lens):
            start = 0
            end = start + length
            segment = attention[:, start:end]
            avg_attention = torch.mean(segment)
            comp_avg_attention.append((avg_attention.item(), compcolors[i]))

        comp_avg_attention = [item[0] for item in comp_avg_attention]  # Extract attention values
        normalized_attention = [round((item / sum(comp_avg_attention)), 4) for item in comp_avg_attention]  # Normalize
        normalized_attention_list.append(normalized_attention)

        del attention
        torch.cuda.empty_cache()

        # dummp the components and average attention in a jsonl file. file name = modelname_corruptionname_correct.jsonl
        sample["Components"] = compname
        sample["Component_lengths"] = comptoken_lens
        sample["Num_Components"] = len(compname)
        sample["attention"] = normalized_attention
        sample["avg_attn_per_comp"] = normalized_attention

        f.write(json.dumps(sample) + "\n")
        #json.dump(sample, f)

        m += 1
    
    # 1 plot for all samples and save the plot with same name modelname_corruptionname_correct.pdf
    avg_normalized_attention = [sum(pair)/len(normalized_attention_list) for pair in zip(*normalized_attention_list)]
    with open(f"plot_results/{model_name_to_save}_{corruption_name}_correct_predicting_avgattn.txt", "w") as file:
        # Iterate over the elements of the list and write them to the file
        for item in avg_normalized_attention:
            file.write(str(item) + '\n')
    # plt.hist(avg_normalized_attention)
    # plt.savefig(f"plot_results/{model_name_to_save}_{corruption_name}_correct_predicting.pdf")