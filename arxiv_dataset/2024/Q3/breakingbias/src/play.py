from transformers import logging

logging.set_verbosity_error()

import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llama_patch import unplace_flash_attn_with_attn

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LLAMA_SIZE = "13"
DATA_PATH = "../data/type2_dataset.csv"
SAVE_PATH = "../outputs/type2_dataset_responses.csv"
model_id = f"meta-llama/Llama-2-{LLAMA_SIZE}b-hf"
peft_model_id = f"iamshnoo/alpaca-2-{LLAMA_SIZE}b-english"
MODEL_CACHE_DIR = f"/scratch/craj/model_cache/llama-2/{LLAMA_SIZE}b"

use_flash_attention = True if torch.cuda.get_device_capability()[0] >= 8 else False

# explicitly disabling flash attention for 70b
if LLAMA_SIZE == "70":
    use_flash_attention = False

config = PeftConfig.from_pretrained(peft_model_id, cache_dir=MODEL_CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_4bit=True,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, cache_dir=MODEL_CACHE_DIR
)

if use_flash_attention:
    unplace_flash_attn_with_attn()

# Load the Lora model
model = PeftModel.from_pretrained(
    model,
    peft_model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)


def format_prompt(instruction, input=None, output=None):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}"
        ),
    }

    if input is None or input == "":
        return PROMPT_DICT["prompt_no_input"].format_map(
            {"instruction": instruction, "output": output}
        )
    else:
        return PROMPT_DICT["prompt_input"].format_map(
            {"instruction": instruction, "input": input, "output": output}
        )


# preferably don't change these. these parameters work good mostly.
generation_config = GenerationConfig(
    num_beams=1, temperature=1.0, top_p=0.95, top_k=50, max_new_tokens=50  # 0.2
)


def evaluate(instruction, input=None):
    prompt = format_prompt(instruction, input=input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    with torch.cuda.amp.autocast():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        out_sentence = output.split("### Response:")[1].strip()
        return out_sentence


df = pd.read_csv(DATA_PATH, header=0)
# print(df.head())

df = df[:10]

# instruction = "Choose between yes or no. Answer in one word."
# input = f"Question: Is Bill Gates rich?"#{question}" # assuming question is a df column that contains the question we want to ask
# out_sentence = evaluate(instruction, input)
# print()
# print("-"*80)
# print("Response:", out_sentence)
# print("-"*80)

# Create empty lists to store the responses
base_responses = []
positive_responses = []
negative_responses = []

# Instruction format is the same for all prompts, as in the given example
# other example instruction formats can be found at : https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
instruction = "Choose between likely and unlikely."

# Loop over each row in the dataframe
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Base prompt
    base_input = f"Question: {row['base_prompt']}"
    base_response = evaluate(instruction, base_input)
    base_responses.append(base_response)
    print("-" * 80)
    print(f"Base Question: {row['base_prompt']}")
    print()
    print(f"Base Response: {base_response}")
    print()

    # Positive prompt
    positive_input = f"Question: {row['positive_contact']}"
    positive_response = evaluate(instruction, positive_input)
    positive_responses.append(positive_response)
    print(f"Positive Question: {row['positive_contact']}")
    print()
    print(f"Positive Response: {positive_response}")
    print()

    # Negative prompt
    negative_input = f"Question: {row['negative_contact']}"
    negative_response = evaluate(instruction, negative_input)
    negative_responses.append(negative_response)
    print(f"Negative Question: {row['negative_contact']}")
    print()
    print(f"Negative Response: {negative_response}")
    print()
    print("-" * 80)

# Add responses to the dataframe
df["base_response"] = base_responses
df["positive_response"] = positive_responses
df["negative_response"] = negative_responses

# Optionally, save the dataframe to a CSV file
# df.to_csv(SAVE_PATH, index=False)
