from transformers import logging

logging.set_verbosity_error()

import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from cappr.huggingface.classify import predict, cache_model
import argparse

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llama", choices=["llama", "yi", "zephyr", "tulu", "noushermes", "vicuna"])
parser.add_argument("--llama_size", type=int, default=13, choices=[7, 13, 34])
parser.add_argument("--use_peft", type=int, default=0, choices=[1, 0])
parser.add_argument("--case", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument("--prompt_type", type=int, default="1", choices=[1, 2, 3])

args = parser.parse_args()

LLAMA_SIZE = str(args.llama_size)
TYPE = str(args.prompt_type)
MODEL_NAME = args.model_name
USE_PEFT = bool(args.use_peft)
CASE = args.case

print(f"Model:{MODEL_NAME}_{LLAMA_SIZE}B Peft:{USE_PEFT} Case:{CASE} Prompt_Type:{TYPE}")

if MODEL_NAME == "llama":
    model_id = f"meta-llama/Llama-2-{LLAMA_SIZE}b-chat-hf"
elif MODEL_NAME == "yi":
    model_id = f"01-ai/Yi-{LLAMA_SIZE}B-Chat"
    USE_PEFT = False
    CASE = 0
elif MODEL_NAME == "zephyr":
    model_id = f"HuggingFaceH4/zephyr-{LLAMA_SIZE}b-beta"
    USE_PEFT = False
    CASE = 0
elif MODEL_NAME == "tulu":
    model_id = f"allenai/tulu-2-dpo-{LLAMA_SIZE}b"
    USE_PEFT = False
    CASE = 0
elif MODEL_NAME == "noushermes":
    model_id = f"NousResearch/Nous-Hermes-Llama2-{LLAMA_SIZE}b"
    USE_PEFT = False
    CASE = 0
elif MODEL_NAME == "vicuna":
    model_id = f"lmsys/vicuna-{LLAMA_SIZE}b-v1.5"
    USE_PEFT = False
    CASE = 0

if MODEL_NAME == "llama":
    MODEL_CACHE_DIR = f"/scratch/craj/model_cache/llama-2-chat/{LLAMA_SIZE}b"
elif MODEL_NAME == "yi":
    MODEL_CACHE_DIR = f"/scratch/craj/model_cache/yi-chat/{LLAMA_SIZE}B"
elif MODEL_NAME == "zephyr":
    MODEL_CACHE_DIR = f"/scratch/craj/model_cache/zephyr/{LLAMA_SIZE}B"
elif MODEL_NAME == "tulu":
    MODEL_CACHE_DIR = f"/scratch/craj/model_cache/tulu/{LLAMA_SIZE}B"
elif MODEL_NAME == "noushermes":
    MODEL_CACHE_DIR = f"/scratch/craj/model_cache/noushermes/{LLAMA_SIZE}B"
elif MODEL_NAME == "vicuna":
    MODEL_CACHE_DIR = f"/scratch/craj/model_cache/vicuna/{LLAMA_SIZE}B"

if USE_PEFT:
    if CASE == 1:
        peft_model_id = "chahatraj/contact-bpn-alldata-13b-en"
        DATA_PATH = "../data/finetune/finetune_alldata/test.csv"
        SAVE_PATH = "../finetune_eval/after_finetune/pbn_alldata_dataset_responses_13B.csv"

    elif CASE == 2:
        peft_model_id = "chahatraj/contact-bpn-newdata-13b-en"
        DATA_PATH = "../data/finetune/finetune_newdata/test.csv"
        SAVE_PATH = "../finetune_eval/after_finetune/pbn_newdata_dataset_responses_13B.csv"

    elif CASE == 3:
        peft_model_id = "chahatraj/contact-bpn-type1-13b-en"
        DATA_PATH = "../data/finetune/finetune_type1/test.csv"
        SAVE_PATH = "../finetune_eval/after_finetune/pbn_type1_dataset_responses_13B.csv"

    elif CASE == 4:
        peft_model_id = "chahatraj/contact-bpn-type2-13b-en"
        DATA_PATH = "../data/finetune/finetune_type2/test.csv"
        SAVE_PATH = "../finetune_eval/after_finetune/pbn_type2_dataset_responses_13B.csv"

    elif CASE == 5:
        peft_model_id = "chahatraj/contact-bpn-type3-13b-en"
        DATA_PATH = "../data/finetune/finetune_type3/test.csv"
        SAVE_PATH = "../finetune_eval/after_finetune/pbn_type3_dataset_responses_13B.csv"

    elif CASE == 6:
        peft_model_id = "chahatraj/contact-bpn-2scenarios-13b-en"
        DATA_PATH = "../data/finetune/finetune_2scenarios/test.csv"
        SAVE_PATH = "../finetune_eval/after_finetune/pbn_2scenarios_dataset_responses_13B.csv"

    elif CASE == 7:
        peft_model_id = "chahatraj/contact-bpn-3keyprinciples-13b-en"
        DATA_PATH = "../data/finetune/finetune_3keyprinciples/test.csv"
        SAVE_PATH = "../finetune_eval/after_finetune/pbn_3keyprinciples_dataset_responses_13B.csv"

    elif CASE == 8:
        peft_model_id = "chahatraj/contact-bpn-6biasdimensions-13b-en"
        DATA_PATH = "../data/finetune/finetune_6biasdimensions/test.csv"
        SAVE_PATH = "../finetune_eval/after_finetune/pbn_6biasdimensions_dataset_responses_13B.csv"
else:
    if CASE == 1:
        DATA_PATH = "../data/finetune/finetune_alldata/test.csv"
        SAVE_PATH = "../finetune_eval/before_finetune/pbn_alldata_dataset_responses_13B.csv"

    elif CASE == 2:
        DATA_PATH = "../data/finetune/finetune_newdata/test.csv"
        SAVE_PATH = "../finetune_eval/before_finetune/pbn_newdata_dataset_responses_13B.csv"

    elif CASE == 3:
        DATA_PATH = "../data/finetune/finetune_type1/test.csv"
        SAVE_PATH = "../finetune_eval/before_finetune/pbn_type1_dataset_responses_13B.csv"

    elif CASE == 4:
        DATA_PATH = "../data/finetune/finetune_type2/test.csv"
        SAVE_PATH = "../finetune_eval/before_finetune/pbn_type2_dataset_responses_13B.csv"

    elif CASE == 5:
        DATA_PATH = "../data/finetune/finetune_type3/test.csv"
        SAVE_PATH = "../finetune_eval/before_finetune/pbn_type3_dataset_responses_13B.csv"

    elif CASE == 6:
        DATA_PATH = "../data/finetune/finetune_2scenarios/test.csv"
        SAVE_PATH = "../finetune_eval/before_finetune/pbn_2scenarios_dataset_responses_13B.csv"

    elif CASE == 7:
        DATA_PATH = "../data/finetune/finetune_3keyprinciples/test.csv"
        SAVE_PATH = "../finetune_eval/before_finetune/pbn_3keyprinciples_dataset_responses_13B.csv"

    elif CASE == 8:
        DATA_PATH = "../data/finetune/finetune_6biasdimensions/test.csv"
        SAVE_PATH = "../finetune_eval/before_finetune/pbn_6biasdimensions_dataset_responses_13B.csv"

    elif CASE == 0:
        DATA_PATH = f"../data/type{TYPE}_dataset.csv"
        SAVE_PATH = f"../outputs/{MODEL_NAME}/ft_type_{TYPE}_dataset_responses_{LLAMA_SIZE}B.csv"


if CASE == 0:
    if int(TYPE) == 1:
        instruction = "Choose between Yes and No."
        options = ["Yes", "No"]
    elif int(TYPE) == 2:
        instruction = "Choose between Likely and Unlikely."
        options = ["Likely", "Unlikely"]
    elif int(TYPE) == 3:
        instruction = "Choose between Mostly and Rarely."
        options = ["Mostly", "Rarely"]


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    return_dict=True,
    load_in_4bit=True,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)

if USE_PEFT:
    model = PeftModel.from_pretrained(
        model,
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR,
    )

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
PREFIX = f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n Follow the instruction:"

def format_prompt(instruction, input):
    # DEFAULT_SYSTEM_PROMPT = """\
    #     You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
    #     """
    # PROMPT_DICT = {
    #     "prompt_input": (
    #         "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n Follow the instruction: {instruction}. Answer in one word. \n Question: {input} \n Answer:\n[/INST]\n"
    #     ),
    # }

    # return PROMPT_DICT["prompt_input"].format_map(
    #     {
    #         "instruction": instruction,
    #         "input": input,
    #         "system_prompt": DEFAULT_SYSTEM_PROMPT,
    #     }
    # )

    PROMPT_DICT = {
        "prompt_input": (
            "{prefix} {instruction}. Answer in one word. \n Question: {input} \n Answer:\n[/INST]\n"
        ),
    }

    return PROMPT_DICT["prompt_input"].format_map(
        {
            "instruction": instruction,
            "input": input,
            "prefix": PREFIX,
        }
    )

print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH, header=0)
# df = df[:100]


base_responses = []
positive_responses = []
negative_responses = []

model_and_tokenizer = (model, tokenizer)
cached_model_and_tokenizer = cache_model(
    model_and_tokenizer, PREFIX
)

for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    if CASE != 0:
        instruction = row["Instruction"]
        if "Choose between Yes and No." in instruction:
            options = ["Yes", "No"]
            instruction = "Choose between Yes and No."
        elif "Choose between Likely and Unlikely." in instruction:
            options = ["Likely", "Unlikely"]
            instruction = "Choose between Likely and Unlikely."
        elif "Choose between Mostly and Rarely." in instruction:
            options = ["Mostly", "Rarely"]
            instruction = "Choose between Mostly and Rarely."

    # Base prompt
    base_input = f"{row['base_prompt']}"
    base_prompt = format_prompt(instruction, input=base_input)

    # Positive prompt
    positive_input = f"{row['positive_contact']}"
    positive_prompt = format_prompt(instruction, input=positive_input)

    # Negative prompt
    negative_input = f"{row['negative_contact']}"
    negative_prompt = format_prompt(instruction, input=negative_input)
    prompts = [base_prompt, positive_prompt, negative_prompt]
    base_response, positive_response, negative_response = predict(
        prompts,
        completions=options,
        model_and_tokenizer=cached_model_and_tokenizer,
        end_of_prompt="",
        batch_size=3
    )
    base_responses.append(base_response)
    positive_responses.append(positive_response)
    negative_responses.append(negative_response)


df["base_response"] = base_responses
df["positive_response"] = positive_responses
df["negative_response"] = negative_responses

print(df.head())
df.to_csv(SAVE_PATH, index=False)
