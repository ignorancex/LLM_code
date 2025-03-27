from transformers import logging

logging.set_verbosity_error()

import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
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
parser.add_argument("--model_name", type=str, default="llama", choices=["llama"])
parser.add_argument("--llama_size", type=int, default=13, choices=[13])
parser.add_argument("--case", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])

args = parser.parse_args()

MODEL_NAME = str(args.model_name)
LLAMA_SIZE = str(args.llama_size)
CASE = int(args.case)

print(f"Model:{MODEL_NAME}_{LLAMA_SIZE}B Case:{CASE}")

model_id = f"meta-llama/Llama-2-{LLAMA_SIZE}b-chat-hf"
MODEL_CACHE_DIR = f"/scratch/craj/model_cache/llama-2-chat/{LLAMA_SIZE}b"
DATA_CACHE_DIR = f"/scratch/craj/datasets_cache/bbq"

if CASE == 1:
    peft_model_id = "chahatraj/contact-bpn-alldata-13b-en"
    SAVE_PATH = "../finetune_eval/after_finetune/bbq/pbn_alldata_bbq_responses_13B.csv"

elif CASE == 2:
    peft_model_id = "chahatraj/contact-bpn-newdata-13b-en"
    SAVE_PATH = "../finetune_eval/after_finetune/bbq/pbn_newdata_bbq_responses_13B.csv"

elif CASE == 3:
    peft_model_id = "chahatraj/contact-bpn-type1-13b-en"
    SAVE_PATH = "../finetune_eval/after_finetune/bbq/pbn_type1_bbq_responses_13B.csv"

elif CASE == 4:
    peft_model_id = "chahatraj/contact-bpn-type2-13b-en"
    SAVE_PATH = "../finetune_eval/after_finetune/bbq/pbn_type2_bbq_responses_13B.csv"

elif CASE == 5:
    peft_model_id = "chahatraj/contact-bpn-type3-13b-en"
    SAVE_PATH = "../finetune_eval/after_finetune/bbq/pbn_type3_bbq_responses_13B.csv"

elif CASE == 6:
    peft_model_id = "chahatraj/contact-bpn-2scenarios-13b-en"
    SAVE_PATH = "../finetune_eval/after_finetune/bbq/pbn_2scenarios_bbq_responses_13B.csv"

elif CASE == 7:
    peft_model_id = "chahatraj/contact-bpn-3keyprinciples-13b-en"
    SAVE_PATH = "../finetune_eval/after_finetune/bbq/pbn_3keyprinciples_bbq_responses_13B.csv"

elif CASE == 8:
    peft_model_id = "chahatraj/contact-bpn-6biasdimensions-13b-en"
    SAVE_PATH = "../finetune_eval/after_finetune/bbq/pbn_6biasdimensions_bbq_responses_13B.csv"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    return_dict=True,
    load_in_4bit=True,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)

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

model_and_tokenizer = (model, tokenizer)
cached_model_and_tokenizer = cache_model(
    model_and_tokenizer, PREFIX
)


def format_prompt(instruction, input):    
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


configs = get_dataset_config_names("lighteval/bbq_helm")

for config in configs:
    print(f"BBQ split: {config}")
    dataset = load_dataset("lighteval/bbq_helm", config, cache_dir=DATA_CACHE_DIR)
    # print(dataset["test"])
    df = pd.DataFrame(dataset["test"])
    # drop the colun "references"
    df.drop(columns=["references"], inplace=True)
    print(df.head())
    print()