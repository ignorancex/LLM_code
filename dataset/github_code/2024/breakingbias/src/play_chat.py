from transformers import logging

logging.set_verbosity_error()

import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LLAMA_SIZE = "13"
DATA_PATH = "../data/type7_dataset.csv"
SAVE_PATH = f"../outputs/ft_type7_dataset_responses_{LLAMA_SIZE}B.csv"
model_id = f"meta-llama/Llama-2-{LLAMA_SIZE}b-chat-hf"
MODEL_CACHE_DIR = f"/scratch/craj/model_cache/llama-2-chat/{LLAMA_SIZE}b"

#Change this
instruction = "Choose between Mostly and Rarely."

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    return_dict=True,
    load_in_4bit=True,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)

# Load the LoRa model
peft_model_id = "chahatraj/contact-base-small-13b-english"
model = PeftModel.from_pretrained(
    model,
    peft_model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)


def format_prompt(instruction, input=None):
    DEFAULT_SYSTEM_PROMPT = """\
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
        """
    PROMPT_DICT = {
        "prompt_input": (
            "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n Follow the instruction {instruction}. Answer in one word. \n Question: {input} \n[/INST]\n"
        ),
        # NOTE for chahat: this next line is wrong. and it's not used anywhere. so its fine.
        "prompt_no_input": (
            "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n Answer in one word. \n Question: {input} \n[/INST]\n"
        ),
    }

    if input is None or input == "":
        return PROMPT_DICT["prompt_no_input"].format_map(
            {"instruction": instruction, "system_prompt": DEFAULT_SYSTEM_PROMPT}
        )
    else:
        return PROMPT_DICT["prompt_input"].format_map(
            {
                "instruction": instruction,
                "input": input,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
            }
        )


# preferably don't change these. these parameters work good mostly.
generation_config = GenerationConfig(
    num_beams=1, temperature=1.0, top_p=0.95, top_k=50, max_new_tokens=5  # 0.2
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
        out_sentence = output.split("[/INST]\n")[1].strip()
        # remove the </s> at the end of the output_sentence
        out_sentence = out_sentence.replace("</s>", "")
        return out_sentence


df = pd.read_csv(DATA_PATH, header=0)

# df = df[:100]

# Create empty lists to store the responses
base_responses = []
positive_responses = []
negative_responses = []

# Loop over each row in the dataframe
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Base prompt
    base_input = f"Question: {row['base_prompt']}"
    base_response = evaluate(instruction, base_input)
    base_responses.append(base_response)
    # print("-"*80)
    # print(f"Base Question: {row['base_prompt']}")
    # print()
    # print(f"Base Response: {base_response}")
    # print()

    # Positive prompt
    positive_input = f"Question: {row['positive_contact']}"
    positive_response = evaluate(instruction, positive_input)
    positive_responses.append(positive_response)
    # print(f"Positive Question: {row['positive_contact']}")
    # print()
    # print(f"Positive Response: {positive_response}")
    # print()

    # Negative prompt
    negative_input = f"Question: {row['negative_contact']}"
    negative_response = evaluate(instruction, negative_input)
    negative_responses.append(negative_response)
    # print(f"Negative Question: {row['negative_contact']}")
    # print()
    # print(f"Negative Response: {negative_response}")
    # print()
    # print("-"*80)

# Add responses to the dataframe
df["base_response"] = base_responses
df["positive_response"] = positive_responses
df["negative_response"] = negative_responses

# Optionally, save the dataframe to a CSV file
df.to_csv(SAVE_PATH, index=False)
