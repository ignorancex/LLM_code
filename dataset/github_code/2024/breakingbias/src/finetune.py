# CURRENTLY USING A SMALL PROPORTION OF DATA FOR FINETUNING

# import pandas as pd
# import numpy as np

# path = "../data/finetune/finetunedata.csv"
# df = pd.read_csv(path)

# # first, shuffle the data
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # keep the first 10% of data as "train.csv", the next "5%" as valid.csv, the rest as "test.csv"
# train = df[:int(len(df)*0.1)]
# valid = df[int(len(df)*0.1):int(len(df)*0.15)]
# test = df[int(len(df)*0.15):]

# train.to_csv("../data/finetune/train.csv", index=False)
# valid.to_csv("../data/finetune/valid.csv", index=False)
# test.to_csv("../data/finetune/test.csv", index=False)

from transformers import logging

logging.set_verbosity_error()

import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig, 
    TrainingArguments
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import wandb
from llama_patch import forward, replace_attn_with_flash_attn

use_flash_attention = True if torch.cuda.get_device_capability()[0] >= 8 else False
replace_attn_with_flash_attn()

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

wandb.init(project=f"contact-theory-chat", name="13b-english")

model_id = "meta-llama/Llama-2-13b-chat-hf"
MODEL_CACHE_DIR = "/scratch/craj/model_cache/llama-2-chat/13b-ft"
OUTPUTS_DIR = "/scratch/craj/model_cache/llama-2-chat/13b-ft/outputs"
MODEL_CKPT = "/scratch/craj/model_cache/llama-2-chat/13b/model"
DATASETS_CACHE_DIR = "/scratch/craj/model_cache/llama-2-chat/13b-ft/datasets"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir=OUTPUTS_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=3e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True,
)

max_seq_length = 1024

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)

model.resize_token_embeddings(len(tokenizer))
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)



def format_prompt(instruction, sys_prompt=None, input=None, output=None):
    if sys_prompt is None:
        DEFAULT_SYSTEM_PROMPT = """\
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
            """
    else:
        DEFAULT_SYSTEM_PROMPT = sys_prompt
    PROMPT_DICT = {
        "prompt_input": (
            "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n[/INST]\n{output}"
        ),
        "prompt_no_input": (
            "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n[/INST]\n{output}"
        ),
    }
    if input is None or input == "":
        return PROMPT_DICT["prompt_no_input"].format_map(
            {
                "instruction": instruction,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "output": output,
            }
        )
    else:
        return PROMPT_DICT["prompt_input"].format_map(
            {
                "instruction": instruction,
                "input": input,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "output": output,
            }
        )


def prepare_datasets(train_data_path, val_data_path):
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    train_data = train_data[["base_prompt", "Instruction", "Response"]]
    val_data = val_data[["base_prompt", "Instruction", "Response"]]
    train_data["split"] = "train"
    val_data["split"] = "validation"
    data = pd.concat([train_data, val_data])
    data = data.rename(
        columns={
            "base_prompt": "input",
            "Instruction": "instruction",
            "Response": "output",
        }
    )
    data = data[["input", "instruction", "output", "split"]]
    data = data.reset_index(drop=True)
    dataset = Dataset.from_pandas(data)

    dataset = dataset.map(
        lambda example: {
            "text": format_prompt(
                instruction=example["input"],
                input=example["instruction"],
                output=example["output"],
            ),
        },
        remove_columns=["instruction", "input", "output"],
    )

    train_dataset = dataset.filter(lambda example: example["split"] == "train")
    eval_dataset = dataset.filter(lambda example: example["split"] == "validation")

    return train_dataset, eval_dataset


data_collator = DataCollatorForCompletionOnlyLM(
    response_template=tokenizer("\n[/INST]\n")["input_ids"][2:],
    tokenizer=tokenizer,
)

train_dataset, eval_dataset = prepare_datasets(
    "../data/finetune/train.csv", 
    "../data/finetune/valid.csv"
)

# MIXING IN ALPACA DATA WITH TRAINING DATA, EVAL DATA REMAINS THE SAME
alpaca_dataset = load_dataset("yahma/alpaca-cleaned")
alpaca_dataset = alpaca_dataset["train"]
alpaca_dataset = alpaca_dataset.map(
    lambda example: {
        "text": format_prompt(
            instruction=example["instruction"], input=example["input"], output=example["output"]
        ),
    },
    remove_columns=["instruction", "input", "output"],
)
splits = alpaca_dataset.train_test_split(test_size=0.2)
alpaca_train_dataset = splits["train"]

# use only 10% of alpaca data
alpaca_train_dataset = alpaca_train_dataset.select(range(int(len(alpaca_train_dataset)*0.1)))
train_dataset = concatenate_datasets([train_dataset, alpaca_train_dataset])
train_dataset = train_dataset.shuffle(seed=SEED)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    data_collator=data_collator,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=args,
)

trainer.train()

# Save model
trainer.save_model(MODEL_CKPT)

model.push_to_hub("chahatraj/contact-base-small-13b-english-v2")
