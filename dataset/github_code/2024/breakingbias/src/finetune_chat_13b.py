# import pandas as pd
# import numpy as np

# path = "../data/finetune/finetunedata.csv"
# df = pd.read_csv(path)
# print(len(df)*0.1)

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
from functools import partial

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig,
                          TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import wandb
from llama_patch import forward, replace_attn_with_flash_attn

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LANG = "english"
LLAMA_SIZE = "13"
# wandb.init(project=f"contact-theory-chat", name=f"{LLAMA_SIZE}b-{LANG}")

use_flash_attention = True if torch.cuda.get_device_capability()[0] >= 8 else False

if LLAMA_SIZE == "70":
    use_flash_attention = False

if use_flash_attention:
    print("Using flash attention")
    replace_attn_with_flash_attn()
    use_flash_attention = True

# Hugging Face model id
model_id = f"meta-llama/Llama-2-{LLAMA_SIZE}b-chat-hf"
MODEL_CACHE_DIR = f"/scratch/craj/model_cache/llama-2-chat/{LLAMA_SIZE}b-ft"
OUTPUTS_DIR = f"/scratch/craj/model_cache/llama-2-chat/{LLAMA_SIZE}b-ft/outputs"
MODEL_CKPT = f"/scratch/craj/model_cache/llama-2-chat/{LLAMA_SIZE}b/model"
DATASETS_CACHE_DIR = f"/scratch/craj/model_cache/llama-2-chat/{LLAMA_SIZE}b-ft/datasets"

# BitsAndBytesConfig int-4 config
# https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)
# https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/llama2#transformers.LlamaConfig
model.config.pretraining_tp = 1  # necessary to reproduce pretraining results correctly

# Validate that the model is using flash attention, by comparing doc strings
if use_flash_attention:
    assert (
        model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__
    ), "Model is not using flash attention"


tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
# https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/llama2#overview
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

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
    disable_tqdm=True,  # disable tqdm since with packing values are in correct
)


def format_prompt(instruction, input, output, sys_prompt=None):
    if sys_prompt is None:
        DEFAULT_SYSTEM_PROMPT = """\
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
            """
    else:
        DEFAULT_SYSTEM_PROMPT = sys_prompt
    PROMPT_DICT = {
        "prompt_input": (
            "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n Follow closely the instruction given next. Instruction: {instruction} Answer in one word. \n Question: {input} \n Answer:\n[/INST]\n{output}"
        )
    }

    return PROMPT_DICT["prompt_input"].format_map(
        {
            "instruction": instruction,
            "input": input,
            "output": output,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        }
    )


max_seq_length = 2048  # max sequence length for model and packing of the dataset

# create dataset in hugingface format
train_data = pd.read_csv("../data/finetune/train.csv")
val_data = pd.read_csv("../data/finetune/valid.csv")
train_data = train_data[["base_prompt", "Instruction", "Response"]]
val_data = val_data[["base_prompt", "Instruction", "Response"]]
train_data["split"] = "train"
val_data["split"] = "validation"
data = pd.concat([train_data, val_data])
data = data.rename(
    columns={"base_prompt": "input", "Instruction": "instruction", "Response": "output"}
)
data = data[["input", "instruction", "output", "split"]]
data = data.reset_index(drop=True)
dataset = Dataset.from_pandas(data)

dataset = dataset.map(
    lambda example: {
        "text": format_prompt(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
            sys_prompt=None,
        ),
    },
    remove_columns=["instruction", "input", "output"],
)

train_dataset = dataset.filter(lambda example: example["split"] == "train")
eval_dataset = dataset.filter(lambda example: example["split"] == "validation")

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=tokenizer("\n[/INST]\n")["input_ids"][2:],
    tokenizer=tokenizer,
)

print(train_dataset[0])

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     dataset_text_field="text",
#     data_collator=data_collator,
#     peft_config=peft_config,
#     max_seq_length=max_seq_length,
#     tokenizer=tokenizer,
#     args=args,
# )

# trainer.train()

# generation_config = GenerationConfig(
#     num_beams=1,
#     temperature=0.8,
#     top_p=0.95,
#     top_k=50
# )

# inst = "Answer in one word. Choose between yes/no."
# question = "Should i collaborate with deaf people?"
# sys_prompt = "Only choose the correct option. Do not add any extra information."
# prompt = format_prompt(instruction=inst, sys_prompt=sys_prompt, input=question, output="")

# inputs = tokenizer(prompt, return_tensors="pt")
# input_ids = inputs["input_ids"].to(model.device)
# with torch.cuda.amp.autocast():
#     generation_output = model.generate(
#         input_ids=input_ids,
#         generation_config=generation_config,
#         return_dict_in_generate=True,
#         output_scores=True,
#         max_new_tokens=10,
#     )
# for s in generation_output.sequences:
#     output = tokenizer.decode(s)
#     out_sentence = output.split("[/INST]\n")[1].strip()
#     out_sentence = out_sentence.replace("</s>", "")
#     print("Answer : ", out_sentence)

# Save model
# trainer.save_model(MODEL_CKPT)

# model.push_to_hub(f"chahatraj/contact-base-small-{LLAMA_SIZE}b-{LANG}")
