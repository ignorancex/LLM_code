import requests
import re
import pandas as pd
import shutil
import time
import random 
import torch
import numpy as np
from tqdm import tqdm
from ast import literal_eval
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import transformers
import datasets

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(42)



max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "meta-llama/Meta-Llama-3-8B", # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
#     token = ""
# )
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

################################################### Data Preparation ###################################################
filename = "../data/final_normal_dataset.tsv" ## for Piqa performance
data = pd.read_csv(filename, sep='\t')
label_names = list(data.columns[2:])
oov_class_map = {"play": "play", "lookthrough": "looking through", "siton": "sit on", "pourfrom": "pour from", "writewith": "write with", "typeon": "type on"}
labels_lower = [x.lower() for x in label_names]
classes = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        classes.append(lab)
    else:
        classes.append(oov_class_map[lab])

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are an AI assistant that has strong reasoning capability. You are given a context containing an object, and you are asked to answer a question about the object based on the context. Just response 'Yes' or 'No'

### Context:
{}

### Object:
{}

### Question:
{}

### Answer:
{}"""

# prompt_template = """You are an AI assistant in writing Wikipedia articles on personalities and your task is to expand the existing content of the given Wikipedia section about the personality: "{}" from the given context. Using the context generate a coherent, insightful and neutral expansion of the existing content. DO NOT use any external information. If it is not possible to expand the content from the context, say so.

# Context: "{}"

# Existing content: "{}: {}"

# Generated content: {}"""

EOS_TOKEN = tokenizer.eos_token
# def formatting_prompts_func(examples):
#     texts = []
#     for person, section_title, incomplete_content, context, output in zip(examples['title'], examples["section"], examples["incomplete_content"], examples["context"], examples["last_paragraph"]):
#         # Must add EOS_TOKEN, otherwise your generation will go on forever!
#         text = alpaca_prompt.format(person, section_title, incomplete_content, context, output) + EOS_TOKEN
#         # text = prompt_template.format(person, context, section_title, incomplete_content, context, output) + EOS_TOKEN
#         texts.append(text)
#     return { "text" : texts, }
# pass

train_prompt = []

# ################# Instruction 1 ###############################
# for idx, row in data.iterrows():

#     object_name = row['Object']
#     sentence = row['Sentence']

#     for itr in range(2, data.shape[1]):
#         class_name = list(data.columns)[itr].lower()
#         ## replacing oov classnames
#         class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

#         question = f"Can human '{class_name}' the {object_name}?"

#         answer = 'Yes' if row[itr] == 1 else 'No'

#         train_prompt.append(
#             alpaca_prompt.format(sentence, object_name, question, answer) + EOS_TOKEN
#         )

############## Instruction 2 ######################
prost_dataset = datasets.load_dataset("corypaik/prost", split="test")

prompt_template2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are an AI assistant that has strong reasoning capability. You are given a question with 4 options and you have to choose the right option. 

### Question:
{question}

### Options:
[0] {option_A}
[1] {option_B}
[2] {option_C}
[3] {option_D}

Only response the 'answer id'. For example if the answer is [0] then response 0. DO NOT respond anything other than <0, 1, 2, 3>.

### Answer:
{answer}"""

def preprocess_llama(examples):

    prompt_list = []

    for question, option_A, option_B, option_C, option_D, answer in zip(examples["ex_question"], examples["A"], examples["B"], examples["C"], examples["D"], examples["label"]):
        prompts = prompt_template2.format(question = question, 
                                        option_A=option_A,
                                        option_B=option_B, 
                                        option_C=option_C, 
                                        option_D=option_D,
                                        answer=answer) + EOS_TOKEN
        prompt_list.append(prompts)
        
    return {"prompts": prompt_list}

prost_dataset = prost_dataset.map(preprocess_llama, batched=True)

for prompts in prost_dataset[:-500]["prompts"]:
    train_prompt.append(prompts)
#################################

random.shuffle(train_prompt)

dataset = datasets.Dataset.from_pandas(pd.DataFrame({'text': train_prompt}))

print(dataset[0])

# dataset = dataset.map(formatting_prompts_func, batched = True,)
################################################### Done Dataprocessing ######################################################################

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    # load_in_8bit_fp32_cpu_offload=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_args = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    # target_modules=["gate_proj","q_proj","lm_head","o_proj","k_proj","embed_tokens","down_proj","up_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir='saved_model/',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
#     evaluation_strategy="epoch",
    optim="paged_adamw_32bit",
    save_steps=500,
    save_total_limit=5,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    # max_steps=5,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="tensorboard"
    )

peft_model = get_peft_model(model, peft_args)

# with open("actual.txt", "w") as content:
#     content.write(str(model))

# with open("peft.txt", "w") as content:
#     content.write(str(peft_model))

print(f"Peft model trainable parameters: {peft_model.print_trainable_parameters()}")

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
#     eval_dataset=test_dataset,
    peft_config=peft_args,
    dataset_text_field="text",
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_params,
    packing=True,
)

trainer.train()

trainer.model.save_pretrained("saved_model/llama-3-8B-instruct_finetuned_prost")
# model.save_pretrained("saved_model/mixed_model_llama-3-8B-instruct")
# tokenizer.save_pretrained("saved_model/mixed_llama-3-8B-instruct")