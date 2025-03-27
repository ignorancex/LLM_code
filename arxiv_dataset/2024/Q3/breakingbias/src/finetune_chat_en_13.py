import random
from functools import partial

import numpy as np
import torch
from datasets import load_dataset
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
wandb.init(project=f"alpaca-2-{LLAMA_SIZE}b-chat", name=f"{LANG}-alpaca-2-chat")

use_flash_attention = True if torch.cuda.get_device_capability()[0] >= 8 else False

if use_flash_attention:
    print("Using flash attention")
    replace_attn_with_flash_attn()
    use_flash_attention = True

# Hugging Face model id
model_id = f"meta-llama/Llama-2-{LLAMA_SIZE}b-chat-hf"
MODEL_CACHE_DIR = f"/projects/antonis/anjishnu/llama-2/{LLAMA_SIZE}b-chat"
OUTPUTS_DIR = "/projects/antonis/anjishnu/finetune"
MODEL_CKPT = "/projects/antonis/anjishnu/finetune/model"
DATASETS_CACHE_DIR = "/projects/antonis/anjishnu/alpaca-cleaned"

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
    num_train_epochs=3,
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


def format_prompt(instruction, sys_prompt=None, input=None, output=None, lang="en"):
    if lang == "en":
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


max_seq_length = 2048  # max sequence length for model and packing of the dataset
if LANG == "english":
    data_path = "yahma/alpaca-cleaned"
else:
    data_path = f"iamshnoo/alpaca-cleaned-{LANG}"
dataset = load_dataset(data_path, cache_dir=DATASETS_CACHE_DIR)
dataset = dataset["train"]

if LANG == "english":
    dataset = dataset.map(
        lambda example: {
            "text": format_prompt(
                instruction=example["instruction"],
                input=example["input"],
                output=example["output"],
                lang="en",
            ),
        },
        remove_columns=["instruction", "input", "output"],
    )

splits = dataset.train_test_split(test_size=0.2)
train_dataset = splits["train"]
eval_dataset = splits["test"]


# print(tokenizer("\n\n### उत्तर:")['input_ids'][2:])
# print(tokenizer("\n\n### Response:")['input_ids'][2:])
# https://github.com/huggingface/trl/issues/598

if LANG == "english":
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer("\n[/INST]\n")["input_ids"][2:],
        tokenizer=tokenizer,
    )

# which id is 29901
# # print(tokenizer.convert_ids_to_tokens([835]))
# print(tokenizer.convert_ids_to_tokens([13291]))
# print(tokenizer.convert_ids_to_tokens([6160]))
# print(tokenizer.convert_ids_to_tokens([12636]))
# # print(tokenizer.convert_ids_to_tokens([29901]))
# print(tokenizer.convert_ids_to_tokens([29871]))


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

# Evaluate - old config
# generation_config = GenerationConfig(
#     num_beams=4,
#     temperature=0.2,
#     top_p=0.95,
# )

# Evaluate - https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/model.py
generation_config = GenerationConfig(num_beams=1, temperature=0.8, top_p=0.95, top_k=50)

if LANG == "english":
    inst = "You are an expert in global cultures and the subject matter of the question. Using that expertise, identify the most accurate answer from the options provided for the given question. Options : (A) right (B) left (C) middle (D) anywhere"
    question = "Which side do people usually keep when driving in the United States?"
    sys_prompt = "Only choose the correct option. Do not add any extra information."
    prompt = format_prompt(
        instruction=question, sys_prompt=sys_prompt, input=inst, output="", lang="en"
    )

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
with torch.cuda.amp.autocast():
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=10,
    )
for s in generation_output.sequences:
    output = tokenizer.decode(s)
    out_sentence = output.split("[/INST]\n")[1].strip()
    # remove the </s> at the end of the output_sentence
    out_sentence = out_sentence.replace("</s>", "")
    print("Answer : ", out_sentence)

# Save model
trainer.save_model(MODEL_CKPT)

model.push_to_hub(f"iamshnoo/alpaca-2-{LLAMA_SIZE}b-{LANG}-chat")
