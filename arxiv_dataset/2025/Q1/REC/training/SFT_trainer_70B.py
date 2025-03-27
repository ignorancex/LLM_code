import json
import gc
import os
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoConfig,
)
from trl import SFTTrainer, is_xpu_available, DataCollatorForCompletionOnlyLM
from datetime import datetime
import random

# Config
base_model_name = "Skywork/Skywork-Critic-Llama-3.1-70B"
output_dir = "./run2"
padding_side_req = "right" #by default mistral uses left, but trl has overflow issues with left padding.
packing = False # not required for small datasets
max_seq_length = 4096

input_data_dir1 = "./data/train.jsonl" # unzip the `REC_data.zip` citations + explanation data and put the file here
input_data_dir2 = "nvidia/HelpSteer2"
input_data_dir3 = "NCSOFT/offsetbias"
input_data_dir4 = "Vezora/Code-Preference-Pairs"

peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# General prompt template for Mistral
SYSTEM_PROMPT = '''Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'''
USER_PROMPT = '''[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]'''

def apply_template(system_prompt, user_prompt, response):
    return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>'''

def apply_template_REC(user_prompt, response):
    return f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>'''

def apply_formatting_helpsteer(example1, example2, tokenizer):
    instruction = example1["prompt"]
    response1 = example1["response"]
    response2 = example2["response"]
    prompt = USER_PROMPT.format(question=instruction, answer_a=response1, answer_b=response2)

    response1_score = 0.65 * example1["helpfulness"] + 0.8 * example1["correctness"] + 0.45 * example1["coherence"]
    response2_score = 0.65 * example2["helpfulness"] + 0.8 * example2["correctness"] + 0.45 * example2["coherence"]
    if response1_score > response2_score:
        response = "[[A]]"
    else:
        response = "[[B]]"
    final_prompt = apply_template(SYSTEM_PROMPT, prompt, response)

    return {"formatted_prompt": final_prompt}

def apply_formatting_offsetbias(example, tokenizer):
    instruction = example["instruction"]
    output1 = example["output_1"]
    output2 = example["output_2"]
    prompt = USER_PROMPT.format(question=instruction, answer_a=output1, answer_b=output2)

    if example["label"] == 1:
        response = "[[A]]"
    else:
        response = "[[B]]"
    final_prompt = apply_template(SYSTEM_PROMPT, prompt, response)

    return {"formatted_prompt": final_prompt}

def apply_formatting_REC(example, tokenizer):
    prompt = example["prompt"]
    output = example["completion"]

    formatted_prompt = apply_template_REC(prompt, output)
    return {"formatted_prompt": formatted_prompt}

def apply_formatting_code(example, tokenizer):
    instruction = example["input"]
    output1 = example["accepted"]
    output2 = example["rejected"]
    prompt1 = USER_PROMPT.format(question=instruction, answer_a=output1, answer_b=output2)
    prompt2 = USER_PROMPT.format(question=instruction, answer_a=output2, answer_b=output1)
    response1 = "[[A]]"
    response2 = "[[B]]"
    
    prompts = [
        apply_template(SYSTEM_PROMPT, prompt1, response1),
        apply_template(SYSTEM_PROMPT, prompt2, response2)
    ]
    # Randomly select one of the prompts
    selected_prompt = random.choice(prompts)
    return {"formatted_prompt": selected_prompt}

def filter_by_length(example, tokenizer):
    len_text = len(tokenizer(example["formatted_prompt"], padding=False, truncation=False)["input_ids"])
    return True if len_text <= max_seq_length else False

def load_and_format_datasets(input_data_dir1, input_data_dir2, input_data_dir3, input_data_dir4, tokenizer):
    # Initialize an empty list to hold the data
    train_ds1 = load_dataset('json', data_files=input_data_dir1, split='train').shuffle(seed=82)
    original_columns=train_ds1.column_names
    #format dataset
    train_dataset1 = train_ds1.map(
        function=lambda example: apply_formatting_REC(example, tokenizer),
        remove_columns=original_columns)
    train_dataset1 = train_dataset1.filter(lambda example: filter_by_length(example, tokenizer))
    
    train_ds2 = load_dataset(input_data_dir2)["train"]
    row_list = []
    #format dataset
    for i in range(0, len(train_ds2), 2):
        row_list.append(apply_formatting_helpsteer(train_ds2[i], train_ds2[i+1], tokenizer))
    train_dataset2 = Dataset.from_list(row_list) 
    train_dataset2 = train_dataset2.filter(lambda example: filter_by_length(example, tokenizer))
    print(f"""train_ds2: {len(train_dataset2)}""")

    train_ds3 = load_dataset(input_data_dir3)["train"]
    original_columns=train_ds3.column_names
    #format dataset
    train_dataset3 = train_ds3.map(
        function=lambda example: apply_formatting_offsetbias(example, tokenizer),
        remove_columns=original_columns)
    train_dataset3 = train_dataset3.filter(lambda example: filter_by_length(example, tokenizer))
    print(f"""train_ds3: {len(train_dataset3)}""")
    
    train_ds4 = load_dataset(input_data_dir4)["train"]
    original_columns=train_ds4.column_names
    #format dataset
    train_dataset4 = train_ds4.map(
        function=lambda example: apply_formatting_code(example, tokenizer),
        remove_columns=original_columns)
    train_dataset4 = train_dataset4.filter(lambda example: filter_by_length(example, tokenizer))
    train_dataset4 = train_dataset4.select(range(10000))
    print(f"""train_ds4: {len(train_dataset4)}""")
    
    # Combine the datasets and shuffle
    train_dataset = concatenate_datasets([train_dataset1, train_dataset2, train_dataset3, train_dataset4]).shuffle(seed=42)
    print(f"""train_ds: {len(train_dataset)}""")

    return train_dataset

def initialize_tokenizer_and_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, #floa16 needed for flash attn https://huggingface.co/docs/transformers/main/en/model_doc/mistral
=        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )
    )
    #base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False #use_cache=False if gradient_checkpointing else True
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                              add_bos_token=False,
                                              add_eos_token=False)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    '''
    tokenizer.pad_token = tokenizer.eos_token #If you set the eos_token as a padding token, the tokenizer will set the eos_token attention mask as “0”. The model will tend to ignore the eos_token and might over-generate tokens, which is not ideal for a down-streaming task. A more suitable option will be unk_token , because of its rarity, it will barely degrade the model’s performance even if we set its attention mask to “0” (i.e., set it as a padding token)
    tokenizer.padding_side = padding_side_req
    tokenizer.model_max_length = max_seq_length ######################## to deal with the larger context window of llama3.1
    '''
    return tokenizer, base_model


def main():
    tokenizer, base_model = initialize_tokenizer_and_model()
    train_dataset = load_and_format_datasets(input_data_dir1, input_data_dir2, input_data_dir3,input_data_dir4, tokenizer)
    print("Example 0:\nPrompt after preprocessing:", train_dataset[0]["formatted_prompt"])
    print("****"*20)
    # training config
    training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_ratio=0.1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        group_by_length=True, #for training efficiency
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,  # about 10x smaller than the Mistral pretraining learning rate
        logging_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        bf16=True,
        do_eval=False,
        #evaluation_strategy="epoch",
        #save_strategy="epoch",
        #save_strategy="epoch",
        save_strategy="steps",
        save_steps=500,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        run_name=f"logs-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    assert tokenizer.padding_side == padding_side_req
    
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        packing=packing,
        dataset_text_field="formatted_prompt",
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        neftune_noise_alpha=5,
        data_collator=collator
    )
    
    print ("After SFTrainer processing input_ids[0]=", trainer.train_dataset["input_ids"][0])
    print ("After SFTrainer processing attention_mask[0]=", trainer.train_dataset["attention_mask"][0])
    
    trainer.train()
    trainer.model.save_pretrained("final_checkpoint")
    tokenizer.save_pretrained("final_checkpoint")
    
    # Flush memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    #trainer.save_model(output_dir)

if __name__ == "__main__":
    main()