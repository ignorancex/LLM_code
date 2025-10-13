from datasets import Dataset
from soar.training.utils_process_data import formatting_prompts_func, get_len_text_sft,get_dataset_HER, get_her_repair_sft
from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template,standardize_sharegpt
import argparse
from trl import SFTTrainer, DPOTrainer, DPOConfig,DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
import resource
import os
import json
import numpy as np
from soar.preprocess import get_dataset
import copy
import shutil


def parse_arguments():
    """Parse command-line arguments for ARC experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Argument parsing for ARC experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Path configurations
    path_group = parser.add_argument_group("Path configurations")
    path_group.add_argument(
        "--base_path",
        type=str,
        default="/home/flowers/work/SOAR/",
        help="Path to git project root (evaluate_model)"
    )
    path_group.add_argument(
        "--path_archive",
        type=str,
        default="",
        help="Path to training data"
    )
    path_group.add_argument(
        "--path_archive_repair",
        type=str,
        default="",
        help="Path to repair training data"
    )
    path_group.add_argument(
        "--path_archive_test_time",
        type=str,
        default="",
        help="Path to training data for test time training"
    )
    path_group.add_argument(
        "--path_save_model",
        type=str,
        default="/home/flowers/work/hf/trained/qwen0.5",
        help="Path where to save model"
    )
    parser.add_argument(
        "--path_base_model",
        type=str,
        default="/home/flowers/work/hf/Qwen2.5-0.5B-Instruct",
        help="Path where HF models are saved"
    )

    # Hyperparameter configurations
    hyperparam_group = parser.add_argument_group("Hyperparameter configurations")
    hyperparam_group.add_argument(
        "--use_rslora",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use vLLM server"
    )
    hyperparam_group.add_argument(
        "--model_len",
        type=int,
        default=32000,
        help="Max sequence length"
    )
    hyperparam_group.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        help="Use Qlora (4-bit loading)"
    )
    hyperparam_group.add_argument(
        "--lora_r",
        type=int,
        default=256,
        help="LoRA rank"
    )
    hyperparam_group.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    hyperparam_group.add_argument(
        "--orpo",
        action=argparse.BooleanOptionalAction,
        help="Use ORPO (check if still working)"
    )
    hyperparam_group.add_argument(
        "--dpo",
        action=argparse.BooleanOptionalAction,
        help="Use DPO (check if still working)"
    )
    hyperparam_group.add_argument(
        "--bs",
        type=int,
        default=1,
        help="Batch size"
    )
    hyperparam_group.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    hyperparam_group.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    hyperparam_group.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        help="Optimizer name"
    )
    hyperparam_group.add_argument(
        "--grad_acc",
        type=int,
        default=32,
        help="Gradient accumulation steps"
    )
    hyperparam_group.add_argument(
        "--train_embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train embedding (check if still working)"
    )
    hyperparam_group.add_argument(
        "--train_on_responses_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on responses only"
    )
    hyperparam_group.add_argument(
        "--shuffle_grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle the order of grid"
    )
    hyperparam_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for optimizer"
    )

    hyperparam_group.add_argument(
        "--use_cot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use chain-of-thought"
    )

    return parser.parse_args()

# optim list here:https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
# lr should be around 8e-6 for Orpo, and 5e-5 for sft 

args = parse_arguments()


resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

# check if args.path_save_model exist if yes exit
if os.path.exists(args.path_save_model):
    if len(os.listdir(args.path_save_model))!= 0:
        print("model already exist")
        exit()
else:
    os.makedirs(args.path_save_model,exist_ok=True)

# add random sleep
import time
import random
time.sleep(random.randint(0, 20))
# load Model
max_seq_length= args.model_len
dtype = None
load_in_4bit= args.load_in_4bit
model_path=args.path_base_model
print(f"Load model from {model_path}")
print("model will be saved to: ",args.path_save_model)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, #/home/flowers/work/hf/Qwen2.5-7B-Instruct-bnb-4bit
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
if args.train_embedding:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj",
                      "up_proj", "down_proj",
                      "embed_tokens", "lm_head",]
else:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
    
model = FastLanguageModel.get_peft_model(
    model,
    r = args.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = target_modules,
    lora_alpha = args.lora_alpha, 
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = args.use_rslora,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
if "qwen" in model_path.lower():    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5", # TODO: do something so it work for all LLM
    )
    tokenizer.eos_token = "<|im_end|>"

elif "llama" in model_path.lower():
    print("use llama template")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    
else:
    pass
# if not "instruct" in model_path.lower():

# load dataset

orpo = args.orpo
max_seq_length_allowed=args.model_len

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#TODO: sample previous dataset (+ post train dataset after FT for gen>0)

train_data, val_data, test_data=get_dataset(data_path=args.base_path)
train_data_arc2, val_data_arc2, _=get_dataset(data_path=args.base_path,arc_2=True)

train_val_data_all=copy.deepcopy(train_data_arc2)
train_val_data_all.update(val_data_arc2)

train_val_data_arc_1=copy.deepcopy(train_data)
train_val_data_arc_1.update(val_data)

train_val_data_all.update(train_val_data_arc_1)

train_val_data=train_val_data_all

print("="*20)
print("Finetune with SFT")
print("="*20)
print("Process dataset:")
dataset=[]

print("path_her: ",args.path_archive)
if args.path_archive!="" :
    
    dataset_her = get_dataset_HER(args.path_archive,train_val_data,n_sample=None,use_fewshot_example=False,use_cot=args.use_cot,shuffle=args.shuffle_grid)
    print("len dataset_her: ",len(dataset_her))
    dataset += dataset_her
print("path_data_repair: ",args.path_archive_repair)
if args.path_archive_repair != "":
    dataset_her_repair = get_her_repair_sft(args.path_archive_repair,train_val_data,shuffle=args.shuffle_grid)
    dataset += dataset_her_repair
    print("len dataset_her_repair: ",len(dataset_her_repair))
if args.path_archive_test_time!="":
    dataset_her_test_time_ft = get_dataset_HER(args.path_archive_test_time,train_val_data,n_sample=None,use_fewshot_example=False,use_cot=args.use_cot,shuffle=args.shuffle_grid)
    print("len dataset_her_test_time_ft: ",len(dataset_her_test_time_ft))
    dataset += dataset_her_test_time_ft
if len(dataset)==0:
    raise ValueError("No data to train")
# normal sft

dataset=Dataset.from_list(dataset)
dataset = standardize_sharegpt(dataset)

dataset = dataset.map(formatting_prompts_func,fn_kwargs = {"tokenizer": tokenizer}, batched = True,)
# print(dataset[5]["text"])
print("init trainer")
# shuffle dataset
dataset = dataset.shuffle(seed=3407)

length = dataset.map(get_len_text_sft,fn_kwargs = {"tokenizer": tokenizer},num_proc = 30)
list_length = np.array(length["len_prompt"])
print("===="*10)
print("===="*10)

print("max seq len =", max(list_length))
print("===="*10)
print("===="*10)

if max(list_length)>max_seq_length_allowed:

    idx_2remove = list_length > max_seq_length_allowed
    dataset = dataset.filter(lambda example, idx: not idx_2remove[idx], with_indices=True)
    length = dataset.map(get_len_text_sft,fn_kwargs = {"tokenizer": tokenizer},num_proc = 30)
    list_length = np.array(length["len_prompt"])
    print("max seq len after filter =", max(list_length))

if args.train_on_responses_only:
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)
    response_template="[/INST]"
    if "mistral" in model_path.lower():
        
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer,mlm=False)
    # elif "mistral" in model_path.lower():
    #     print("mistr resp only")
    #     trainer = train_on_responses_only(
    #         trainer,
    #         instruction_part = "</s>[INST]",
    #         response_part = "[/INST]",
    #     )    

else:
    data_collator=None

if args.train_embedding:
    from unsloth import UnslothTrainer, UnslothTrainingArguments
    print("use UnslothTrainer")
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = data_collator,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = UnslothTrainingArguments(
            per_device_train_batch_size = args.bs,
            gradient_accumulation_steps = args.grad_acc,
            save_strategy="no",
            # warmup_steps = 5,
            num_train_epochs = args.epochs, # Set this for 1 full training run.
            # max_steps = 60,
            learning_rate = args.lr,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = args.optim,
            weight_decay = args.weight_decay,
            lr_scheduler_type = "cosine",
            warmup_ratio = 0.1,
            seed = 3407,
            output_dir = args.path_save_model,

            embedding_learning_rate = 5e-6,
        ),
    )
else:
    print("use SFTTrainer")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = data_collator,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = args.bs,
            gradient_accumulation_steps = args.grad_acc,
            save_strategy="no",
            num_train_epochs = args.epochs, 
            learning_rate = args.lr,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = args.optim,
            weight_decay = args.weight_decay,
            lr_scheduler_type = "cosine",
            warmup_ratio = 0.1,
            seed = 3407,
            output_dir = args.path_save_model,
        ),
    )
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
# train on completion only
if args.train_on_responses_only:
    print("train on responses only")
    from unsloth.chat_templates import train_on_responses_only
    if "llama" in model_path.lower():
        print("llama resp only")
        trainer = train_on_responses_only(
                trainer,
                instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
                response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
            )

    elif "qwen" in model_path.lower():
        print("qwen resp only")
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )    
    else:
        pass

space = tokenizer(" ", add_special_tokens = False).input_ids[0]
# print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))

# if args.old_version:
#     trainer_stats = trainer.train()
# else:
# from unsloth import unsloth_train
# trainer_stats = unsloth_train(trainer)#.train()

trainer_stats = trainer.train()
#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
print("="*20)
print("save model: ",args.path_save_model)
print("="*20)

model.save_pretrained_merged(args.path_save_model, tokenizer, save_method = "merged_16bit",)
with open(args.path_save_model+"/trainer_logs.json", 'w') as f:
    json.dump(trainer.state.log_history, f)
