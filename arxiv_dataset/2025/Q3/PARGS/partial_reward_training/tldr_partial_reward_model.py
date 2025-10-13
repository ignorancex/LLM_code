# This training script is adapted from the TRL library: https://github.com/huggingface/trl/

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python tldr_partial_reward_model.py \
    --model_name_or_path="OpenAssistant/reward-model-deberta-v3-large"\
    --output_dir="partial_reward_modeling_tldr" \
    --per_device_train_batch_size=16 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=5e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
"""
# sys.argv[1] -- output path

import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, GPT2Tokenizer, AutoModelForCausalLM
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import os
import torch.utils.data as data_utils
from transformers import GPT2Tokenizer, DataCollatorWithPadding, DefaultDataCollator, DataCollatorForSeq2Seq, RobertaTokenizer
from datasets import load_dataset, load_from_disk, Dataset # huggingface datasets
from typing import *
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import wandb


tqdm.pandas()


if __name__ == "__main__":
    wandb.init(project="Reward_Models")
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    reward_config, model_config = parser.parse_args_into_dataclasses()
    
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    reward_config.fp16 = True
    
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )
    model.config.pad_token_id = model.config.eos_token_id

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    path = f"./"

    # raw_datasets_train = load_from_disk(f'{path}/summarize_from_feedback_train')
    # raw_datasets_eval = load_from_disk(f'{path}/summarize_from_feedback_eval')
    raw_datasets_train = load_from_disk(f'{path}/summarize_from_feedback_train_partial')
    raw_datasets_eval = load_from_disk(f'{path}/summarize_from_feedback_eval_partial')

    train_dataset = raw_datasets_train
    idxs = np.random.choice(len(raw_datasets_eval), size=1024, replace=False)
    eval_dataset = raw_datasets_eval.select(idxs)


    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()
    trainer.save_model(reward_config.output_dir)
