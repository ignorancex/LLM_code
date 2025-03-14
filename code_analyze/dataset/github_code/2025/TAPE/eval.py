import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from config_llama import MyLlamaConfig

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    torch_dtype = torch.float16

    if args.seed is not None:
        set_seed(args.seed)

    if 'c4' not in args.dataset_cache_dir:
        raw_datasets = load_dataset("arrow", data_files={'test': f"{args.dataset_cache_dir}/test/data*.arrow"}, streaming=True)
        # raw_datasets = load_from_disk(args.dataset_cache_dir)
        raw_datasets = raw_datasets["test"]
    else:
        from train import load_json_dataset
        args.output_dir = 'debug'
        args.do_train, args.do_eval, args.do_predict = False, True, False
        raw_datasets = load_json_dataset(args, "/scratch/gpfs/DATASETS/hugging_face/c4/en")
        raw_datasets = raw_datasets["validation"]
    print(raw_datasets)


    if args.config_name:
        config = MyLlamaConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = MyLlamaConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You need to specify config."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if config.rpe_type in ['yarn', 'adayarn']:
    #     config.rope_scaling = {
    #         "type": config.rpe_type,
    #         "factor": training_args.rope_scaling_factor
    #         }
    #     config.rope_scaling["original_max_position_embeddings"] = training_args.model_max_position_embeddings
    config.max_position_embeddings = args.block_size

    try:
        module_name = config.rpe_type
        MyLlamaForCausalLM = __import__(f"models.llama.{module_name}", fromlist=["MyLlamaForCausalLM"]).MyLlamaForCausalLM
    except:
        rpe_types = [
            "rope", "sincos", "randrope", "alibi", "adarope", "yarn", 
            "t5rb", "fire", "xpos", "nope", "adayarn", "adalibi",
        ]
        raise NotImplementedError(f"Unknown positional embedding {module_name}, choose from {rpe_types}")
    # if config.rpe_type == "bipe_rope" or config.rpe_type == "rope":
    #     LlamaForCausalLM = MyLlamaForCausalLM_bipe_rope
    # elif config.rpe_type == "bipe_alibi" or config.rpe_type == "alibi":
    #     LlamaForCausalLM = MyLlamaForCausalLM_bipe_alibi
    # elif config.rpe_type == 'adape':
    #     from models.llama.adarope import MyLlamaForCausalLM
    #     LlamaForCausalLM = MyLlamaForCausalLM
    # elif config.rpe_type== 'ada_rope':
    #     from models.llama.ada_rope import MyLlamaForCausalLM
    #     LlamaForCausalLM = MyLlamaForCausalLM
    # elif config.rpe_type == 'new_rope':
    #     from models.llama.new_rope import MyLlamaForCausalLM
    #     LlamaForCausalLM = MyLlamaForCausalLM
    # else:
    #     raise NotImplementedError

    # if 'debug':
    #     from models.llama.new_rope import MyLlamaForCausalLM
    #     config.position_size = 36
    #     LlamaForCausalLM = MyLlamaForCausalLM

    model = MyLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        # ignore_mismatched_sizes=True,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    from utils import infer_columns_of_dataset
    column_names = infer_columns_of_dataset(raw_datasets)
    # print(raw_datasets.features)

    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    os.makedirs(f"{args.dataset_cache_dir}/tokenized", exist_ok=True)
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        # num_proc=args.preprocessing_num_workers,
        # load_from_cache_file=not args.overwrite_cache,
        # cache_file_name=f"{args.dataset_cache_dir}/tokenized/tokenized_datasets_validation.arrow",
        # desc="Running tokenizer on dataset",
    )

    if args.block_size is None:
        block_size = tokenizer.model_max_length

    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size - block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(f"{args.dataset_cache_dir}/{args.block_size}", exist_ok=True)
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        # num_proc=args.preprocessing_num_workers,
        # load_from_cache_file=not args.overwrite_cache,
        # cache_file_name=f"{args.dataset_cache_dir}/{args.block_size}/lm_datasets_validation.arrow",
        # desc=f"Grouping texts in chunks of {block_size}",
    )

    def extract_name(path, type):
        if type == "data":
            return path.split('_')[-1]
        elif type == "model":
            paths = path.split('/')
            name = [part for part in paths if 'pile' in part or 'c4' in part][-1]
            return name.rpartition('_')[0]

    data_name = extract_name(args.dataset_cache_dir, "data")
    model_name = extract_name(args.model_name_or_path, "model")

    eval_dataset = lm_datasets

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    accelerator = Accelerator()

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), leave=False):
        with torch.no_grad():
            # if batch["input_ids"] is None:
            #     continue
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    
    csv_file = './assets/results.csv'

    if accelerator.is_main_process:
        import csv
        # 检查文件是否存在，如果不存在则写入标题行
        if not os.path.isfile(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['data', 'model', 'block_size', 'perplexity'])

        # 写入数据行
        with open(csv_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data_name, model_name, args.block_size, perplexity])

if __name__ == "__main__":
    main()