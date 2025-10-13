import argparse
import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_from_disk
from peft import LoraConfig, PeftModelForCausalLM, get_peft_model

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

os.environ["WANDB_DISABLED"] = "true"


logger = logging.getLogger(__name__)


def parse_cmd_args():
    # TODO: switch to HfArgumentParser
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_source_length", type=int, default=8192)
    parser.add_argument(
        "--checkpoint_path", type=str, default="microsoft/Phi-3-mini-128k-instruct"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed to use during training with the HF trainer",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="path to deepspeed config file",
    )
    parser.add_argument("--task", type=str, default="baseline")
    args = parser.parse_args()
    return args


def main():

    args = parse_cmd_args()

    set_seed(args.seed)

    ###################
    # Hyper-parameters
    ###################

    training_config = {
        "do_train": True,
        "do_eval": True,
        "do_predict": False,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # "optim": "",
        "learning_rate": 5.0e-06,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.2,
        "num_train_epochs": args.num_train_epochs,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        # "predict_with_generate": True,
        # "generation_num_beams": 5,
        # "generation_config": GenerationConfig(**generation_config),
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "log_level": "info",
        "logging_strategy": "epoch",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "log_on_each_node": False,
        "report_to": "tensorboard",
        # "ddp_backend": args.ddp_backend,
        "deepspeed": args.deepspeed_config,
        # "resume_from_checkpoint": args.resume_from_checkpoint,
    }

    # https://huggingface.co/docs/peft/v0.14.0/en/package_reference/lora#peft.LoraConfig

    peft_config = {
        "r": 24,  # rank
        "lora_alpha": 32,  # The alpha parameter for Lora scaling
        "lora_dropout": 0.05,  # The dropout probability for Lora layers
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        # "modules_to_save": ["embed_tokens", "lm_head"],
        "modules_to_save": None,
    }
    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
        + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_conf}")
    logger.info(f"PEFT parameters {peft_conf}")

    ################
    # tokenizer loading
    ################
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    tokenizer.model_max_length = args.max_source_length + args.max_new_tokens
    tokenizer.pad_token = (
        tokenizer.unk_token
    )  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"

    ################
    # Model Loading
    ################
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs)

    # if args.task == "e2e" or args.task == "multitask":
    #     # for e2e and multitask settings, add special tokens and expand the token embeddings
    #     special_tokens_dict = {"additional_special_tokens": ["<|plan|>", "<|summary|>"]}
    #     tokenizer.add_special_tokens(special_tokens_dict)
    #     model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    peft_model = get_peft_model(model, peft_conf)
    # peft_model = PeftModelForCausalLM(model, peft_conf)
    print("### TRAINABLE PARAMETERS ###")
    peft_model.print_trainable_parameters()
    print("############################")

    ################
    # Data Loading
    ################

    dataset = load_from_disk(args.dataset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation_for_training"]

    ###########
    # Training
    ###########

    response_template = "<|assistant|>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=peft_model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=tokenizer.model_max_length,
        dataset_text_field="text",
        tokenizer=tokenizer,
        data_collator=collator,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    trainer.train()
    # metrics = train_result.metrics
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
