import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info
from open_r1.constants import system_prompt_registry
from open_r1.utils import *
from open_r1.arguments import *
from open_r1.dataset import MMR1SFTDataset

logger = logging.getLogger(__name__)

processor = None

def collate_fn(examples):
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    prompts = [json.loads(example['prompt']) for example in examples]
    texts = [
        processor.apply_chat_template(prompt, tokenize=False) for prompt in prompts
    ]  # Prepare texts for processing
    image_inputs = [example['image'] for example in examples]
    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    os.makedirs(os.path.join(training_args.output_dir, 'config'), exist_ok=True)
    save_args_to_txt(script_args, os.path.join(training_args.output_dir, 'config', 'script_args.txt'))
    save_args_to_txt(training_args, os.path.join(training_args.output_dir, 'config', 'training_args.txt'))
    save_args_to_txt(model_args, os.path.join(training_args.output_dir, 'config', 'model_args.txt'))
    
    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    dataset = MMR1SFTDataset(script_args)
    ################
    # Load tokenizer
    ################
    global processor
    
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if hasattr(script_args, "max_pixels"):
        processor.image_processor.max_pixels = script_args.max_pixels
    
    ###################
    # Load model
    ###################
    logger.info("*** Loading model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["MMR1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
