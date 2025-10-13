""" Fully finetuning the library models for sequence classification on GLUE."""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
# import GPUtil


import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import get_last_checkpoint, TrainOutput
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
# from transformers.training_args import ParallelMode

from peft_pretraining.modeling_llama import LlamaForSequenceClassification
from peft_pretraining.aroma import AROMAModel, AROMALinear
from peft_pretraining.training_utils import get_scheduler
# from peft_pretraining.base_dataset import DataCollatorForSupervisedDataset
from peft_pretraining.commonsense_dataset import build_commonsense_dataset


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    val_set_size: Optional[int] = field(
        default=1000,
        metadata={
            "help": "Number of examples to use for validation (taken from training set)"
        }
    )

    def __post_init__(self):
        if self.dataset_dir and self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    wandb_project: Optional[str] = field(
        default="ar1lora_llama3-8B",
        metadata={"help": "WandB project name"}
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name"}
    )
    wandb_tags: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of tags for WandB run"}
    )

@dataclass
class AROMAArguments:
    """
    Arguments for AROMA training
    """
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha scaling"}
    )
    T_in: Optional[int] = field(
        default=None,
        metadata={"help": "Reset LoRA weights every N steps"}
    )
    cycle_length: Optional[int] = field(
        default=None,
        metadata={"help": "Optimizer reset cycle length"}
    )
    train_scaling: bool = field(
        default=False,
        metadata={"help": "Make LoRA scaling trainable"}
    )
    reset_optimizer_on_relora: bool = field(
        default=True,
        metadata={"help": "Reset optimizer when resetting LoRA, 默认以0.999 prune optimizer"}
    )
    num_training_steps: int = field(
        default=10000,
        metadata={"help": "Total number of training steps"}
    )
    scheduler_type: str = field(
        default="cosine_restarts",
        metadata={"help": "Scheduler type to use: linear, cosine, or cosine_restarts"}
    )
    min_lr_ratio: float = field(
        default=0.1,
        metadata={"help": "Minimum lr ratio"}
    )
    first_warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of steps for the first warmup in the scheduler"}
    )
    restart_warmup_steps: Optional[int] = field(
        default=50,
        metadata={"help": "Number of steps for the restart warmup in the scheduler"}
    )
    optimizer_random_pruning: float = field(
        default=0.0,
        metadata={"help": "Use random pruning to reduce optimizer matrix internal dimensionality, 输入一个random prune比例"}
    )
    optimizer_magnitude_pruning: float = field(
        default=0.0,
        metadata={"help": "Use magnitude pruning to reduce optimizer matrix internal dimensionality, 输入一个按幅度prune比例"}
    )
    # 添加新参数
    convergence_threshold: float = field(
        default=1e-4,
        metadata={"help": "Threshold for weight convergence check"}
    )
    check_convergence: bool = field(
        default=True,
        metadata={"help": "Whether to check convergence of weights"}
    )
    convergence_window: int = field(
        default=3,
        metadata={"help": "Convergence window size for each module"}
    )
    convergence_patience: int = field(
        default=3,
        metadata={"help": "Convergence patience for all modules"}
    )
    lora_check_frequency: int = field(
        default=10,
        metadata={"help": "Inner check interval"}
    )
    max_steps_before_reset: int = field(
        default=100,
        metadata={"help": "Maximum inner steps"}
    )
    lora_change_threshold: float = field(
        default=1e-4,
        metadata={"help": "Inner convergence threshold"}
    )

class AROMATrainer(Trainer):
    def __init__(self, AROMA_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AROMA_args = AROMA_args
        self.update_step = 0

        self.all_converged = False
        self.convergence_patience = self.AROMA_args.convergence_patience
        self.convergence_counter = 0

        if self.is_world_process_zero():
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.args.logging_dir)

    def log_metrics(self, split, metrics):
        if not self.is_world_process_zero():
            return
            
        if hasattr(self.model, "get_convergence_status"):
            convergence_status = self.model.get_convergence_status()
            metrics.update({
                "converged_modules": sum(convergence_status.values()),
                "total_modules": len(convergence_status),
                "convergence_ratio": sum(convergence_status.values()) / len(convergence_status)
            })
        
        prefixed_metrics = {
            f"{split}/{k}": v 
            for k, v in metrics.items()
        }
            
        if hasattr(self, 'writer'):
            for k, v in prefixed_metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, self.state.global_step)
        
        super().log_metrics(split, prefixed_metrics)
    
    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        logger.info(f"Creating my scheduler")
        if optimizer is None:
            optimizer = self.optimizer

        if self.AROMA_args.scheduler_type == "cosine_restarts":
            if self.AROMA_args.restart_warmup_steps is None:
                self.AROMA_args.restart_warmup_steps = self.AROMA_args.first_warmup_steps

        self.lr_scheduler = get_scheduler(
            optimizer,
            scheduler_type=self.AROMA_args.scheduler_type,
            num_training_steps=self.AROMA_args.num_training_steps,
            warmup_steps=self.AROMA_args.first_warmup_steps,
            min_lr_ratio=self.AROMA_args.min_lr_ratio,
            cycle_length=self.AROMA_args.cycle_length,
            restart_warmup_steps=self.AROMA_args.restart_warmup_steps,
            adjust_step=0,
        )
        return self.lr_scheduler

    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     self.update_step += 1
        
    #     loss = super().training_step(model, inputs, num_items_in_batch)
        
    #     if (self.update_step % self.args.gradient_accumulation_steps == 0):
    #         if self.is_world_process_zero():
    #             if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #                 num_reset, modules = model.module.check_lora_changes()
    #                 total_lora_layers = sum(1 for _ in model.module.modules() if isinstance(_, AROMALinear))
    #                 # logger.info(f"Total LoRA layers: {total_lora_layers}")
    #             else:
    #                 num_reset, modules = model.check_lora_changes()
    #                 total_lora_layers = sum(1 for _ in model.modules() if isinstance(_, AROMALinear))
    #                 # logger.info(f"Total LoRA layers: {total_lora_layers}")

    #             need_reset = (num_reset == total_lora_layers)
    #             if need_reset:
    #                 logger.info(f"All {num_reset} LoRA layers need reset")

    #             reset_tensor = torch.tensor(1 if need_reset else 0, device=model.device)
    #         else:
    #             reset_tensor = torch.tensor(0, device=model.device)
                
    #         if torch.distributed.is_initialized():
    #             torch.distributed.broadcast(reset_tensor, src=0)
    #             torch.distributed.barrier()
                
    #         need_reset = reset_tensor.item() == 1
            
    #         if need_reset:
    #             if self.is_world_process_zero():
    #                 logger.info(f"Performing LoRA reset at step {self.state.global_step}")
    #                 if modules:
    #                     logger.info(f"Modules to reset: {', '.join(modules)}")
    #                 self._perform_lora_reset(model)

    #                 logger.info(f"Performing optimizer reset at step {self.state.global_step}")
    #                 self._reset_optimizer()
                    
    #             if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #                 all_converged = model.module.check_all_converged()
    #             else:
    #                 all_converged = model.check_all_converged()
                
    #             if all_converged:
    #                 self.convergence_counter += 1
    #                 logger.info(f"All modules converged! Counter: {self.convergence_counter}/{self.convergence_patience}")
                    
    #                 if self.convergence_counter >= self.convergence_patience:
    #                     logger.info("Training stopped due to convergence")
    #                     self.all_converged = True
    #                     if self.is_world_process_zero():
    #                         self._save_checkpoint(model, trial=None)
    #                         logger.info("Final checkpoint saved before early stopping")
    #                     raise TrainerStopException("All modules converged")
    #             else:
    #                 self.convergence_counter = 0
        
    #     # if (self.state.global_step > 0 and 
    #     #     self.AROMA_args.cycle_length is not None and  
    #     #     self.state.global_step % self.AROMA_args.cycle_length == 0): # 这样写不行 虽然到T_in步才check但是第一次check的weight_change全为0 不知为啥
    #     # if (self.AROMA_args.T_in is not None and
    #     #     self.update_step % self.args.gradient_accumulation_steps == 0 and
    #     #     self.update_step / self.args.gradient_accumulation_steps % self.AROMA_args.T_in == 0):
    #     # if (self.update_step % self.args.gradient_accumulation_steps == 0 and need_reset):
    #     #     logger.info(f"Performing optimizer reset at step {self.state.global_step}")
    #     #     logger.info(f"Performing optimizer reset at update_step {self.update_step}")
    #     #     self._reset_optimizer()

    #     return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.update_step += 1

        if not isinstance(inputs, dict):
            inputs = dict(inputs)
            
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}

        loss = super().training_step(model, inputs, num_items_in_batch)
        
        if (self.AROMA_args.T_in is not None and
            self.update_step % self.args.gradient_accumulation_steps == 0 and
            self.update_step / self.args.gradient_accumulation_steps % self.AROMA_args.T_in == 0):
            
            if self.is_world_process_zero():
                logger.info(f"Performing lora reset at global_step {self.state.global_step}")
                logger.info(f"Performing lora reset at update_step {self.update_step}")
                self._perform_lora_reset(model)

                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    all_converged = model.module.check_all_converged()
                else:
                    all_converged = model.check_all_converged()
                    
                converged_tensor = torch.tensor(1 if all_converged else 0, device=model.device)
            else:
                converged_tensor = torch.tensor(0, device=model.device)

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(converged_tensor, src=0)
                torch.distributed.barrier()
                
            all_converged = converged_tensor.item() == 1
                
            if all_converged:
                self.convergence_counter += 1
                logger.info(f"All modules converged! Counter: {self.convergence_counter}/{self.convergence_patience}")
                
                if self.convergence_counter >= self.convergence_patience:
                    logger.info("Training stopped due to convergence")
                    self.all_converged = True
                    if self.is_world_process_zero():
                        self._save_checkpoint(model, trial=None)
                        logger.info("Final checkpoint saved before early stopping")
                    raise TrainerStopException("All modules converged")
            else:
                self.convergence_counter = 0
        
        if (self.AROMA_args.T_in is not None and
            self.update_step % self.args.gradient_accumulation_steps == 0 and
            self.update_step / self.args.gradient_accumulation_steps % self.AROMA_args.T_in == 0):
            logger.info(f"Performing optimizer reset at step {self.state.global_step}")
            logger.info(f"Performing optimizer reset at update_step {self.update_step}")
            self._reset_optimizer()

        return loss
    
    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except TrainerStopException as e:
            logger.info(f"Training stopped: {str(e)}")
            
            metrics = {}
            
            if hasattr(self.model, "get_convergence_status"):
                convergence_status = self.model.get_convergence_status()
                converged_modules = sum(convergence_status.values())
                total_modules = len(convergence_status)
                metrics.update({
                    "train/converged_modules": converged_modules,
                    "train/total_modules": total_modules,
                    "train/convergence_ratio": converged_modules / total_modules
                })
            
            if hasattr(self, 'train_dataset'):
                metrics["train/train_samples"] = len(self.train_dataset)
                
            logger.info("***** train metrics *****")
            for key, value in metrics.items():
                logger.info(f"  {key:30} = {value:>7}")
                
            self.log_metrics("train", metrics)
            self.save_metrics("train", metrics)
            
            return TrainOutput(
                self.state.global_step,
                float(self.state.total_flos) if self.state.total_flos else 0,
                metrics
            )

    def _perform_lora_reset(self, model):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            convergence_status = model.module.get_convergence_status()
            
            logger.info("Module convergence status:")
            for name, converged in convergence_status.items():
                logger.info(f"  {name}: {'Converged' if converged else 'Training'}")
            
            model.module.merge_check_and_reinit()
            logger.info("Completed merge_check_and_reinit for model")
        else:
            logger.warning("Model is not wrapped with DDP")    

    def _reset_optimizer(self):
        from peft_pretraining.training_utils import optimizer_reset
        optimizer_reset(
            self.optimizer,
            reset_params=[p for n, p in self.model.named_parameters() if "lora_" in n],
            optimizer_state_keys=["exp_avg", "exp_avg_sq"], # Adam优化器的state_keys
            reset_optimizer_on_relora=self.AROMA_args.reset_optimizer_on_relora,
            optimizer_random_pruning=self.AROMA_args.optimizer_random_pruning,
            optimizer_magnitude_pruning=self.AROMA_args.optimizer_magnitude_pruning,
        )

class TrainerStopException(Exception):
    pass

class DetailedLoggingCallback(TrainerCallback):
    def __init__(self):
        self.training_tracker = {
            'loss': [],
            'learning_rate': [],
            'epoch': []
        }
        self.current_step = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("=== Training Start===")
        
        if state is not None:
            logger.info(f"Starting from global step: {state.global_step}")
            logger.info(f"Starting from epoch: {state.epoch}")
        
        logger.info(f"Training batch size: {args.train_batch_size}")
        logger.info(f"Number of epochs: {args.num_train_epochs}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Weight decay: {args.weight_decay}")

        return control
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.current_step % args.logging_steps == 0:
            logger.info(f"Enter my on_step_end")
            if state.log_history:
                latest_log = state.log_history[-1]
                loss = latest_log.get('loss', None)
                lr = latest_log.get('learning_rate', None)
                epoch = latest_log.get('epoch', None)
                
                if loss is not None:
                    self.training_tracker['loss'].append(loss)
                    self.training_tracker['learning_rate'].append(lr)
                    self.training_tracker['epoch'].append(epoch)

        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.training_tracker['loss']:
            avg_loss = sum(self.training_tracker['loss']) / len(self.training_tracker['loss'])
            logger.info(f"Epoch {int(state.epoch)} finished. Average loss: {avg_loss:.4f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training completed!")
        if self.training_tracker['loss']:
            final_avg_loss = sum(self.training_tracker['loss']) / len(self.training_tracker['loss'])
            logger.info(f"Final average loss: {final_avg_loss:.4f}")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        logger.info("\n=== Evaluation ===")
        logger.info(f"Current step: {state.global_step}")
        logger.info(f"Current epoch: {state.epoch}")
        logger.info(f"Evaluation metrics: {metrics}")
        return control

class SampleOutputCallback(TrainerCallback):
        def __init__(self, tokenizer, eval_dataset, num_samples=3):
            self.tokenizer = tokenizer
            self.eval_dataset = eval_dataset
            self.num_samples = num_samples

        def on_epoch_begin(self, args, state, control, **kwargs):
            print("\n" + "="*50)
            print(f"Epoch {state.epoch}: Training Samples")
            print("="*50)
            
            train_dataset = kwargs['train_dataloader'].dataset
            indices = random.sample(range(len(train_dataset)), self.num_samples)
            train_samples = [train_dataset[i] for i in indices]
            
            for i, sample in enumerate(train_samples, 1):
                print(f"\nSample {i}:")
                print("-"*30)
                
                input_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                print(f"Input Text:\n{input_text}\n")
                
                labels = sample['labels']
                labels = [l if l != -100 else self.tokenizer.pad_token_id for l in labels]
                label_text = self.tokenizer.decode(labels, skip_special_tokens=True)
                print(f"Label Text:\n{label_text}\n")
                
                print(f"Input length: {len(sample['input_ids'])}")
                print(f"Label length: {len(sample['labels'])}")
                print("="*50)

        def on_evaluate(self, args, state, control, model, **kwargs):
            print("\n" + "="*50)
            print(f"Step {state.global_step}: Validation Samples and Predictions")
            print("="*50)
            
            indices = random.sample(range(len(self.eval_dataset)), self.num_samples)
            eval_samples = [self.eval_dataset[i] for i in indices]
            
            model.eval()
            
            for i, sample in enumerate(eval_samples, 1):
                print(f"\nValidation Sample {i}:")
                print("-"*30)
                
                input_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                print(f"Input Text:\n{input_text}\n")
                
                labels = sample['labels']
                labels = [l if l != -100 else self.tokenizer.pad_token_id for l in labels]
                label_text = self.tokenizer.decode(labels, skip_special_tokens=True)
                print(f"True Label:\n{label_text}\n")
                
                self.tokenizer.padding_side = "left"
                self.tokenizer.pad_token_id = 0
                
                inputs = {
                    'input_ids': torch.tensor([sample['input_ids']]).to(model.device),
                    'attention_mask': torch.tensor([sample['attention_mask']]).to(model.device) if 'attention_mask' in sample else None
                }
                inputs = {k: v for k, v in inputs.items() if v is not None}
                
                with torch.no_grad():
                    beam_outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        num_beams=4,
                        no_repeat_ngram_size=2,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    
                    sample_outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                beam_text = self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
                sample_text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
                
                print(f"Beam Search Prediction:\n{beam_text}\n")
                print(f"Sampling Prediction:\n{sample_text}\n")
                
                print(f"Input length: {len(sample['input_ids'])}")
                print(f"Label length: {len(sample['labels'])}")
                print(f"Beam prediction length: {len(beam_outputs[0])}")
                print(f"Sample prediction length: {len(sample_outputs[0])}")
                print("="*50)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                print("\nCurrent training metrics:")
                metrics = {k: v for k, v in logs.items() if not k.startswith('train_')}
                for key, value in metrics.items():
                    print(f"{key}: {value:.4f}")

def send_email(subject, content):
    if not hasattr(send_email, 'is_main_process'):
        send_email.is_main_process = os.environ.get('LOCAL_RANK', '0') == '0'
    
    if not send_email.is_main_process:
        return

    receiver = "your email address"
    try:
        cmd = f'echo "{content}" | mail -s "{subject}" {receiver}'
        os.system(cmd)
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.error(f"Email sending failed: {str(e)}")

def main():
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AROMAArguments)) 
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, AROMA_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, AROMA_args = parser.parse_args_into_dataclasses()

    send_email(
        "Training started",
        f"Training task has started running\n"
        f"Start time: {start_time}\n"
        f"Training parameters:\n"
        f"- learning rate: {training_args.learning_rate}\n"
        f"- batch size: {training_args.per_device_train_batch_size}\n"
        f"- T_in: {AROMA_args.T_in}\n"
        f"- LoRA rank: {AROMA_args.lora_r}"
    )

    # Setup logging
    logging.basicConfig(
        format="\033[1;32m%(asctime)s\033[0m | \033[1m%(levelname)s\033[0m | \033[1;34m%(name)s\033[0m - \033[1m%(message)s\033[0m",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dataset_name = data_args.dataset_name.lower()
    
    logger.info(f"training_args.local_rank: {training_args.local_rank}")

    if training_args.output_dir:
        task_output_dir = os.path.join(training_args.output_dir, dataset_name)
        training_args.output_dir = os.path.join(task_output_dir, "output", timestamp)
        training_args.logging_dir = os.path.join(task_output_dir, "runs", timestamp)

        if training_args.local_rank in [-1, 0]:
            os.makedirs(training_args.output_dir, exist_ok=True)
            os.makedirs(training_args.logging_dir, exist_ok=True)

    training_args.include_inputs_for_metrics = True

    # def get_gpu_memory_info():
    #     gpus = GPUtil.getGPUs()
    #     memory_info = []
    #     for gpu in gpus:
    #         memory_info.append({
    #             'id': gpu.id,
    #             'memory_used': gpu.memoryUsed,  # MB
    #             'memory_total': gpu.memoryTotal,  # MB
    #             'memory_util': gpu.memoryUtil * 100  # %
    #         })
    #     return memory_info

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")

    # Detecting last checkpoint.
    last_checkpoint = None
    logger.info(f"training_args.output_dir: {training_args.output_dir}")
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # ###################################
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        # num_labels=num_labels,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        # cache_dir=model_args.cache_dir,
        # use_fast=model_args.use_fast_tokenizer,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True
    )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if model_args.model_name_or_path is None: 
        model = LlamaForSequenceClassification(config)
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )

    def show_parameters(model):
        for name, param in model.named_parameters():
            num_params = param.numel()
            logger.info(f"layer: {name:<20} | shape: {str(param.shape):<20} | num_params: {num_params:<10} | trainable: {param.requires_grad}")

    params_before = sum(p.numel() for p in model.parameters())

    # ###################################
    if training_args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(training_args.local_rank)
        device = torch.device("cuda", training_args.local_rank)
    
    try:
        model = model.to(device)
    except RuntimeError as e:
        logger.error(f"Error moving model to device: {e}")
        logger.info("Trying alternative device initialization...")
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            raise RuntimeError("No CUDA device available")

    if AROMA_args.T_in is not None:
        need_linear_weight = True
        logger.info(f"Wrapping model with AROMA ({need_linear_weight=})")
        
        model = AROMAModel(
            model,
            r=AROMA_args.lora_r,
            lora_alpha=AROMA_args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "down_proj", "up_proj"],
            lora_dropout=0.1,
            trainable_scaling=AROMA_args.train_scaling,
            keep_original_weights=True,
            lora_only=not need_linear_weight,
            convergence_threshold=AROMA_args.convergence_threshold,
            check_convergence=AROMA_args.check_convergence,
            convergence_window=AROMA_args.convergence_window,
            lora_check_frequency=AROMA_args.lora_check_frequency,
            max_steps_before_reset=AROMA_args.max_steps_before_reset,
            lora_change_threshold=AROMA_args.lora_change_threshold,
        )
    
    params_after = sum(p.numel() for p in model.parameters())
    added_floats = params_after - params_before

    # print params and trainable params
    # logger.info(f"\n{model}\n")
    logger.info(f"Total params  before LoRA: {params_before / 1_000_000:.2f}M")
    logger.info(f"Total params  after  LoRA: {params_after / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"In total, added {added_floats / 1_000_000:.2f}M parameters to the model")
    show_parameters(model)

    if training_args.local_rank in [-1, 0]:
        config = model.config
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        
        config.save_pretrained(training_args.output_dir)

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    )
    eval_dataset=None
    train_dataset = None

    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            path = Path(data_args.dataset_dir)
            files = [os.path.join(path,file.name) for file in path.glob("*.json")]

            logger.info("\n=== Raw data samples ===")
            raw_dataset = load_dataset("json", data_files=files)
            logger.info(raw_dataset['train'][0])

            train_dataset, eval_dataset = build_commonsense_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                val_set_size=data_args.val_set_size,
                )

            # logger.info("\n=== Preprocessed samples ===")
            # logger.info("\n1. Generated prompt:")
            # from peft_pretraining.commonsense_dataset import generate_prompt
            # logger.info(generate_prompt(raw_dataset['train'][0]))
            
            # logger.info("\n2. Tokenized results:")
            # logger.info(f"Input ids: {train_dataset[0]['input_ids'][:256]}...")
            # logger.info(f"Labels: {train_dataset[0]['labels'][:256]}...")
            
            # logger.info("\n3. Decoded content:")
            # decoded_input = tokenizer.decode(train_dataset[0]['input_ids'])
            # logger.info(f"Decoded input:\n{decoded_input}")
            
            # logger.info("\n4. Statistics:")
            # logger.info(f"Input length: {len(train_dataset[0]['input_ids'])}")
            # logger.info(f"Label length: {len(train_dataset[0]['labels'])}")
            # logger.info(f"Maximum length setting: {data_args.max_seq_length}")
            # logger.info("="*50)
            
            # logger.info(f"\nNum train_samples: {len(train_dataset)}")
            # if eval_dataset:
            #     logger.info(f"Num val_samples: {len(eval_dataset)}")

    # Initialize our Trainer
    trainer = AROMATrainer(
        AROMA_args=AROMA_args,
        # task_name=data_args.task_name,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        callbacks=[DetailedLoggingCallback(), SampleOutputCallback(tokenizer,eval_dataset)],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate before training ***")
        metrics = trainer.evaluate()
        logger.info(f"Initial evaluation results: {metrics}")

    # #########################
    # Training!
    if training_args.do_train:
        logger.info("*** Training! ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        # train_start_time = time.time()
        
        # initial_gpu_info = get_gpu_memory_info()
        # logger.info(f"Initial GPU Memory Usage: {initial_gpu_info}")

        # class TimeCallback(transformers.TrainerCallback):
        #     def __init__(self):
        #         self.epoch_start_time = None
        #         self.epoch_times = []

        #     def on_epoch_begin(self, args, state, control, **kwargs):
        #         if args.local_rank in [-1, 0]:
        #             self.epoch_start_time = time.time()
        #             gpu_info = get_gpu_memory_info()
        #             logger.info(f"Epoch {state.epoch} GPU Memory Usage: {gpu_info}")

        #     def on_epoch_end(self, args, state, control, **kwargs):
        #         if args.local_rank in [-1, 0]:
        #             epoch_time = time.time() - self.epoch_start_time
        #             self.epoch_times.append(epoch_time)
        #             gpu_info = get_gpu_memory_info()
        #             logger.info(f"Epoch {state.epoch} completed in {epoch_time:.2f} seconds")
        #             logger.info(f"Epoch {state.epoch} GPU Memory Usage: {gpu_info}")

        # time_callback = TimeCallback()
        # trainer.add_callback(time_callback)

        train_result = trainer.train(resume_from_checkpoint=checkpoint) # Train!
        metrics = train_result.metrics

        # total_train_time = time.time() - train_start_time
        # logger.info(f"Total training time: {total_train_time:.2f} seconds")
        # logger.info(f"Average time per epoch: {sum(time_callback.epoch_times)/len(time_callback.epoch_times):.2f} seconds")
        
        # final_gpu_info = get_gpu_memory_info()
        # logger.info(f"Final GPU Memory Usage: {final_gpu_info}")

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        if training_args.do_eval:
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)

        if training_args.local_rank <= 0:
            trainer.save_model()
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Evaluation!
    if training_args.do_eval and eval_dataset:

        logger.info("*** Evaluate! ***")

        metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix="eval",
        )

        metrics["eval_samples"] = len(eval_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** AROMA Training Finished! ***")

    end_time = datetime.now()
    duration = end_time - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    send_email(
        "Training finished",
        f"Training task has finished running\n"
        f"Start time: {start_time}\n"
        f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Running time: {duration}\n"
        f"Output directory: {training_args.output_dir}\n"
    )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
