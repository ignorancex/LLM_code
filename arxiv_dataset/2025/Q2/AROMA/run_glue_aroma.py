""" Fully finetuning the library models for sequence classification on GLUE."""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import time
from datetime import datetime, timedelta
import json


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
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, TrainOutput
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from peft_pretraining.modeling_llama import LlamaForSequenceClassification
from peft_pretraining.aroma1 import AROMAModel, AROMALinear
from peft_pretraining.training_utils import get_scheduler


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
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
    def __init__(self, AROMA_args=None, task_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AROMA_args = AROMA_args
        self.task_name = task_name
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.update_step += 1
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        if (self.update_step % self.args.gradient_accumulation_steps == 0):
            if self.is_world_process_zero():
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    num_reset, modules = model.module.check_lora_changes()
                    total_lora_layers = sum(1 for _ in model.module.modules() if isinstance(_, AROMALinear))
                    # logger.info(f"Total LoRA layers: {total_lora_layers}")
                else:
                    num_reset, modules = model.check_lora_changes()
                    total_lora_layers = sum(1 for _ in model.modules() if isinstance(_, AROMALinear))
                    # logger.info(f"Total LoRA layers: {total_lora_layers}")

                need_reset = (num_reset == total_lora_layers)
                if need_reset:
                    logger.info(f"All {num_reset} LoRA layers need reset")

                reset_tensor = torch.tensor(1 if need_reset else 0, device=model.device)
            else:
                reset_tensor = torch.tensor(0, device=model.device)
                
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(reset_tensor, src=0)
                torch.distributed.barrier()
                
            need_reset = reset_tensor.item() == 1
            
            if need_reset:
                if self.is_world_process_zero():
                    logger.info(f"Performing LoRA reset at step {self.state.global_step}")
                    if modules:
                        logger.info(f"Modules to reset: {', '.join(modules)}")
                    self._perform_lora_reset(model)

                    logger.info(f"Performing optimizer reset at step {self.state.global_step}")
                    logger.info(f"Performing optimizer reset at update_step {self.update_step}")
                    self._reset_optimizer()
                    
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    all_converged = model.module.check_all_converged()
                else:
                    all_converged = model.check_all_converged()
                
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

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        super().save_model(output_dir, _internal_call)
        
        if self.is_world_process_zero():
            base_config = {
                "_name_or_path": "roberta-base",
                "architectures": ["RobertaForSequenceClassification"],
                "finetuning_task": self.task_name,
                "num_labels": 3 if self.task_name == "mnli" else 2,
                "problem_type": "single_label_classification",
            }

            model_config = self.model.config.to_dict()
            config_dict = {**model_config, **base_config}
            
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")

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

def main():
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AROMAArguments)) 
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, AROMA_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, AROMA_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

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
    task_name = data_args.task_name.lower()
    
    logger.info(f"training_args.local_rank: {training_args.local_rank}")

    if training_args.output_dir:
        task_output_dir = os.path.join(training_args.output_dir, task_name)
        training_args.output_dir = os.path.join(task_output_dir, "output", timestamp)
        training_args.logging_dir = os.path.join(task_output_dir, "runs", timestamp)

        if training_args.local_rank in [-1, 0]:
            os.makedirs(training_args.output_dir, exist_ok=True)
            os.makedirs(training_args.logging_dir, exist_ok=True)

    training_args.batch_eval_metrics = True
    training_args.label_names = ["labels"]
    training_args.include_inputs_for_metrics = True
    training_args.fp16=True
    training_args.bf16=True


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

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    local_glue_path = "path/to/your/local/glue/.cache/huggingface/datasets/glue/"
    task_path = os.path.join(local_glue_path, data_args.task_name)
    
    if not os.path.exists(task_path):
        raise ValueError(f"Task path {task_path} does not exist. Please make sure the GLUE dataset is downloaded.")
    
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            # use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Number of Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
            logger.info(f"Label list: {label_list}")
            logger.info(f"Number of labels: {num_labels}")
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
            logger.info(f"Label list: {label_list}")

    # ###################################
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_name_or_path is None: 
        model = LlamaForSequenceClassification(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    def show_parameters(model):
        for name, param in model.named_parameters():
            num_params = param.numel()
            logger.info(f"Layer: {name:<20} | Shape: {str(param.shape):<20} | Number of parameters: {num_params:<10} | Trainable: {param.requires_grad}")

    params_before = sum(p.numel() for p in model.parameters())

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

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
            # target_modules=["q", "k", "v", "o", "wi", "wo"],
            target_modules=["query", "key", "value", "dense"],
            # target_modules=["attn", "attention", "mlp"],
            # target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
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

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    def preprocess_function(examples):        
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        
        result = tokenizer(
            *texts,
            padding="max_length" if data_args.pad_to_max_length else False,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        if "label" in examples:
            # if label_to_id is not None:
            #     # Map labels to IDs (not necessary for GLUE tasks)
            #     result["labels"] = [label_to_id[l] for l in examples["label"]]
            # else:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]

        assert "input_ids" in result, f"Missing input_ids in result: {result.keys()}"
        assert "attention_mask" in result, f"Missing attention_mask in result: {result.keys()}"
        assert "labels" in result, f"Missing labels in result: {result.keys()}"

        if random.random() < 0.001:
            logger.info("=== Sample Preprocessing Result ===")
            logger.info(f"Keys in result: {result.keys()}")
            logger.info(f"Labels shape: {len(result['labels'])}")
            logger.info(f"Sample labels: {result['labels'][:5]}")
            logger.info(f"Unique labels: {set(result['labels'])}")
            logger.info("================================")

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        preprossed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets["train"].column_names,
        )

    if training_args.do_train:
        if "train" not in preprossed_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = preprossed_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"1 Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if "validation" not in preprossed_datasets and "validation_matched" not in preprossed_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = preprossed_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"] # 用validation data 做evaluating
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in preprossed_datasets and "test_matched" not in preprossed_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = preprossed_datasets["test_matched" if data_args.task_name == "mnli" else "test"] # 用test data 做predicting
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3): 
    #         logger.info(f"2 Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction, compute_result: bool = False):
        if not compute_result:
            # logger.info("Not last eval step, returning None")
            return None
        
        if training_args.local_rank not in [-1, 0]:
            return None
    
        logger.info("Last eval step, computing metrics")
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        label_ids = p.label_ids
        
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()
        if isinstance(p.label_ids, torch.Tensor):
            label_ids = p.label_ids.detach().cpu()
        else:
            label_ids = p.label_ids
            
        preds = np.array(preds)
        label_ids = np.array(label_ids)
        
        if is_regression:
            preds = np.squeeze(preds)
            result = {
                "mse": ((preds - label_ids) ** 2).mean().item(), 
                "pearson": float(metric.compute(predictions=preds, references=label_ids)["pearson"])
            }
        else:
            preds = np.argmax(preds, axis=1)

            logger.info(f"Processed predictions: {preds[:10]}")
            logger.info(f"Unique predictions: {np.unique(preds, return_counts=True)}")
            logger.info(f"Unique labels: {np.unique(label_ids, return_counts=True)}")

            result = metric.compute(predictions=preds, references=label_ids)
            
            if data_args.task_name == "cola":
                result = {"matthews_correlation": float(result["matthews_correlation"])}
            elif data_args.task_name == "sst2":
                result = {"accuracy": float(result["accuracy"])}
            elif data_args.task_name == "mrpc":
                result = {"accuracy": float(result["accuracy"]), "f1": float(result["f1"])}
            elif data_args.task_name == "stsb":
                result = {"pearson": float(result["pearson"]), "spearmanr": float(result["spearmanr"])}
            elif data_args.task_name == "qqp":
                result = {"accuracy": float(result["accuracy"]), "f1": float(result["f1"])}
            elif data_args.task_name == "mnli":
                result = {"accuracy": float(result["accuracy"])}
            elif data_args.task_name == "qnli":
                result = {"accuracy": float(result["accuracy"])}
            elif data_args.task_name == "rte":
                result = {"accuracy": float(result["accuracy"])}
            elif data_args.task_name == "wnli":
                result = {"accuracy": float(result["accuracy"])}
            
        logger.info(f"Computed metrics: {result}")
        return result


    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if train_dataset is not None:
        logger.info("=== Training Dataset Info ===")
        logger.info(f"Number of training examples: {len(train_dataset)}")
        if 'labels' in train_dataset.features:
            labels = train_dataset['labels']
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"Training set label distribution: {dict(zip(unique, counts))}")

    if eval_dataset is not None:
        logger.info("=== Evaluation Dataset Info ===")
        logger.info(f"Number of evaluation examples: {len(eval_dataset)}")
        if 'labels' in eval_dataset.features:
            labels = eval_dataset['labels']
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"Evaluation set label distribution: {dict(zip(unique, counts))}")

    if predict_dataset is not None:
        logger.info("=== Prediction Dataset Info ===")
        logger.info(f"Number of prediction examples: {len(predict_dataset)}")
        if 'labels' in predict_dataset.features:
            labels = predict_dataset['labels']
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"Prediction set label distribution: {dict(zip(unique, counts))}")
    
    # Initialize our Trainer
    trainer = AROMATrainer(
        AROMA_args=AROMA_args,
        task_name=data_args.task_name,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        callbacks=[DetailedLoggingCallback()],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if hasattr(trainer.model, 'module'):
        base_model = trainer.model.module
    else:
        base_model = trainer.model

    if hasattr(base_model.__class__, '_get_label_names'):
        label_names = base_model.__class__._get_label_names(base_model.config)
        logger.info(f"Model expected label names: {label_names}")
    else:
        import inspect
        forward_params = inspect.signature(base_model.forward).parameters
        logger.info(f"Model forward method parameters: {list(forward_params.keys())}")

    # #########################
    # Training!
    if training_args.do_train:
        logger.info("*** Training! ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint) # Train!
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        if training_args.do_eval:
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation!
    if training_args.do_eval:
        logger.info("*** Checking evaluation dataset ***")
        logger.info(f"Eval dataset features: {eval_dataset.features}")
        sample_item = eval_dataset[0]
        logger.info(f"Sample eval item keys: {sample_item.keys()}")
        if "labels" in sample_item:
            logger.info(f"Sample labels: {sample_item['labels']}")
        else:
            logger.warning("No 'labels' found in eval dataset!")
            available_keys = sample_item.keys()
            logger.info(f"Available keys: {available_keys}")

        logger.info("*** Evaluate! ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = preprossed_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(
                eval_dataset=eval_dataset,
                metric_key_prefix="eval",
            ) # Evaluate!

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    # Prediction!
    if training_args.do_predict:
        logger.info("*** Predict! ***")
        logger.info(f"Model num_labels: {model.config.num_labels}")
        logger.info(f"Task name: {data_args.task_name}")
        logger.info(f"Total examples in predict dataset: {len(predict_dataset)}")
        
        if data_args.task_name != "mrpc" and 'labels' in predict_dataset.features:
            logger.info("Removing labels for test set as they are placeholder values (-1)")
            predict_dataset = predict_dataset.remove_columns(['labels'])
        
        logger.info(f"Dataset features: {predict_dataset.features}")
        
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            predict_output = trainer.predict(predict_dataset, metric_key_prefix="predict")
            predictions = predict_output.predictions
            
            if training_args.local_rank in [-1, 0]:
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                
                predictions = np.array(predictions)
                
                if not is_regression:
                    predictions = np.argmax(predictions, axis=1)
                
                logger.info(f"\n=== Prediction Results for {task} ===")
                logger.info(f"Predictions shape: {predictions.shape}")
                logger.info(f"Sample predictions: {predictions[:10]}")
                logger.info(f"Prediction distribution: {np.unique(predictions, return_counts=True)}")

                output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[int(item)]
                            writer.write(f"{index}\t{item}\n")
                
                logger.info(f"Predictions saved to {output_predict_file}")
                
                metrics = {}
                max_predict_samples = (
                    data_args.max_predict_samples if data_args.max_predict_samples is not None 
                    else len(predict_dataset)
                )
                metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
                
                trainer.log_metrics("predict", metrics)
                trainer.save_metrics("predict", metrics)

        logger.info("Prediction completed!")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    logger.info("*** AROMA Training Finished! ***")

    end_time = datetime.now()
    duration = end_time - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
