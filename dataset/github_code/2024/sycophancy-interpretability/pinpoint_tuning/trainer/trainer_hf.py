#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:19
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:23
Description        : Huggingface Trainer for DeepSpeed.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import logging
import math
import os
import shutil
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional

import safetensors
import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer
from transformers.data.data_collator import DataCollatorMixin
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TRAINING_ARGS_NAME, PREFIX_CHECKPOINT_DIR
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME

from trainer.callbacks import LoggerCallback, SaveCallback, ProfilerCallback
from utils import logging_rank, delete_zero_hf
from utils import SUPPORTED_MODEL_CLASSES

logger = logging.getLogger('DeepSpeed')


class HuggingfaceTrainer(Trainer):
    """
    Huggingface Trainer for DeepSpeed.
    """

    def __init__(self, args: Namespace, model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer, train_dataset: Dataset,
                 data_collator: DataCollatorMixin) -> None:
        """Initialize Trainer.

        Parameters
        ----------
        args : Namespace
            Arguments from argparse.
        model : PreTrainedModel
            Model to be trained.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        train_dataset : Dataset
            Training dataset.
        data_collator : DataCollatorMixin
            Data collator used to collate data.
        """

        super().__init__(model=model,
                         tokenizer=tokenizer,
                         args=args,
                         train_dataset=train_dataset,
                         data_collator=data_collator)

        model.config.use_cache = False

        self.args = args
        self.previous_output_dir = None
        self.latest_previous_output_dir = None

        self._print_trainning_info()
        self._setup_callbacks()

    def _print_trainning_info(self) -> None:
        """Print training information.
        """
        train_dataset_size = len(self.train_dataset)

        batch_size = (self.args.per_device_train_batch_size *
                      self.args.world_size *
                      self.args.gradient_accumulation_steps)

        # Training steps
        total_steps = math.ceil(
            train_dataset_size / batch_size) * self.args.num_train_epochs
        warmup_steps = (int(total_steps *
                            self.args.warmup_ratio) if self.args.warmup_ratio
                        > 0.0 else self.args.warmup_steps)

        num_gpus = torch.cuda.device_count()
        global_batch_size = (self.args.train_batch_size *
                             self.args.gradient_accumulation_steps *
                             self.args.world_size)

        logging_rank(
            f" * num_gpus = {num_gpus}, train_dataset_size = {train_dataset_size}, batch_size = {batch_size}"
        )
        logging_rank(
            f" * total_steps = {total_steps}, warmup_steps = {warmup_steps}, eval_steps = {self.args.eval_steps}, save_steps = {self.args.save_steps}"
        )
        logging_rank(
            f" * global_batch_size = {global_batch_size}, train_batch_size = {self.args.train_batch_size}, accumulation_steps = {self.args.gradient_accumulation_steps}"
        )

    def _setup_callbacks(self) -> None:
        """Setup callbacks.
        """

        # Add logging and save callbacks
        self.add_callback(LoggerCallback())
        self.add_callback(SaveCallback())

        # Add profiling callback if needed
        if self.args.profile:
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=5,
                                                 warmup=1,
                                                 active=1,
                                                 repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.args.logging_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True)

            self.add_callback(ProfilerCallback(prof))

    def train(self,
              resume_from_checkpoint: str = None,
              trial: Dict[str, Any] = None,
              ignore_keys_for_eval: List[str] = None,
              **kwargs) -> None:
        """Start training.

        Parameters
        ----------
        resume_from_checkpoint : str, optional
            Path to resume training, by default None
        trial : Dict[str, Any], optional
            Trial, by default None
        ignore_keys_for_eval : List[str], optional
            Ignore keys when evalating, by default None
        """

        if resume_from_checkpoint is not None:
            self.previous_output_dir = resume_from_checkpoint
            checkpoint_name = os.path.basename(resume_from_checkpoint)
            try:
                checkpoint_step = int(checkpoint_name.split("-")[-1])
                if checkpoint_step % self.args.save_steps != 0:
                    self.latest_previous_output_dir = resume_from_checkpoint
            except Exception as e:
                logging_rank(f"get checkpoint_step failed: {checkpoint_name}")

        super().train(resume_from_checkpoint=resume_from_checkpoint,
                      trial=trial,
                      ignore_keys_for_eval=ignore_keys_for_eval,
                      **kwargs)

    def save_model(self,
                   output_dir: Optional[str] = None,
                   _internal_call: bool = False,
                   keep_zero: bool = False) -> None:
        """Save model to output_dir.

        Parameters
        ----------
        output_dir : Optional[str], optional
            Output directory, by default None
        _internal_call : bool, optional
            _internal_call, by default False
        keep_zero : bool, optional
            Whether to keep zero states when saving, by default False
        """

        super().save_model(output_dir=output_dir,
                           _internal_call=_internal_call)

        # Delete Zero states in previous checkpoint since we won't resume from it
        if not self.args.save_only_model and self.previous_output_dir and keep_zero == False:
            delete_zero_hf(self.previous_output_dir)

        # Update previous_output_dir
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        self.previous_output_dir = os.path.join(self.args.output_dir,
                                                checkpoint_folder)

        # Update latest_previous_output_dir (which may be save due to training time interval)
        if hasattr(self.control, "latest") and self.control.latest:
            if self.latest_previous_output_dir is not None:
                if torch.distributed.get_rank() == 0:
                    logger.info(
                        f" ! Remove the previous latest checkpoint {self.latest_previous_output_dir}"
                    )
                    shutil.rmtree(self.latest_previous_output_dir)
            self.latest_previous_output_dir = os.path.join(
                self.args.output_dir, checkpoint_folder)

        self.control.last_time = time.time()
        torch.distributed.barrier()

    def _save(self, output_dir: Optional[str] = None, state_dict: Any = None) -> None:
        """Save model to output_dir.

        Parameters
        ----------
        output_dir : Optional[str], optional
            Output directory, by default None
        state_dict : Any, optional
            state_dict to be saved, by default None
        """

        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f" - Saving model checkpoint to {output_dir}")

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, SUPPORTED_MODEL_CLASSES):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), SUPPORTED_MODEL_CLASSES):
                unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors)
            else:
                logger.info(
                    " @ Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir,
                                                 SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict,
                               os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if self.args.peft_type == "prompt_tuning":
                #self.model.prompt_encoder["default"].embedding.weight.data = state_dict['prompt_encoder.default.embedding.weight']
                self.model.prompt_encoder_weight = state_dict[
                    'prompt_encoder.default.embedding.weight']

            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def build_trainer_hf(args: Namespace, model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizer, train_dataset: Dataset,
                     data_collater: DataCollatorMixin) -> Trainer:
    """Huggingface Trainer.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    model : PreTrainedModel
        Model to be trained.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to tokenize data.
    train_dataset : Dataset
        Training dataset.
    data_collater : DataCollatorMixin
        Data collater used to collate data.

    Returns
    -------
    Trainer
        Huggingface Trainer.
    """

    trainer = HuggingfaceTrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collater,
    )

    return trainer
