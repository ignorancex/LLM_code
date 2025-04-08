import copy
import math
import os
import shutil
import sys
import time
import uuid
from argparse import Namespace
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Optional, List, Callable, Dict, Union, Any, Tuple

import evaluate
import numpy as np
import torch
import webdataset as wds
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, is_torch_tpu_available, get_scheduler, IntervalStrategy, TrainerCallback, \
    TrainerControl, TrainerState, TrainingArguments
from transformers.debug_utils import DebugOption
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.integrations import WandbCallback
from transformers.trainer_pt_utils import find_batch_size, IterableDatasetShard, nested_numpify, nested_concat
from transformers.trainer_utils import has_length, EvalPrediction, denumpify_detensorize, PredictionOutput, \
    speed_metrics
from transformers.utils import logging

from evaluation.cider.cider_huggingface import Cider
from utils import postprocess_text, CustomCosineScheduler, CustomScheduler, \
    is_main_process, get_world_size, load_log_history_from_file, CustomEvalLoopOutput, save_predictions, \
    CustomEvalPrediction

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

is_debug = 'pydevd' in sys.modules
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, test_dataset: Optional[Dataset] = None, cider: Optional[Cider] = None, custom_args: Optional[Namespace] = None,
                 collate_fn: Optional[Callable] = None, **kwargs):
        self.test_dataset = test_dataset
        self.cider = cider
        self.custom_args = custom_args
        # self.process_sample = process_sample
        self.collate_fn = collate_fn

        super().__init__(**kwargs)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        inputs["output_dir"] = self.args.output_dir

        return inputs

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.custom_args.custom_lr_scheduler == 'CustomCosineScheduler':
                logger.info('Using a CustomCosineScheduler as scheduler.')
                self.lr_scheduler = CustomCosineScheduler(optimizer=self.optimizer if optimizer is None else optimizer,
                                                          warmup_steps=self.args.warmup_steps,
                                                          learning_rate=self.args.learning_rate,
                                                          steps_min=self.custom_args.steps_min,
                                                          lr_min=self.custom_args.lr_min)
            elif self.custom_args.custom_lr_scheduler == 'CustomScheduler':
                logger.info('Using a CustomScheduler as scheduler.')
                self.lr_scheduler = CustomScheduler(optimizer=self.optimizer if optimizer is None else optimizer,
                                                    warmup_steps=self.args.warmup_steps,
                                                    start_decreasing_steps=self.custom_args.start_decreasing_steps,
                                                    learning_rate=self.args.learning_rate,
                                                    steps_min=self.custom_args.steps_min,
                                                    lr_min=self.custom_args.lr_min)
            elif self.custom_args.custom_lr_scheduler == 'TransformerScheduler':
                logger.info('Using a LambdaLR (Transformer like scheduler) as scheduler.')

                def lr_scheduler(optim):
                    def lambda_lr(s):
                        d_model = self.model.decoder.config.n_embd
                        warm_up = self.args.warmup_steps
                        lr_multiplier = self.custom_args.lr_multiplier
                        s += 1
                        return (d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5) * lr_multiplier

                    return LambdaLR(optim, lambda_lr)

                self.lr_scheduler = lr_scheduler(self.optimizer if optimizer is None else optimizer)
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
        return self.lr_scheduler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        train_dataloader = wds.WebLoader(
            train_dataset, 
            batch_size=None,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers, 
            pin_memory=self.args.dataloader_pin_memory,
            )

        return train_dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[wds.DataPipeline] = None) -> wds.WebLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = wds.WebLoader(
            eval_dataset,
            batch_size=None,
            collate_fn=self.data_collator,
            # num_workers=self.args.dataloader_num_workers,
            num_workers=0,  # awful workaround!
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            )

        return eval_dataloader
    
    def get_test_dataloader(self, test_dataset: Optional[wds.DataPipeline] = None) -> wds.WebLoader:
        if test_dataset is None and self.test_dataset is None:
            raise ValueError("Trainer: testuation requires an test_dataset.")
        test_dataset = test_dataset if test_dataset is not None else self.test_dataset
        test_dataloader = wds.WebLoader(
            test_dataset,
            batch_size=None,
            collate_fn=self.data_collator,
            # num_workers=self.args.dataloader_num_workers,
            num_workers=0,  # awful workaround!
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            )

        return test_dataloader

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval).metrics
            test_metrics = self.predict(ignore_keys=ignore_keys_for_eval).metrics
            metrics = dict()
            metrics.update(eval_metrics)
            metrics.update(test_metrics)
            self.log(metrics)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def predict(
        self,
        test_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        if hasattr(self.state,
                   'best_model_checkpoint') and self.state.best_model_checkpoint is not None and self.args.load_best_model_at_end:
            output.metrics['best_model_checkpoint'] = self.state.best_model_checkpoint

        save_predictions(output, self.tokenizer, trainer=self, split=metric_key_prefix)

        return output

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs['output_dir'] = self.args.output_dir
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    # Workaround to back propagate the gradient
    @torch.enable_grad()
    def compute_SCST_reward(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        max_length = self.args.generation_max_length
        num_beams = self.args.generation_num_beams
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": max_length if max_length is not None else self.model.config.max_length,
            "num_beams": num_beams if num_beams is not None else self.model.config.num_beams,
            'num_return_sequences': num_beams if num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "output_scores": True,
            "return_dict_in_generate": True
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if inputs.get('main_input_name'):
            main_input_name = inputs.get('main_input_name')
            model.main_input_name = main_input_name
            if hasattr(model, 'encoder'):
                model.encoder.main_input_name = main_input_name
        elif hasattr(model, "encoder") and model.encoder.main_input_name != model.main_input_name:
            main_input_name = model.encoder.main_input_name
        else:
            main_input_name = model.main_input_name

        generation_inputs = inputs[main_input_name]

        generation_inputs.requires_grad = True
        output = model.generate_with_backpropagation(
            generation_inputs,
            **gen_kwargs,
        )
        generated_tokens = output['sequences']
        log_probs = model.compute_transition_beam_scores(output.sequences, output.scores, output.beam_indices, self.tokenizer.eos_token_id)
        log_probs = log_probs.view(labels.shape[0], num_beams, -1)

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels[labels == -100] = self.tokenizer.pad_token_id
        references = [self.tokenizer.batch_decode(label, skip_special_tokens=True) for label in labels]
       
        predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        processed_predictions, processed_references = postprocess_text(predictions, references)
        processed_references = list(chain(*([ref,] * num_beams for ref in processed_references)))

        cache_dir = None
        cider = evaluate.load('evaluation/cider/cider_huggingface.py', experiment_id=str(uuid.uuid4()), cache_dir=cache_dir)
        cider.doc_frequency = self.cider.doc_frequency
        cider.ref_len = self.cider.ref_len

        reward = cider.compute(predictions=processed_predictions, references=processed_references,
                               return_scores=True)['cider'].astype(np.float32)
        reward = torch.from_numpy(reward).to(labels.device).view(labels.shape[0], num_beams)
        reward_baseline = torch.mean(reward, -1, keepdim=True)
        loss = -(torch.sum(log_probs, -1) / torch.sum(log_probs != 0, -1)) * (reward - reward_baseline)
        loss = loss.mean()

        return (loss, generated_tokens) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Union[PredictionOutput, CustomEvalLoopOutput]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            dataloader=eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        if hasattr(self.state, 'best_model_checkpoint') and self.state.best_model_checkpoint is not None and self.args.load_best_model_at_end:
            output.metrics['best_model_checkpoint'] = self.state.best_model_checkpoint

        if self.args.do_train and not (self.args.do_eval or self.args.do_predict):
            save_predictions(output, self.tokenizer, trainer=self, split=metric_key_prefix)

        return output

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        gen_kwargs.update({'__key__': inputs.get('__key__')})
        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return loss, generated_tokens, labels

    def evaluation_loop(
        self,
        dataloader: wds.WebLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> CustomEvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        keys_host = list()
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_keys = list()
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        pad_token_id = self.model.config.pad_token_id
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            inputs['output_dir'] = self.args.output_dir
            if is_debug and step > 5:
                break
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            keys = copy.copy(inputs['__key__'])
            labels = copy.copy(inputs['labels'])
            inputs['labels'] = inputs['labels'][:, 0, :]
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].type(model.dtype)
            if 'vision_encoder_outputs' in inputs:
                inputs['vision_encoder_outputs'] = inputs['vision_encoder_outputs'].type(model.dtype)
            if inputs.get('main_input_name'):
                main_input_name = inputs.get('main_input_name')
                model.main_input_name = main_input_name
                if hasattr(model, 'encoder'):
                    model.encoder.main_input_name = main_input_name
            loss, logits, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = inputs["input_ids"] if args.include_inputs_for_metrics else None
            labels = labels.to(model.device)
            # pad anc concat methods work only on the 1st dimension, workaround: double swapaxes(1, -1)
            labels = labels.swapaxes(1, -1)
            labels = labels.contiguous()

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if keys is not None:
                gathered_keys = [None for _ in range(get_world_size())]
                if torch.distributed.is_initialized():
                    torch.distributed.all_gather_object(gathered_keys, keys)
                else:
                    gathered_keys = keys
                keys_host.extend(kk for k in gathered_keys for kk in k)
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels, pad_index=pad_token_id)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels,
                                                                               padding_index=pad_token_id)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=pad_token_id)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

            if (step % self.args.logging_steps == 0 or is_debug) and is_main_process():
                world_size_observed_num_examples = observed_num_examples * get_world_size()
                logger.info(f'{step=}, {world_size_observed_num_examples=}')

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if keys_host:
            all_keys = keys_host
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=pad_token_id)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(
                    CustomEvalPrediction(predictions=all_preds, keys=all_keys, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return CustomEvalLoopOutput(predictions=all_preds, keys=all_keys, label_ids=all_labels, metrics=metrics,
                                    num_samples=num_samples)


class CustomWandbCallback(WandbCallback):
    def __init__(self, custom_args):
        self.custom_args = custom_args

        super().__init__()

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also override the following environment
        variables:

        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training. Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to disable gradient logging or `"all"` to
                log gradients and parameters.
            WANDB_PROJECT (`str`, *optional*, defaults to `"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (`bool`, *optional*, defaults to `False`):
                Whether or not to disable wandb entirely. Set *WANDB_DISABLED=true* to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if not state.is_world_process_zero:
            return

        logger.info(
            'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
        )
        combined_dict = {**args.to_sanitized_dict()}

        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}
        trial_name = state.trial_name
        init_args = {}
        custom_entity = self.custom_args.get('entity')
        if custom_entity:
            init_args['entity'] = custom_entity
        if trial_name is not None:
            run_name = trial_name
            init_args["group"] = args.run_name
        else:
            run_name = args.run_name

        wandb_dir = Path('wandb/')
        if not wandb_dir.exists():
            wandb_dir = None

        # get wandb id from file and eventually resume the run
        wandb_id_file = Path(args.output_dir).joinpath('wandb_id')
        if wandb_id_file.exists() and not args.overwrite_output_dir:
            wandb_id = wandb_id_file.read_text().strip()
            logger.info(f'Resuming wandb run with {wandb_id=}')
            init_args['id'] = wandb_id
            init_args['resume'] = 'allow'

        os.environ["WANDB__SERVICE_WAIT"] = "300"

        if self._wandb.run is None:
            self._wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                name=run_name,
                dir=wandb_dir,
                **init_args,
            )
        # add config parameters (run may have been created manually)
        self._wandb.config.update(combined_dict, allow_val_change=True)
        self._wandb.config.update({'custom_args': self.custom_args}, allow_val_change=True)

        # save wandb id to file
        if not wandb_id_file.exists() and is_main_process():
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(wandb_id_file, mode='w') as f:
                f.write(self._wandb.run.id)

        # define default x-axis (for latest wandb versions)
        if getattr(self._wandb, "define_metric", None):
            self._wandb.define_metric("train/global_step")
            self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

        # keep track of model topology and gradients, unsupported on TPU
        if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
            self._wandb.watch(
                model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
            )


class CustomDeleteCheckpointsCallback(TrainerCallback):
    def __init__(self, save_limit=5, metric='cider'):
        self.save_limit = save_limit
        self.metric = metric.lower()

    def get_metrics_from_log_history(self, log_history):
        # is_eval_or_test_log = lambda k: 'eval' in k or 'test' in k
        is_eval_or_test_log = lambda k: self.metric in k
        eval_or_test_logs = list(filter(lambda s: any(is_eval_or_test_log(key) for key in s.keys()), log_history))

        metrics = defaultdict(dict)
        for log in eval_or_test_logs:
            step = log.get('step')
            metrics[step] = log

        return metrics

    @staticmethod
    def select_best_eval_test_steps(metrics, metric='cider'):
        best_eval = {step: logs[f'eval_{metric}'] for step, logs in metrics.items() if f'eval_{metric}' in logs.keys()}
        best_test = {step: logs[f'test_{metric}'] for step, logs in metrics.items() if f'test_{metric}' in logs.keys()}
        # save 2 steps for each split
        best_k_eval_steps = sorted(best_eval, key=best_eval.get, reverse=True)[:2]
        best_k_test_steps = sorted(best_test, key=best_test.get, reverse=True)[:2]
        best_k_steps = set(best_k_eval_steps + best_k_test_steps)

        return best_k_steps

    def delete_checkpoints(self, output_dir, checkpoint_prefix='checkpoint', dry_run=False):
        log_history = load_log_history_from_file(output_dir)
        if not log_history:
            logger.error(f'log_history not found! Skipping experiment {output_dir}')
            return

        metrics = self.get_metrics_from_log_history(log_history)
        if not metrics:
            logger.error(f'metrics not found! Skipping experiment {output_dir}')
            return

        all_steps = list(metrics.keys())
        best_steps = self.select_best_eval_test_steps(metrics)
        all_checkpoints = {step: Path(output_dir).joinpath(f"{checkpoint_prefix}-{step}") for step in all_steps}
        steps_to_be_deleted = set(sorted(all_steps)[:-self.save_limit])
        steps_to_be_deleted -= best_steps

        for step in sorted(steps_to_be_deleted):
            checkpoint = all_checkpoints[step]
            if not dry_run and checkpoint.exists() and checkpoint.is_dir():
                logger.warning(f"Deleting older checkpoint [{checkpoint}]...")
                try:
                    shutil.rmtree(checkpoint)
                except OSError as e:
                    logger.error(e)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero or self.save_limit == -1:
            return

        should_delete = args.eval_steps > 0 and state.global_step > 1 and args.eval_delay <= state.global_step and \
                        args.evaluation_strategy == IntervalStrategy.STEPS and \
                        (state.global_step - 1) % args.eval_steps == 0
        if not should_delete:
            return

        self.delete_checkpoints(args.output_dir)


class GlobalStepCallback(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        global_step = state.global_step

        # Look for "global_step" attributes in model and children
        def update_global_step(m: torch.nn.Module):
            if hasattr(m, 'global_step'):
                setattr(m, 'global_step', global_step)

        model.apply(update_global_step)
