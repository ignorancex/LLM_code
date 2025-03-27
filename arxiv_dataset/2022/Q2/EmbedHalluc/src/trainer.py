import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data.dataset import Dataset


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(opt.img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], * self.opt.img_shape)
        return img


class EmbeddingDataset(Dataset):
    def __init__(self, opt):
        with open('cache_embeddings/%s_%s_%s_%s.json'%(opt.model_name_or_path, opt.few_shot_type, opt.task_name, opt.seed)) as json_file:
            data = json.load(json_file)   ####{'0':[emb, emb, ...], '1':[emb, emb, ...]}
        self.embeddings = []
        self.labels = []
        self.mean_embs = []
        for i in data:
            mean_emb = torch.zeros(torch.tensor(data['0'][0]).shape)
            for emb_i in range(len(data[i])):
                mean_emb += torch.tensor(data[i][emb_i])
                self.embeddings.append(torch.tensor(data[i][emb_i]).squeeze(0))
                self.labels.append(torch.tensor(int(i)).long())
            mean_emb = mean_emb.mean(0)    #### mean embeddings for class "i"
            self.mean_embs.append(mean_emb)

        self.num_classes = len(data)
        self.data_len = len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index].float(), self.labels[index], self.mean_embs[int(self.labels[index])]

    def __len__(self):
        return self.data_len


from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
import torch.nn.functional as F

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T=0.4):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2) / y_s.shape[0]
        return loss

########## The following part is copied from Transformers' trainer (3.4.0) ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import transformers
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)
from transformers.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_tpu_sampler,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging

from tqdm import tqdm, trange

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))

class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )


    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        ###############
        params = {}
        for n, p in self.model.named_parameters():
            if self.args.fix_layers > 0:
                if 'encoder.layer' in n:
                    try:
                        layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                    except:
                        print(n)
                        raise Exception("")
                    if layer_num >= self.args.fix_layers:
                        print('yes', n)
                        params[n] = p
                    else:
                        print('no ', n)
                elif 'embeddings' in n:
                    print('no ', n)
                else:
                    print('yes', n)
                    params[n] = p
            else:
                params[n] = p
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        emb_optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.emb_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        emb_scheduler = get_linear_schedule_with_warmup(
            emb_optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        from argparse import Namespace
        opt = Namespace()
        opt.model_name_or_path = self.args.model_name_or_path
        opt.task_name = self.args.task_name
        opt.seed = self.args.seed
        opt.few_shot_type = self.args.few_shot_type
        embeddings_set = EmbeddingDataset(opt)
        emb, lbl, _ = embeddings_set.__getitem__(0)
        opt.n_classes=embeddings_set.num_classes
        opt.img_shape = emb.shape


        def loopy(dataloader):
            while True:
                for x in iter(dataloader): yield x

        embed_dataloader = DataLoader(embeddings_set, batch_size=self.args.emb_batch_size, shuffle=True)
        embed_iter = loopy(embed_dataloader)
        ##### generator
        opt.latent_dim = 100
        generator = Generator(opt)
        ckpt = torch.load('cache_generators/%s_%s_%s_%s.pth'%(opt.model_name_or_path, opt.few_shot_type, opt.task_name, opt.seed))
        generator.load_state_dict(ckpt)
        generator = generator.cuda()

        #########################   Load Teacher Model   ###########################
        if self.args.model_args.few_shot_type == 'finetune':
            if self.args.config.model_type == 'roberta':
                model_GEN0 = AutoModelForSequenceClassification.from_pretrained(self.args.model_args.model_name_or_path,
                                                                                from_tf=False,
                                                                                config=self.args.config,
                                                                                cache_dir=self.args.model_args.cache_dir)
            elif self.args.config.model_type == 'bert':
                model_GEN0 = AutoModelForSequenceClassification.from_config(self.args.config)
                model_GEN0.resize_token_embeddings(len(self.model.tokenizer))
                resize_token_type_embeddings(model_GEN0, new_num_types=10, random_segment=self.args.model_args.random_segment)
        else:
            if self.args.config.model_type == 'roberta':
                model_GEN0 = RobertaForPromptFinetuning.from_pretrained(self.args.model_args.model_name_or_path,
                                                                        from_tf=False,
                                                                        config=self.args.config,
                                                                        cache_dir=self.args.model_args.cache_dir)

            else:
                model_GEN0 = BertForPromptFinetuning(self.args.config)
                model_GEN0.resize_token_embeddings(len(self.model.tokenizer))
                resize_token_type_embeddings(model_GEN0, new_num_types=10, random_segment=self.args.model_args.random_segment)

        model_GEN0.model_args = self.args.model_args
        model_GEN0.data_args = self.args.data_args
        model_GEN0.label_word_list = self.args.label_word_list

        ckpt = torch.load(os.path.join('./../LM-IMG/saved_models', f'%s_%s_%s_%s.pth'%(self.args.model_name_or_path, self.args.few_shot_type, self.args.task_name, self.args.seed)))
        model_GEN0.load_state_dict(ckpt, strict=True)
        model_GEN0 = model_GEN0.cuda()
        print("********************* GEN0 MODEL LOADED *********************")

        kd_criterion = DistillKL(T=self.args.kd_temperature).cuda()

        ####################################################

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                ###########################
                embs, lbls, mean_embs = next(embed_iter)
                embs, lbls, mean_embs = embs.cuda(), lbls.cuda(), mean_embs.cuda()
                z = torch.cuda.FloatTensor(np.random.normal(0, 1, (embs.shape[0], opt.latent_dim)))
                with torch.no_grad():
                    sampled_embs = generator(z, lbls)
                    sampled_embs = sampled_embs.detach()
                inputs_embeds = (sampled_embs + mean_embs) / 2   ##### /2 is added   [B, 128, 1024]

                if self.args.few_shot_type == 'finetune':    #### [CLS] xxx xxx xxx xxx
                    output = model.forward(inputs_embeds=inputs_embeds, labels=lbls)
                    if self.args.lbl_calib is True:
                        with torch.no_grad():
                            output_GEN0 = model_GEN0.forward(inputs_embeds=inputs_embeds, labels=lbls)
                            pseudo_labels = output_GEN0[1].detach()
                else:    #### prompt based language learner   #### [CLS] xxx xxx xxx xxx [MASK] xxx [SEP]
                                                              #### [CLS] xxx xxx [MAKS] xxx xxx xxx [SEP]
                    # inputs_embeds.shape == [B, 128, 1024]   ## [CLS] xxx xxx .... [MASK]
                    if self.args.model_type == 'roberta':
                        mask_input_id = [50264]
                        mask_embedding = model.roberta.embeddings(
                                input_ids=torch.as_tensor([mask_input_id]).cuda(),   ### 50 is (h * w + 1), where h=w=7
                                position_ids=None,
                                token_type_ids=None,
                                inputs_embeds=None
                            )   ### [1, 1, 1024]
                    else:
                        mask_input_id = [103]
                        mask_embedding = model.bert.embeddings(
                                input_ids=torch.as_tensor([mask_input_id]).cuda(),   ### 50 is (h * w + 1), where h=w=7
                                position_ids=None,
                                token_type_ids=None,
                                inputs_embeds=None
                            )   ### [1, 1, 1024]

                    mask_pos = torch.randint(low=1, high=inputs_embeds.shape[1]-2, size=(inputs_embeds.shape[0],1))
                    for i in range(len(mask_pos)):
                        inputs_embeds[i, mask_pos[i][0], :] = mask_embedding
                    # raise NotImplementedError
                    ##### continue
                    output = model(inputs_embeds=inputs_embeds, mask_pos=mask_pos, labels=lbls)
                    if self.args.lbl_calib is True:
                        with torch.no_grad():
                            output_GEN0 = model_GEN0(inputs_embeds=inputs_embeds, mask_pos=mask_pos, labels=lbls)
                            pseudo_labels = output_GEN0[1].detach()

                if self.args.lbl_calib is False:
                    auxiliary_loss = output[0]    ### output[0]: loss, output[1]: logits, output[2]: hidden_states, output[3]: attentions
                    model.zero_grad()
                    auxiliary_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    emb_optimizer.step()
                    emb_scheduler.step()
                    model.zero_grad()
                else:
                    output_logits = output[1]
                    kd_loss = kd_criterion(output_logits, pseudo_labels)
                    model.zero_grad()
                    kd_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    emb_optimizer.step()
                    emb_scheduler.step()
                    model.zero_grad()
                ###########################

                tr_loss += self.training_step(model, inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16:
                        norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if transformers.is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["norm"] = norm.item()
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    # ----------------------------------------------------------------------
                    # BEGIN CHANGES.
                    # ----------------------------------------------------------------------

                    metrics = None
                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        output = self.evaluate()
                        metrics = output.metrics
                        objective = self.dev_objective(metrics)
                        if objective > self.objective:
                            logger.info("Best dev result: {}".format(objective))
                            self.objective = objective
                            self.save_model(self.args.output_dir)

                    # ----------------------------------------------------------------------
                    # END CHANGES.
                    # ----------------------------------------------------------------------


                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
