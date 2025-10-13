import datetime
import json
import logging
import os
import time
from pathlib import Path
import functools

import torch
import torch.distributed as dist
import webdataset as wds
from common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from common.registry import registry
from common.utils import is_url
from data.data_utils import concat_datasets, reorg_datasets_by_split
from data.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, always_wrap_policy
from torch.distributed.fsdp import ShardingStrategy

from optimizer import *
import copy
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import pandas as pd
from runners.runner_base import RunnerBase


@registry.register_runner("runner_robust_ft")
class RunnerRobustFT(RunnerBase):
    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)
    
    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """

        print("Device Check", self._model.device != self.device)
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                # 1 FSDP
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu]
                    )
                    print(self._wrapped_model)
                    # my_auto_wrap_policy = functools.partial(
                    #     size_based_auto_wrap_policy, min_num_params=100
                    # )
                    # self._wrapped_model = FSDP(
                    #     self._model, device_id=self.config.run_cfg.gpu,
                    #     cpu_offload=CPUOffload(offload_params=False),
                    #     auto_wrap_policy=my_auto_wrap_policy,  # Wrap layers with at least 1M parameters
                    #     # auto_wrap_policy=always_wrap_policy,  # Wrap all layers
                    #     use_orig_params=True,
                    #     sharding_strategy=ShardingStrategy.FULL_SHARD,
                    # )
                    # print(self._wrapped_model)
                    
            else:
                self._wrapped_model = self._model

        return self._wrapped_model
    
    @property
    def optimizer(self):
        if self._optimizer is None:
            trainable_params = [p for p in self._model.parameters() if p.requires_grad]  # TODO
            
            # for name, param in self._model.named_parameters():
            #     print(f"{name} requires_grad: {param.requires_grad}")
            
            opt = self.config.run_cfg.get("opt", "adam")
            # lr_scale = self.config.run_cfg.get("lr_layer_decay", 1)
            weight_decay = self.config.run_cfg.get("weight_decay", 0.05)
            lr=float(self.config.run_cfg.init_lr)
            # beta2 = self.config.run_cfg.get("beta2", 0.999)
            
            use_lora = self.config.model_cfg.use_lora
            if use_lora == 1:
                use_lora = True
            else:
                use_lora = False
            
            if opt == "sgdp":
                # Initalize optimizer parameters
                optimizer_params = {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": 0.9,
                    "nesterov": True,
                    "k": 1, 
                    #"exclude_set": {'module.head.weight','module.head.bias'}
                } 
                # Cache pre-trained model weights 
                params_to_opt = [x[1] for x in self._model.named_parameters() if x[1].requires_grad]
                params_to_opt_name = [x[0] for x in self._model.named_parameters() if x[1].requires_grad]
                params_anchor = copy.deepcopy(params_to_opt)
                param_group = [{'params':params_to_opt,
                                'pre': params_anchor, 
                                'name': params_to_opt_name}]
                self._optimizer = SGDP(param_group,**optimizer_params)

            elif opt == "adamp":
                # Initalize optimizer parameters
                optimizer_params = {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "k": int(self.config.run_cfg.get("adamp_k", 1)),
                    "use_lora": use_lora,
                    #"exclude_set": {'module.head.weight','module.head.bias'}
                } 

                # Cache pre-trained model weights 
                params_to_opt_name = [x[0] for x in self._model.named_parameters() if x[1].requires_grad]
                params_to_opt = [x[1] for x in self._model.named_parameters() if x[1].requires_grad]
                
                if use_lora:
                    param_group = [{'params':params_to_opt, 
                                    'name': params_to_opt_name}]
                else:
                    params_anchor = copy.deepcopy(params_to_opt)
                    # put params_anchor to cpu
                    for p in params_anchor:
                        p.data = p.data.cpu()
                    param_group = [{'params':params_to_opt,
                                    'pre': params_anchor, 
                                    'name': params_to_opt_name}]
                self._optimizer = AdamP(param_group,**optimizer_params)
            
            elif opt == "adamh":
                optimizer_params = {
                    "lr": lr,
                    "weight_decay": weight_decay, #args.weight_decay, 1
                    "use_lora": use_lora,
                    "norm_type": "l2",
                } 
                params_to_opt = [x[1] for x in self._model.named_parameters() if x[1].requires_grad]
                if use_lora:
                    param_group = [{'params':params_to_opt}]
                else:
                    params_anchor = copy.deepcopy(params_to_opt)
                    # put params_anchor to cpu
                    for p in params_anchor:
                        p.data = p.data.cpu()
                    param_group = [{'params':params_to_opt,
                                    'pre': params_anchor}]
                self._optimizer = AdamH(param_group,**optimizer_params)
            
            elif opt == "adam":
                self._optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
            else:
                optimizer_params = {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": 0.9,
                    "nesterov": True,
                }   
                self._optimizer = torch.optim.SGD(trainable_params, **optimizer_params)
            
            print("optimizer: ", self._optimizer)
        
        return self._optimizer
    
    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        print("url ", url_or_filename)
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        use_lora = int(self.config.model_cfg.get("use_lora", 0))
        lora_alpha = int(self.config.model_cfg.get("lora_alpha", 16))
        lora_rank = int(self.config.model_cfg.get("lora_rank", 4))
        target_modules = self.config.model_cfg.get("target_modules", "q_proj k_proj v_proj o_proj").split()

        if use_lora == 1:
            lora_config = LoraConfig(
                r=lora_rank,  # 4
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=target_modules,
            )
            
            logging.info(lora_config)
            
            self._model = get_peft_model(self._model, lora_config)
            logging.info(self._model.print_trainable_parameters())

        for key in list(state_dict.keys()):
            start_key = "base_model.model.model."
            start_key_2 = "model."
            if key.startswith(start_key):
                state_dict[key[len(start_key):]] = state_dict.pop(key)
            elif key.startswith(start_key_2):
                state_dict[key[len(start_key_2):]] = state_dict.pop(key)
        # Load the current state_dict of the model
        current_state_dict = self.unwrap_dist_model(self.model).state_dict()
        for key in state_dict.keys():
            assert key in current_state_dict, f"key {key} not in current_state_dict"
            current_state_dict[key] = state_dict[key]

        self.unwrap_dist_model(self.model).load_state_dict(current_state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))