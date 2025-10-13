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

from collections import defaultdict
import inspect
import math
import os
import re
import random
import shutil
import sys
import scipy.stats as sps
from scipy.stats import ortho_group
import time
from functools import partial
from pathlib import Path
from typing import Dict, Union, Any
from typing import TYPE_CHECKING, Optional, Iterable
from transformers.optimization import (
    get_cosine_schedule_with_warmup, 
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from sklearn.linear_model import LogisticRegressionCV
from torch import nn
from torch.func import functional_call, jvp
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params,
    is_fairscale_available,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

# from torch.optim.optimizer import StateDict, params_t
import wandb
from gradient_pruning.pruning_utils import (
    fast_random_mask_like,
    estimate_pretrained_model_magnitude_pruning_threshold,
    compute_named_parameters_to_sparsity,
)
from metrics import f1
from utils import OPT_PERTURBATION_LEVEL_TO_REGEX

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
        

def generate_rotation_matrix(d, num_rotations=None, device='cpu'):
    if num_rotations is None:
        num_rotations = d
    Q = torch.eye(d, device=device)
    for _ in range(num_rotations):
        i, j = torch.randint(0, d, (2,), device=device)
        while i == j:
            j = torch.randint(0, d, (1,), device=device)
            j = j.item()
        theta = torch.rand(1, device=device) * 2 * math.pi
        c = torch.cos(theta)
        s = torch.sin(theta)
        col_i = Q[:, i].clone()
        col_j = Q[:, j].clone()
        Q[:, i] = c * col_i - s * col_j
        Q[:, j] = s * col_i + c * col_j
    return Q

def generate_orthogonal_matrix(d, num_rotations=None, device='cpu'):
    Q = generate_rotation_matrix(d, num_rotations, device=device)
    if torch.rand(1, device=device) < 0.5:
        Q[0, :] = -Q[0, :]
    return Q

def generate_reflection_matrix(d, device='cpu'):
    Q = torch.eye(d, device=device)
    idx = random.randint(0, d - 1)
    Q[idx, idx] = -1
    return Q

def generate_semi_diagonal(n, m, distribution=sps.norm, device='cpu', dtype=torch.float32):
    p = min(n, m)

    sigma_np = distribution.rvs(size=p)
    
    sigma = torch.tensor(sigma_np, device=device, dtype=dtype)

    Sigma = torch.zeros((n, m), device=device, dtype=dtype)
    Sigma[torch.arange(p), torch.arange(p)] = sigma

    return Sigma

def create_random_matrix(n, m, device='cpu'):
    U = generate_rotation_matrix(n, device=device)
    V = generate_rotation_matrix(m, device=device)
    
    S = generate_semi_diagonal(n, m, device=device)
    
    return U @ S @ V.T, U, V, S

def torch_ortho_rvs(dim: int, device="cpu", dtype=torch.float32):
    z = torch.randn(dim, dim, device=device, dtype=dtype)

    q, r = torch.linalg.qr(z)

    d = torch.diagonal(r, 0)
    ph = d.sign()
    q *= ph.unsqueeze(0)

    return q

def generate_micola_matrix(
    dim,
    num_blocks: int = 10,
    use_PL: bool = True,
    use_P:  bool = True,
    use_PR: bool = True,
    device='cpu',
):
    base = dim // num_blocks
    rem  = dim %  num_blocks
    blocks = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    block_sizes_L = blocks
    block_sizes_R = blocks


    if block_sizes_L:
        bsL = block_sizes_L
        maxL = max(bsL)
        L_blocks = []
        for b in bsL:
            X = torch.randn(b, b, device=device)
            Qb, Rb = torch.linalg.qr(X)
            sign = torch.sign(torch.diagonal(Rb, 0))
            Qb *= sign
            if torch.det(Qb) < 0:
                Qb[:, 0] *= -1
            L_blocks.append(Qb)
        L = torch.block_diag(*L_blocks).to(device)
    else:
        L = torch.eye(dim, device=device)

    if block_sizes_R:
        bsR = block_sizes_R
        R_blocks = []
        for b in bsR:
            X = torch.randn(b, b, device=device)
            Qb, Rb = torch.linalg.qr(X)
            sign = torch.sign(torch.diagonal(Rb, 0))
            Qb *= sign
            if torch.det(Qb) < 0:
                Qb[:, 0] *= -1
            R_blocks.append(Qb)
        R = torch.block_diag(*R_blocks).to(device)
    else:
        R = torch.eye(dim, device=device)

    idx_PL = torch.randperm(dim, device=device) if use_PL else torch.arange(dim, device=device)
    idx_P  = torch.randperm(dim, device=device) if use_P  else torch.arange(dim, device=device)
    idx_PR = torch.randperm(dim, device=device) if use_PR else torch.arange(dim, device=device)

    M1 = R[:, idx_PR]
    M2 = M1[idx_P, :]
    M3 = L @ M2
    A  = M3[idx_PL, :]

    return A

def generate_reflection_matrix(d, device='cpu'):
    Q = torch.eye(d, device=device)
    i = torch.randint(0, d, (1,), device=device).item()
    Q[i, i] = -1
    return Q

def generate_rotation_via_householders(d, device='cpu'):
    v1 = torch.randn(d, device=device)
    v1 /= v1.norm()

    v2 = torch.randn(d, device=device)
    v2 -= (v1 * v2).sum() * v1
    v2 /= v2.norm()

    I = torch.eye(d, device=device)
    H1 = I - 2.0 * v1.unsqueeze(1) @ v1.unsqueeze(0)
    H2 = I - 2.0 * v2.unsqueeze(1) @ v2.unsqueeze(0)
    Q = H2 @ H1
    return Q

def generate_housholder_and_reflection_matrix(d, device='cpu'):
    Q = generate_rotation_via_householders(d, device=device)
    if torch.rand(1, device=device) < 0.5:
        Q[0, :] = -Q[0, :]
    return Q

######### FAST ONE ###########

def create_random_matrix_fast_big_matrix(param_shapes, device='cpu'):
    max_n = max(n for _, (n, m) in param_shapes)
    max_m = max(m for _, (n, m) in param_shapes)

    # U_big = generate_orthogonal_matrix(max_n, device=device)  107 sec / iter
    # V_big = generate_orthogonal_matrix(max_m, device=device)


    # U_big = generate_reflection_matrix(max_n, device=device)  0.1 sec / iter
    # V_big = generate_reflection_matrix(max_m, device=device)

    # U_big = torch.FloatTensor(ortho_group.rvs(max_n)).to(device)  8 sec / iter
    # V_big = torch.FloatTensor(ortho_group.rvs(max_m)).to(device)


    U_big = torch_ortho_rvs(max_n, device=device)  # 0.2 sec / iter ----> GREAT
    V_big = torch_ortho_rvs(max_n, device=device)

    E_dict = {}
    for name, (n, m) in param_shapes:
        k = min(n, m)
        U_slice = U_big[:n, :k]      # n x k
        V_slice = V_big[:m, :k]      # m x k
        # E = U @ I_nm @ V.T
        E = U_slice @ V_slice.T
        E_dict[name] = (E, U_slice, V_slice)
    return E_dict

def create_random_matrix_fast_per_param(param_shapes, device='cpu'):

    # U_big = generate_orthogonal_matrix(max_n, device=device)  107 sec / iter
    # V_big = generate_orthogonal_matrix(max_m, device=device)


    # U_big = generate_reflection_matrix(max_n, device=device)  0.1 sec / iter
    # V_big = generate_reflection_matrix(max_m, device=device)

    # U_big = torch.FloatTensor(ortho_group.rvs(max_n)).to(device)  8 sec / iter
    # V_big = torch.FloatTensor(ortho_group.rvs(max_m)).to(device)


    # U_big = torch_ortho_rvs(max_n, device=device)  # 0.2 sec / iter ----> GREAT
    # V_big = torch_ortho_rvs(max_n, device=device)

    E_dict = {}
    for name, (n, m) in param_shapes:
        k = min(n, m)

        U = torch_ortho_rvs(n, device=device)
        V = torch_ortho_rvs(m, device=device)

        U_slice = U[:, :k]  # (n x k)
        V_slice = V[:, :k]  # (m x k)

        E = U_slice @ V_slice.T  # (n x m)
        E_dict[name] = (E, U_slice, V_slice)

    return E_dict

def create_random_matrix_fast(param_shapes, device='cpu'):
    shape_to_names = defaultdict(list)
    for name, shape in param_shapes:
        shape_to_names[shape].append(name)

    E_dict = {}
    for (n, m), names in shape_to_names.items():
        k = min(n, m)
        # U = torch_ortho_rvs(n, device=device)
        # V = torch_ortho_rvs(m, device=device)
        U = generate_micola_matrix(n, device=device)
        V = generate_micola_matrix(m, device=device)
        U_k = U[:, :k]
        V_k = V[:, :k]
        E = U_k @ V_k.T
        for name in names:
            E_dict[name] = (E.clone(), U_k.clone(), V_k.clone())
    return E_dict


############################

def generate_orthogonal_approx(G, device='cpu'):
    assert len(G.shape) == 2, "Input must be a 2D tensor"
    m, n = G.shape
    
    # U_e m x m
    U_e = generate_orthogonal_matrix(m, device=device)
    
    #V_e n x n
    V_e = generate_orthogonal_matrix(n, device=device)
    
    approx = U_e @ V_e.T
    
    return approx

# Semiorthogonal setup

def generate_semi_rotation_matrix(m, n, num_rotations=None, device='cpu'):
    if num_rotations is None:
        num_rotations = min(m, n)

    k = min(m, n)
    V = torch.zeros((m, n), device=device)
    for i in range(min(m, k)):
        V[i, i] = 1.0

    for _ in range(num_rotations):
        i, j = torch.randint(0, k, (2,), device=device)
        while i == j:
            j = torch.randint(0, k, (1,), device=device)
            j = j.item()

        theta = torch.rand(1, device=device) * 2 * math.pi
        c = torch.cos(theta)
        s = torch.sin(theta)

        col_i = V[:, i].clone()
        col_j = V[:, j].clone()
        V[:, i] = c * col_i - s * col_j
        V[:, j] = s * col_i + c * col_j

    return V

def generate_semi_orthogonal_matrix(m, n, num_rotations=None, device='cpu'):
    V = generate_semi_rotation_matrix(m, n, num_rotations, device)
    if torch.rand(1, device=device) < 0.5:
        V[0, :] = -V[0, :]
    return V

def generate_semi_orthogonal_approx(G, device='cpu'):
    assert len(G.shape) == 2, "Input must be a 2D tensor"
    m, n = G.shape
    V_e = generate_semi_orthogonal_matrix(m, n, device=device)
    return V_e

###################

def sample_ortho_approx(shape, device='cpu'):
    assert len(shape) == 2, "Input dimension must be 2D"
    m, n = shape
    p = max(m, n)
    E = torch_ortho_rvs(p, device=device)
    return E[:m, :n]

########################################################################################

class OurTrainer(Trainer):

    # ZO-Bench added: new parameters to our traininer
    def __init__(self, evaluate_func, dev_samples, eval_samples, perturb_module_regex, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base class
        self.evaluate_func = evaluate_func
        self.dev_samples = dev_samples
        self.eval_samples = eval_samples
        self.perturb_module_regex = perturb_module_regex

    # def create_optimizer_and_scheduler(self, num_training_steps):
        # self.optimizer = SGD(self.model.parameters(), lr=self.args.learning_rate,)
        # self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_training_steps=self.args.max_steps, num_warmup_steps=1000)
        # self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer, num_warmup_steps=100, num_training_steps=self.args.max_steps, power=4)
    # #     # self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=1000, num_training_steps=self.args.max_steps)

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()  # ----newly-added

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type in ["opt", "llama", "mistral"]:
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2", "llama", "mistral"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4  # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:  # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # overload the optimizer here
        # Added zo_jaguar
        if args.trainer == "zo_adam":
            self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            # self.optimizer = {name: Adam([param], lr=args.learning_rate) for name, param in self.model.named_parameters()}
            # assert args.lr_scheduler
            # assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        elif args.trainer == "zo_sgd":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            # self.optimizer = {name: SGD([param], lr=args.learning_rate) for name, param in self.model.named_parameters()}
            # print(f"### args.lr_scheduler: {args.lr_scheduler_type}")
            # assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        elif args.trainer == "zo_jaguar":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            # self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate)
            # self.optimizer = {name: SGD([param], lr=args.learning_rate) for name, param in self.model.named_parameters()}
            # print(f"### args.lr_scheduler: {args.lr_scheduler_type}")
            # assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        elif args.trainer == "zo_muon":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            # self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate)
            '''and smth else'''
        elif args.trainer == "zo_muon_sampling":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        elif args.trainer == "zo_ns_jaguar":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        else:
            # assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            if args.optimizer == "adam":
                self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "sgd":
                self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.max_steps, eta_min=1e-8)
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        # Main training loop
        total_steps = 0
        self.sparse_grad_rng = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_sparsity = None  # None, float, or dict

        if self.args.sparse_gradient_group == "layer" or self.args.gradient_sparsity is None:
            self.gradient_sparsity = self.args.gradient_sparsity
            print(f"### layer-wise gradient sparsity = {self.gradient_sparsity}")
        elif self.args.sparse_gradient_group == "global":
            threshold = estimate_pretrained_model_magnitude_pruning_threshold(model, self.args.gradient_sparsity)
            self.gradient_sparsity = compute_named_parameters_to_sparsity(model, threshold)
            print(f"### global gradient sparsity, weight magnitude threshold = {threshold}")

        for epoch in range(epochs_trained, num_train_epochs):
            print(f"-------------------------- Training Epoch {epoch} --------------------------")
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            # Start one epoch training
            for step, inputs in enumerate(epoch_iterator):

                if total_steps % self.args.sparse_gradient_resample_steps == 0:
                    self.sparse_grad_random_seed = np.random.randint(1000000000)

                total_steps += 1

                # torch.cuda.synchronize()
                step_start_time = time.time()

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # MeZO added: estimate gradient
                # Added zo_jaguar
                if args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt"]:
                    if args.module_wise_perturbation:
                        assert args.q == 1, "module-wise perturbation only supports q=1"
                        if args.coordinate_perturbation:
                            tr_loss_step = self.zo_step_with_module_wise_perturbation_coordinate(model, inputs)
                        else:
                            tr_loss_step = self.zo_step_with_module_wise_perturbation(model, inputs)
                    elif args.q == 1:
                        tr_loss_step = self.zo_step(model, inputs)
                    elif args.q > 1:
                        tr_loss_step = self.zo_step_v1(model, inputs)
                    else:
                        raise ValueError(f"q={args.q} is not supported.")
                elif args.trainer == "zo_jaguar":
                    tr_loss_step = self.zo_jaguar_step(model, inputs)
                    self.tr_loss_step = tr_loss_step
                elif args.trainer == "zo_muon":
                    tr_loss_step = self.zo_muon_step(model, inputs)
                elif args.trainer == "zo_muon_sampling":
                    tr_loss_step = self.zo_muon_sampling_step(model, inputs)
                elif args.trainer == "zo_ns_jaguar":
                    tr_loss_step = self.zo_ns_jaguar_step(model, inputs)
                elif args.trainer == "zo_conserv":
                    tr_loss_step = self.zo_conserv_step(model, inputs)
                elif args.trainer == "forward_grad":
                    tr_loss_step = self.forward_grad_step(model, inputs)
                else:
                    if (
                            ((step + 1) % args.gradient_accumulation_steps != 0)
                            and args.local_rank != -1
                            and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    # Added zo_jaguar
                    if args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt", "zo_conserv", "zo_jaguar", "zo_muon", "zo_muon_sampling"]:
                        self.zo_update(model)
                    elif args.trainer == "forward_grad":
                        self.forward_grad_update(model)
                    else:
                        self.gradient_update(model)

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # torch.cuda.synchronize()
                train_step_duration = time.time() - step_start_time

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if self.args.eval_steps is not None and (total_steps + 1) % self.args.eval_steps == 0:
                    print(
                        f"=========================> Evaluating at step {total_steps + 1}... <=========================")
                    val_metrics = self.evaluate_func([], self.dev_samples)
                    test_metrics = self.evaluate_func([], self.eval_samples)
                    if "accuracy" in test_metrics:
                        self.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})
                        wandb.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})
                    else:
                        keys = list(test_metrics.keys())
                        log_dict = {}
                        for k in keys:
                            log_dict['test_' + k] = test_metrics[k]
                            log_dict['val_' + k] = val_metrics[k]
                        self.log(log_dict)
                        wandb.log(log_dict)

                max_memory_allocated = 0
                for device_id in range(torch.cuda.device_count()):
                    # this is not accurate since max memory does not happen simultaneously across all devices
                    max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
                self.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                          "step_consumption": train_step_duration * 1000})
                wandb.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                           "step_consumption": train_step_duration * 1000})

            if step < 0:
                # Why would this happen? I don't know, but let's be safe.
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        wandb.log(metrics)
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############

    def gradient_update(self, model):
        args = self.args

        # Gradient clipping
        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
            # deepspeed does its own clipping

            if self.do_grad_scaling:
                # Reduce gradients first for XLA
                if is_torch_tpu_available():
                    gradients = xm._fetch_gradients(self.optimizer)
                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                # AMP: gradients need unscaling
                self.scaler.unscale_(self.optimizer)

            if is_sagemaker_mp_enabled() and args.fp16:
                self.optimizer.clip_master_grads(args.max_grad_norm)
            elif hasattr(self.optimizer, "clip_grad_norm"):
                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                self.optimizer.clip_grad_norm(args.max_grad_norm)
            elif hasattr(model, "clip_grad_norm_"):
                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                model.clip_grad_norm_(args.max_grad_norm)
            else:
                # Revert to normal clipping otherwise, handling Apex or full precision
                nn.utils.clip_grad_norm_(
                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                    args.max_grad_norm,
                )

        # Optimizer step
        optimizer_was_run = True
        if self.deepspeed:
            pass  # called outside the loop
        elif is_torch_tpu_available():
            if self.do_grad_scaling:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                xm.optimizer_step(self.optimizer)
        elif self.do_grad_scaling:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
        else:
            self.optimizer.step()

        if optimizer_was_run and not self.deepspeed:
            self.lr_scheduler.step()
        model.zero_grad()

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

        for name, param in self.named_parameters_to_optim:
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(
                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def get_grad_sparsity_by_name(self, name):
        if self.gradient_sparsity is None:
            return None
        elif isinstance(self.gradient_sparsity, float):
            return self.gradient_sparsity
        elif isinstance(self.gradient_sparsity, dict):
            return self.gradient_sparsity[name]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # Sparse gradient
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            if grad_sparsity is not None:
                param.grad[fast_random_mask_like(param.grad, grad_sparsity, generator=self.sparse_grad_rng)] = 0

        return loss.detach()

    @staticmethod
    def grouped_module_iter(model: nn.Module, group_level: str) -> Iterable[nn.Module]:
        """
        Iterate over the top-level modules of a model and yield groups of modules that should be trained together.

        Parameters
        ----------
        model: nn.Module
            The model to iterate over.
        group_level: str
            The level at which to iterate over the model. One of "transformer", "mlp-attn", "linear"
        """
        reg_pattern = OPT_PERTURBATION_LEVEL_TO_REGEX[group_level]
        for name, module in model.named_modules():
            if re.match(reg_pattern, name):
                yield name, module

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation_coordinate(self, model, inputs):
        """Update the parameters right after perturbing the parameters."""
        args = self.args
        perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        all_losses = []

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            all_losses.append(loss1)

            for _ in range(args.q):
                if self.args.perturbation_mode == "one_side":
                    self.zo_perturb_parameters(scaling_factor=-1)
                    loss2 = self.zo_forward(model, inputs)
                    self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
                else:  # two side perturbation
                    self.zo_perturb_parameters(scaling_factor=-2)
                    loss2 = self.zo_forward(model, inputs)
                    self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                    # Reset model back to its parameters at start of step
                    self.zo_perturb_parameters(scaling_factor=1)

                # Set the random seed to ensure that we sample the same z for perturbation/update
                torch.manual_seed(self.zo_random_seed)
                self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
                for name, param in self.named_parameters_to_optim:
                    # Resample z
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                     dtype=param.data.dtype)
                    grad_sparsity = self.get_grad_sparsity_by_name(name)
                    if grad_sparsity is not None:
                        z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                    if args.trainer == "zo_sign_opt":
                        graddiff_times_z = np.sign(self.projected_grad) * z
                    else:
                        graddiff_times_z = self.projected_grad * z

                    param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                    self.optimizer.step()  # will only update grad that is not None.
                    param.grad = None  # avoid further update.

        assert self.args.gradient_accumulation_steps == 1

        print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation(self, model, inputs):
        """Update all parameters once after perturbing all the parameters."""
        args = self.args
        perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        all_losses = []
        module_name_to_projected_grads = {}

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            all_losses.append(loss1)

            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            module_name_to_projected_grads[module_name] = self.projected_grad

        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.projected_grad = module_name_to_projected_grads[module_name]

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                grad_sparsity = self.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                if args.trainer == "zo_sign_opt":
                    graddiff_times_z = np.sign(self.projected_grad) * z
                else:
                    graddiff_times_z = self.projected_grad * z

                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                self.optimizer.step()  # will only update grad that is not None.
                param.grad = None  # avoid further update.

        assert self.args.gradient_accumulation_steps == 1

        print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(all_losses).mean()

    ### NEW PERAMETER PERTRUB JAGUAR ### :
    """def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

        for name, param in self.named_parameters_to_optim:
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0
            param.data = param.data + scaling_factor * z * self.args.zo_eps"""

    def jaguar_pertrub_parsmeters(self, param, selected_rows, selected_cols, scaling_factor=1):
        """
        Perturb the parameter at the selected submatrix with tau.
        Input:
        - param: the parameter to perturb (torch.Tensor)
        - selected_rows: list of row indices to perturb
        - selected_cols: list of column indices to perturb
        - scaling_factor: scaling factor for the perturbation (float)
        """
        tau = self.args.zo_tau
        tau_update_vector = torch.zeros_like(param.data)
        # tau_update_vector[r, c] = τ
        tau_update_vector[selected_rows[:, None], selected_cols] = tau  
        param.data = param.data + scaling_factor * tau_update_vector
    
    def zo_muon_pertrub_parameters(self, E, scaling_factor=1, random_seed=None):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        for name, param in self.named_parameters_to_optim:
            # grad_sparsity = self.get_grad_sparsity_by_name(name)
            # z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # if grad_sparsity is not None:
                # z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0
            param.data.add(scaling_factor * E * self.args.zo_tau)
    
    # Muon:
    # @torch.compile()
    @torch.no_grad()
    def zo_muon_sampling_step(self, model, inputs, debug=False):
        args = self.args
        tau = args.zo_tau

        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        for _, p in named:
            p.grad = None

        seed = np.random.randint(1_000_000_000)
        torch.manual_seed(seed)
        if hasattr(self, 'sparse_grad_rng'):
            self.sparse_grad_rng.manual_seed(seed)

        # E
        # shapes = [(name, tuple(param.shape)) for name, param in named if param.ndim >= 2 and param.size(0) < 10000]
        # E_dict = create_random_matrix_fast(shapes, device=model.device)

        # Perturb
        def zo_muon_perturb_parameters(scaling_factor=1):
            for name, param in named:
                # if name in E_dict:
                if param.ndim >= 2 and param.size(0) < 10000:
                    E = sample_ortho_approx(param.data.shape, device=model.device)
                    param.data.add_(tau * scaling_factor * E)
                else:
                    z = torch.randn_like(param)
                    mask = getattr(self, 'get_grad_sparsity_by_name', lambda x: None)(name)
                    if mask is not None:
                        z[fast_random_mask_like(z, mask, generator=self.sparse_grad_rng)] = 0
                    param.data.add_(tau * scaling_factor * z)

        # f(z_+)
        zo_muon_perturb_parameters(1)
        loss1 = self.zo_forward(model, inputs)
        # f(z_-)
        zo_muon_perturb_parameters(-2)
        loss2 = self.zo_forward(model, inputs)
        zo_muon_perturb_parameters(1)

        delta = (loss1 - loss2) / (2 * args.zo_eps)
        # rho = sign(f(z_+) - f(z_-))
        rho = torch.sign(loss1 - loss2)

        self.optimizer.zero_grad()
        for name, param in named:
            # if name in E_dict:
            if param.ndim >= 2 and param.size(0) < 10000:
                # _, U, V = E_dict[name]
                E = sample_ortho_approx(param.data.shape, device=model.device)
                grad = rho * E
            else:
                z = torch.randn_like(param)
                mask = getattr(self, 'get_grad_sparsity_by_name', lambda x: None)(name)
                if mask is not None:
                    z[fast_random_mask_like(z, mask, generator=self.sparse_grad_rng)] = 0
                grad = (rho if args.trainer=='zo_sign_opt' else delta) * z
            param.grad = grad.to(param.dtype)

        self.optimizer.step()

        if debug:
            print(f"loss1={loss1:.6f}, loss2={loss2:.6f}, delta={delta:.6f}, rho={rho:.2f}")

        return loss1

    
    # Base ZO-MUON
    @torch.no_grad()
    def zo_muon_step(self, model, inputs):
        args = self.args

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None 

        self.zo_random_seed = np.random.randint(1000000000)

        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        if args.q == 1:
            if args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / args.zo_eps).item()
            else:  
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * args.zo_eps)).item()
                self.zo_perturb_parameters(scaling_factor=1)
        else:
            raise NotImplementedError("q > 1 is not implemented")

        torch.manual_seed(self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

            if args.trainer == "zo_sign_opt":
                grad_update = np.sign(self.projected_grad) * z
            else:
                grad_update = self.projected_grad * z

            if param.ndim >= 2 and param.size(0) < 10000:
                g = grad_update.to(torch.bfloat16)
                original_shape = g.shape
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)  
                g_ortho = zeropower_via_newtonschulz5(g, steps=5)
                # g_ortho = generate_orthogonal_approx(g, device=g.device)
                if len(original_shape) > 2:
                    g_ortho = g_ortho.view(original_shape)
                grad_update_final = g_ortho.to(param.data.dtype)
            else:
                grad_update_final = grad_update

            param.grad = grad_update_final
            self.optimizer.step()
            param.grad = None
            # param.data.add_(grad_update_final, alpha=-self._get_learning_rate())

        for _, param in self.named_parameters_to_optim:
            param.grad = None

        assert args.gradient_accumulation_steps == 1

        return loss1


    @torch.no_grad()
    def zo_jaguar_step(self, model, inputs, debug=True):
        args = self.args
        beta = args.zo_beta
        use_smoothing = args.zo_use_smoothing
        tau = args.zo_tau
        # lr = self._get_learning_rate()
        # if lr != 0:
        #     tau = lr
        # else:
        #     tau = args.zo_tau
        # print(tau)
        self.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

        self.zo_random_seed = np.random.randint(1_000_000_000)
        torch.manual_seed(self.zo_random_seed)

        selected_indices = {}
        original_values = {}

        self.optimizer.zero_grad()
        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                n_elements = param.data.shape[0]
                k = max(1, n_elements // 10)
                indices = torch.randperm(n_elements, device=param.device)[:k]
                selected_indices[name] = indices
                original_values[name] = param.data[indices].clone()
            else:
                n_rows, n_cols = param.data.shape
                k = max(1, n_rows // 10)
                m = max(1, n_cols // 10)
                selected_rows = torch.randperm(n_rows, device=param.device)[:k]
                selected_cols = torch.randperm(n_cols, device=param.device)[:m]
                selected_indices[name] = (selected_rows, selected_cols)
                original_values[name] = param.data[selected_rows[:, None], selected_cols].clone()

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] += tau
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] += tau
        loss1 = self.zo_forward(model, inputs)

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] = original_values[name] - tau
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] = original_values[name] - tau
        loss2 = self.zo_forward(model, inputs)

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] = original_values[name]
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] = original_values[name]

        # rho = sign(f(z_+) - f(z_-))
        rho = (loss1 - loss2).item() / (2 * tau)

        for name, param in self.named_parameters_to_optim:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.zero_()

            grad_update = rho
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                if use_smoothing:
                    param.grad[indices] = beta * param.grad[indices] + (1 - beta) * grad_update
                else:
                    param.grad[indices] = grad_update
            else:
                selected_rows, selected_cols = selected_indices[name]
                if use_smoothing:
                    param.grad[selected_rows[:, None], selected_cols] = beta * param.grad[selected_rows[:, None], selected_cols] + (1 - beta) * grad_update
                else:
                    param.grad[selected_rows[:, None], selected_cols] = grad_update

            param.grad = torch.sign(param.grad)

        self.optimizer.step()
        

        if debug:
            print(f"loss1={loss1.item():.4f}, loss2={loss2.item():.4f}, rho={rho:.2f}")

        assert args.gradient_accumulation_steps == 1
        return loss1

    @torch.no_grad()
    def zo_ns_jaguar_step(self, model, inputs, debug=False):
        args = self.args
        tau = args.zo_tau
        beta = args.zo_beta
        use_smoothing = args.zo_use_smoothing

        self.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

        self.zo_random_seed = np.random.randint(1_000_000_000)
        torch.manual_seed(self.zo_random_seed)

        selected_indices = {}
        original_values = {}

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                n_elements = param.data.shape[0]
                k = max(1, n_elements // 10)
                indices = torch.randperm(n_elements, device=param.device)[:k]
                selected_indices[name] = indices
                original_values[name] = param.data[indices].clone()
            else:
                n_rows, n_cols = param.data.shape
                k = max(1, n_rows // 10)
                m = max(1, n_cols // 10)
                selected_rows = torch.randperm(n_rows, device=param.device)[:k]
                selected_cols = torch.randperm(n_cols, device=param.device)[:m]
                selected_indices[name] = (selected_rows, selected_cols)
                original_values[name] = param.data[selected_rows[:, None], selected_cols].clone()

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] += tau
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] += tau
        loss1 = self.zo_forward(model, inputs)

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] = original_values[name] - tau
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] = original_values[name] - tau
        loss2 = self.zo_forward(model, inputs)

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] = original_values[name]
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] = original_values[name]

        rho = (loss1 - loss2).item() / (2 * tau)

        for name, param in self.named_parameters_to_optim:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.zero_()

            grad_update = rho
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                if use_smoothing:
                    param.grad[indices] = beta * param.grad[indices] + (1 - beta) * grad_update
                else:
                    param.grad[indices] = grad_update
                param.grad[indices] = torch.sign(param.grad[indices])
            else:
                selected_rows, selected_cols = selected_indices[name]
                if use_smoothing:
                    param.grad[selected_rows[:, None], selected_cols] = beta * param.grad[selected_rows[:, None], selected_cols] + (1 - beta) * grad_update
                else:
                    param.grad[selected_rows[:, None], selected_cols] = grad_update

                param.grad = zeropower_via_newtonschulz5(param.grad, steps=10).to(param.data.dtype)

        self.optimizer.step()
        # param.grad = None

        if debug:
            print(f"loss1={loss1.item():.4f}, loss2={loss2.item():.4f}, rho={rho:.2f}")

        assert args.gradient_accumulation_steps == 1
        return loss1

    @torch.no_grad()
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                grad_sparsity = self.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                self.optimizer.step()  # will only update grad that is not None.
                # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                param.grad = None  # avoid further update.

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_step_v1(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Works with q > 1. But for q > 1, it is not memory efficient.
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)

        for i_q in range(args.q):  # TODO shall we change the seed?
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                if i_q == 0:
                    param.grad = graddiff_times_z / args.q
                else:
                    param.grad += graddiff_times_z / args.q
                # if i_q == args.q - 1:
                #     self.optimizer.step()  # TODO If q > 1, We cannot use this trick anymore. This will cause repeated update.
                #     # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                #     param.grad = None

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q
        self.optimizer.step()
        self.optimizer.zero_grad()

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_step_v2(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Works with q > 1. But for q > 1, it is not memory efficient.
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)

        seed_list = []
        projected_grad_list = []
        for i_q in range(args.q):  # TODO shall we change the seed?
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)
            seed_list.append(self.zo_random_seed)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            projected_grad_list.append(self.projected_grad)

        # difference from v1: switch the order of for loop
        # to save memory
        for name, param in self.named_parameters_to_optim:
            for i_q in range(args.q):
                # Set the random seed to ensure that we sample the same z for perturbation/update
                torch.manual_seed(seed_list[i_q])

                graddiff_times_z = torch.zeros_like(param.data, device=param.data.device,
                                                    dtype=param.data.dtype)

                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    graddiff_times_z += np.sign(projected_grad_list[i_q]) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = torch.sign(projected_grad_list[i_q] * z)
                else:
                    # ----mezo original
                    graddiff_times_z += projected_grad_list[i_q] * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                if i_q == args.q - 1:
                    param.grad = graddiff_times_z.detach()
                    self.optimizer[name].step()
                    # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                    param.grad = None

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_conserv_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        update in the conservative way, i.e. 
        reject the update if it's not decreasing
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None

        loss0 = self.zo_forward(model, inputs)

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        if self.args.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            loss2 = self.zo_forward(model, inputs)
            self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
        else:  # two side perturbation
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)
            self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

            # Reset model back to its parameters at start of step
            self.zo_perturb_parameters(scaling_factor=1)

        def update_params(sign=1.0):
            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                param.grad = sign * graddiff_times_z
                # self.optimizer[name].step()
                self.optimizer.step()
                # param.data = param.data - graddiff_times_z / args.q
                param.grad = None

        update_params()
        loss1 = self.zo_forward(model, inputs)

        update_params(sign=-2.0)
        loss2 = self.zo_forward(model, inputs)

        # conduct the update in the conservative way
        # choose from the three and take the minimum loss one
        if loss1 > loss0:
            if loss0 < loss2:
                update_params()
        else:
            if loss1 < loss2:
                update_params(2.0)

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        # # Optimizer step
        # self.optimizer.step()
        # print(type(self.optimizer), self.optimizer)
        self.lr_scheduler.step()  # NOTE When we use own optimizer, this will no longer update the lr anymore.
        # self.optimizer.zero_grad()
        # model.zero_grad()

    @staticmethod
    @torch.no_grad()
    def functional_call_loss(params, names, buffers, model, batch):
        params = {k: v for k, v in zip(names, params)}
        outputs = functional_call(model, (params, buffers), tuple(), kwargs=batch)
        return outputs

    def forward_grad_step(self, model, inputs):
        """
        Forward Gradient Method

        Paper: Gradients without Backpropagation
        https://arxiv.org/pdf/2202.08587.pdf
        """
        args = self.args
        # print(model.__dict__)
        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None
                # this is not memory efficient.
                # param.grad = torch.zeros_like(param.data)

        # Sample the random seed for sampling vs
        self.zo_random_seed = np.random.randint(1000000000)
        torch.manual_seed(self.zo_random_seed)

        loss = 0
        vs = [torch.randn_like(p) for _, p in self.named_parameters_to_optim]

        assert args.q == 1, "q > 1"

        # fixme: this is a workaround for device map error when using jvp
        inputs = {
            k: v.to(device=model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        f = partial(
            self.functional_call_loss,
            names=[n for n, _ in self.named_parameters_to_optim], buffers=dict(model.named_buffers()),
            model=model, batch=inputs
        )

        # jvp profiling
        # torch.cuda.reset_peak_memory_stats()
        loss_, jvp_ = jvp(f, (list([p for _, p in self.named_parameters_to_optim]),), (vs,))
        # print(f"JVP peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        # print(len(jvp_), jvp_[0], jvp_[1].shape, len(jvp_[2][0][0]))
        # for v, p in zip(vs, [p for _, p in self.named_parameters_to_optim]):
        #     p.grad += v * jvp_[0].to(p.device)
        jvp_ = jvp_[0]
        with torch.no_grad():
            for v, (n, p) in zip(vs, [(n, p) for n, p in self.named_parameters_to_optim]):
                # p.grad += v * jvp_[0].to(p.device)
                # p.grad.add_(v * jvp_[0].to(p.device))
                # grad = v * jvp_[0].to(p.device) / args.q

                if "bias" not in n and "layer_norm" not in n and "layernorm" not in n:
                    p.data.sub_(self._get_learning_rate() * (v * jvp_.to(p.device) + args.weight_decay * p.data))
                else:
                    p.data.sub_(self._get_learning_rate() * (v * jvp_.to(p.device)))
        loss += loss_[0].item()

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # for name, param in self.named_parameters_to_optim:
        #     if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
        #         param.data = param.data - self._get_learning_rate() * (param.grad + args.weight_decay * param.data)
        #     else:
        #         param.data = param.data - self._get_learning_rate() * (param.grad)

        return torch.tensor(loss)

    def forward_grad_update(self, model):
        """
        Update the parameters with the estimated gradients from forward_grad_step.
        """
        args = self.args
        # for name, param in self.named_parameters_to_optim:
        #     if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
        #         param.data = param.data - self._get_learning_rate() * (param.grad + args.weight_decay * param.data)
        #     else:
        #         param.data = param.data - self._get_learning_rate() * (param.grad)

        self.lr_scheduler.step()

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
                ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
                or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
                or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
