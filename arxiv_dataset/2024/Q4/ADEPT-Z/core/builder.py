"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-03-31 17:48:41
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:37:28
"""

#########
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.loss import AdaptiveLossSoft, KLLossMixed
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.types import Device

from core.acc_proxy.acc_predictor import AccuracyPredictor, RobustnessPredictor
from core.cost.cost_predictor import CostPredictor
from core.datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    FashionMNISTDataset,
    MNISTDataset,
    SVHNDataset,
)
from core.models import *
from core.optimizer.base import evo_args
from core.optimizer.utils import Evaluator

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_weight_optimizer(model: nn.Module, name: str = None) -> Optimizer:
    name = (name or configs.weight_optimizer.name).lower()

    weight_decay = float(getattr(configs.weight_optimizer, "weight_decay", 0))
    bn_weight_decay = float(getattr(configs.weight_optimizer, "bn_weight_decay", 0))
    bias_decay = float(getattr(configs.weight_optimizer, "bias_decay", 0))
    perm_decay = float(getattr(configs.weight_optimizer, "perm_decay", 0))
    dc_decay = float(getattr(configs.weight_optimizer, "dc_decay", 0))
    groups = {
        str(d): []
        for d in set(
            [
                weight_decay,
                bn_weight_decay,
                bias_decay,
                perm_decay,
                dc_decay,
            ]
        )
    }

    conv_linear = tuple([nn.Linear, _ConvNd] + list(getattr(model, "_conv_linear", [])))
    for m in model.modules():
        if isinstance(m, conv_linear):
            groups[str(weight_decay)].append(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                groups[str(bias_decay)].append(m.bias)
        elif isinstance(m, _BatchNorm) and not bn_weight_decay:
            if m.weight is not None and m.weight.requires_grad:
                groups[str(bn_weight_decay)].append(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                groups[str(bn_weight_decay)].append(m.bias)
        elif isinstance(m, SuperCRLayer):
            if hasattr(m, "weight") and m.weight.requires_grad:
                groups[str(perm_decay)].append(m.weight)
        elif isinstance(m, SuperDCFrontShareLayer):
            if hasattr(m, "weight") and m.weight.requires_grad:
                groups[str(dc_decay)].append(m.weight)

    selected_params = []
    for v in groups.values():
        selected_params += v

    params_grad = model.weight_params
    other_params = list(set(params_grad) - set(selected_params))
    groups[str(weight_decay)] += (
        other_params  # unassigned parameters automatically assigned to weight decay group
    )

    assert len(params_grad) == sum(len(p) for p in groups.values())
    params = [
        dict(params=params, weight_decay=float(decay_rate))
        for decay_rate, params in groups.items()
    ]
    return make_optimizer(params, name, configs.weight_optimizer)


def make_arch_optimizer(model: nn.Module, name: str = None) -> Optimizer:
    name = (name or configs.arch_optimizer.name).lower()

    theta_decay = float(getattr(configs.arch_optimizer, "weight_decay", 5e-4))
    theta = [model.super_layer.sampling_coeff]
    params = [
        dict(params=theta, weight_decay=theta_decay),
    ]
    return make_optimizer(params, name, configs.arch_optimizer)


def make_dataloader(
    cfg: dict = None, splits=["train", "valid", "test"]
) -> Tuple[DataLoader, DataLoader]:
    cfg = cfg or configs.dataset
    name = cfg.name.lower()
    if name == "mnist":
        train_dataset, validation_dataset, test_dataset = (
            MNISTDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                binarize_threshold=0.273,
                digits_of_interest=list(range(10)),
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "fashionmnist":
        train_dataset, validation_dataset, test_dataset = (
            FashionMNISTDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar10":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR10Dataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar100":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR100Dataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "svhn":
        train_dataset, validation_dataset, test_dataset = (
            SVHNDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            cfg.img_height,
            cfg.img_width,
            dataset_dir=cfg.root,
            transform=cfg.transform,
        )
        validation_dataset = None

    train_loader = (
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=configs.run.batch_size,
            shuffle=int(cfg.shuffle),
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if train_dataset is not None
        else None
    )

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = (
        torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if test_dataset is not None
        else None
    )

    return train_loader, validation_loader, test_loader


def make_model(
    device: Device, model_cfg: Optional[str] = None, random_state: int = None, **kwargs
) -> nn.Module:
    model_cfg = model_cfg or configs.model
    name = model_cfg.name
    if "mlp" in name.lower():
        model = eval(name)(
            n_feat=configs.dataset.img_height * configs.dataset.img_width,
            n_class=configs.dataset.n_class,
            hidden_list=model_cfg.hidden_list,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=model_cfg.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=model_cfg.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "cnn" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            hidden_list=model_cfg.hidden_list,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            v_max=configs.quantize.v_max,
            # v_pi=configs.quantize.v_pi,
            act_thres=model_cfg.act_thres,
            photodetect=model_cfg.photodetect,
            bias=False,
            device=device,
            super_layer_name=configs.super_layer.name,
            super_layer_config=configs.super_layer.arch,
            bn_affine=model_cfg.bn_affine,
        ).to(device)
        model.reset_parameters(random_state)
        # model.super_layer.set_sample_arch(configs.super_layer.sample_arch)
    elif "supervgg" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            hidden_list=model_cfg.hidden_list,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            v_max=configs.quantize.v_max,
            # v_pi=configs.quantize.v_pi,
            act_thres=model_cfg.act_thres,
            photodetect=model_cfg.photodetect,
            bias=False,
            device=device,
            super_layer_name=configs.super_layer.name,
            super_layer_config=configs.super_layer.arch,
            bn_affine=model_cfg.bn_affine,
        ).to(device)
    elif "vgg" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            act_thres=model_cfg.act_thres,
            bias=False,
            device=device,
            bn_affine=model_cfg.bn_affine,
        ).to(device)
        model.reset_parameters(random_state)
    elif "superresnet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            hidden_list=model_cfg.hidden_list,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            v_max=configs.quantize.v_max,
            # v_pi=configs.quantize.v_pi,
            act_thres=model_cfg.act_thres,
            photodetect=model_cfg.photodetect,
            bias=False,
            device=device,
            super_layer_name=configs.super_layer.name,
            super_layer_config=configs.super_layer.arch,
            bn_affine=model_cfg.bn_affine,
        ).to(device)
        model.reset_parameters(random_state)
    elif "resnet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            act_thres=model_cfg.act_thres,
            bias=False,
            device=device,
            bn_affine=model_cfg.bn_affine,
        ).to(device)
    elif "efficientnet" in name.lower():
        model = eval(name)(
            pretrained=True,
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            device=device,
        ).to(device)
    elif "mobilenet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            num_classes=configs.dataset.num_classes,
            device=device,
        ).to(device)
        # model.reset_parameters()
    elif "unet" in name.lower():
        model = eval(name)(
            in_channels=configs.dataset.in_channels,
            out_channels=model_cfg.out_channels,
            dim=model_cfg.dim,
            act_func=model_cfg.act_func,
            domain_size=model_cfg.domain_size,
            grid_step=model_cfg.grid_step,
            buffer_width=model_cfg.buffer_width,
            dropout_rate=model_cfg.dropout_rate,
            drop_path_rate=model_cfg.drop_path_rate,
            aux_head=model_cfg.aux_head,
            aux_head_idx=model_cfg.aux_head_idx,
            pos_encoding=model_cfg.pos_encoding,
            device=device,
            **kwargs,
        ).to(device)
    elif "dpe" in name.lower():
        model = eval(name)(
            n_pads=model_cfg.n_pads,
            n_ports=model_cfg.n_ports,
            act_cfg=model_cfg.act_cfg,
            hidden_dims=model_cfg.hidden_dims,
            dropout=model_cfg.dropout_rate,
            device=device,
            **kwargs,
        ).to(device)

    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {model_cfg.name}")
    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configs.run.n_epochs),
            eta_min=float(configs.scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    # cfg = cfg or configs.criterion
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif name == "adaptive":
        criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    elif name == "mixed_kl":
        criterion = KLLossMixed(
            T=getattr(cfg, "T", 3),
            alpha=getattr(cfg, "alpha", 0.9),
        )
    else:
        raise NotImplementedError(name)
    return criterion


def make_search_engine(
    name: str = None, model=None, calibration_loader=None, criterion=None, device=None
):
    name = name or configs.evo_search.name
    arg = evo_args(
        es_population_size=configs.evo_search.population_size,
        es_parent_size=configs.evo_search.parent_size,
        matrix_size=configs.super_layer.arch.n_waveguides,
        es_mutation_size=configs.evo_search.mutation_size,
        es_mutation_rate_dc=configs.evo_search.mutation_rate_dc,
        es_mutation_rate_cr=configs.evo_search.mutation_rate_cr,
        es_mutation_rate_block=configs.evo_search.mutation_rate_block,
        es_mutation_ops_dc=configs.evo_search.mutation_ops_dc,
        es_mutation_ops_cr=configs.evo_search.mutation_ops_cr,
        es_crossover_size=configs.evo_search.crossover_size,
        es_crossover_rate_dc=configs.evo_search.crossover_rate_dc,
        es_crossover_cr_split_ratio=configs.evo_search.crossover_cr_split_ratio,
        es_crossover_rate_block=configs.evo_search.crossover_rate_block,
        es_n_iterations=configs.evo_search.n_iterations,
        es_n_global_search=configs.evo_search.n_global_search,
        es_score_mode=configs.evo_search.score_mode,
        es_num_procs=configs.evo_search.num_procs,
        es_constr=configs.evo_search.constr,  # robustness as constraint
    )

    alg_cfg = dict()
    for alg_name, cfg in configs.evo_search.acc_proxy.items():
        weight = float(cfg.weight)
        if weight < 1e-5:
            continue
        if alg_name == "gradnorm":
            alg_cfg[alg_name] = dict(
                weight=weight,
                config=dict(
                    resolution=configs.dataset.img_height,
                    batch_size=cfg.batch_size,
                    fp16=True,
                    in_channels=configs.dataset.in_channels,
                ),
            )
        elif alg_name == "zico":
            alg_cfg[alg_name] = dict(
                weight=weight,
                config=dict(
                    calib_dataloader=calibration_loader,
                    criterion=criterion,
                    fp16=True,
                    device=device,
                ),
            )
        elif alg_name == "zen":
            alg_cfg[alg_name] = dict(
                weight=weight,
                config=dict(
                    mixup_gamma=cfg.mixup_gamma,
                    resolution=configs.dataset.img_height,
                    batch_size=cfg.batch_size,
                    repeat=cfg.repeat,
                    fp16=True,
                    device=device,
                    in_channels=configs.dataset.in_channels,
                ),
            )
        elif alg_name == "params":
            alg_cfg[alg_name] = dict(
                weight=weight,
                config=dict(
                    in_channels=configs.dataset.in_channels,
                ),
            )
        elif alg_name == "sparsity":
            alg_cfg[alg_name] = dict(
                weight=weight,
                config=dict(),
            )
        elif alg_name == "expressivity":
            alg_cfg[alg_name] = dict(
                weight=weight,
                config=dict(
                    num_samples=cfg.num_samples,
                    num_steps=cfg.num_steps,
                    verbose=False,
                ),
            )
        else:
            raise NotImplementedError

    acc_predictor = AccuracyPredictor(
        model,
        alg_cfg=alg_cfg,
    )

    cost_predictor = CostPredictor(
        model,
        cost_name=configs.evo_search.cost.mode,
        work_freq=float(configs.evo_search.cost.work_freq),
        work_prec=int(configs.evo_search.cost.work_prec),
    )

    robustness_predictor = RobustnessPredictor(
        model,
        alg=configs.evo_search.robustness.mode,
        num_samples=int(configs.evo_search.robustness.num_samples),
        phase_noise_std=float(configs.evo_search.robustness.phase_noise_std),
        sigma_noise_std=float(configs.evo_search.robustness.sigma_noise_std),
        dc_noise_std=float(configs.evo_search.robustness.dc_noise_std),
        cr_tr_noise_std=float(configs.evo_search.robustness.cr_tr_noise_std),
        cr_phase_noise_std=float(configs.evo_search.robustness.cr_phase_noise_std)
        * (np.pi / 180),
    )

    evaluator = Evaluator(
        arg,
        acc_predictor,
        cost_predictor,
        robustness_predictor,
        score_mode=arg.es_score_mode,
        multiobj=configs.evo_search.multiobj,  # enable multiobj
        num_procs=int(arg.es_num_procs),
    )

    es_engine = eval(name)(
        population_size=arg.es_population_size,
        parent_size=arg.es_parent_size,
        matrix_size=arg.matrix_size,
        mutation_size=arg.es_mutation_size,
        mutation_rate_dc=arg.es_mutation_rate_dc,
        mutation_rate_cr=arg.es_mutation_rate_cr,
        mutation_rate_block=arg.es_mutation_rate_block,
        mutation_ops_dc=arg.es_mutation_ops_dc,
        mutation_ops_cr=arg.es_mutation_ops_cr,
        crossover_size=arg.es_crossover_size,
        crossover_rate_dc=arg.es_crossover_rate_dc,
        crossover_cr_split_ratio=arg.es_crossover_cr_split_ratio,
        crossover_rate_block=arg.es_crossover_rate_block,
        super_layer=model.super_layer,
        constraints=arg.es_constr,
        evaluator=evaluator,
    )

    return es_engine, evaluator
