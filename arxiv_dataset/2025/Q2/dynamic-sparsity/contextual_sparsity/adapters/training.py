# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from contextual_sparsity.utils.logging import CSVLogger
from contextual_sparsity.utils.misc import get_batch_size

log = logging.getLogger("adapters")

LOSSES = ["ce", "kd", "mse"]
ADAPTER_WEIGHT_FILE = "adapters.pyt"
ADAPTER_CONF_FILE = "adapters_conf.yaml"
ADAPTER_TRAIN_LOG = "adapter_train_log.csv"


def validation_loss(
    sparse_model: nn.Module,
    dataloader: DataLoader,
    preprocess_batch: Callable,
    loss_fn: Callable,
) -> float:
    """
    Computes the validation loss over the given dataloader.

    Args:
        sparse_model (nn.Module): Sparse model to evaluate on.
        dataloader (DataLoader): DataLoader to use for evaluation
        preprocess_batch (Callable): data preprocessing function to call before the model forward
        loss_fn (Callable): Loss function to compute on validation

    Returns
        float: Average validation loss over the given dataloader.
    """

    tot_loss = 0.0
    n = 0.0
    with torch.no_grad():
        sparse_model.adapters.eval()
        for batch in dataloader:
            batch = preprocess_batch(batch)
            batch_size = get_batch_size(batch)

            loss = loss_fn(batch=batch, model=sparse_model)
            tot_loss += loss.item() * batch_size
            n += batch_size

    tot_loss /= n
    return tot_loss


def _original_forward(batch: Any, model: nn.Module) -> Any:
    """
    Performs forward pass on the given data using the dense model.
    Adapters and Masking Hooks are disabled and re-enabled after the forward pass.

    Args:
        batch (Any): Batch to perform forward pass on.
        model (nn.Module): Model to use for the forward pass.
    Returns:
        Any: Output of the forward pass on the original model.
    """
    with torch.no_grad():
        # Disable all masking hooks and adapters
        for adapter in model.adapters.values():
            adapter.disable()
        for masking_hook in model.masking_hooks:
            masking_hook.set_dense()

        gt_out = model(**batch)

        # Enable everything again
        for masking_hook in model.masking_hooks:
            masking_hook.set_sparse()
        for adapter in model.adapters.values():
            adapter.enable()
    return gt_out


def cross_entropy_loss(batch: Any, model: nn.Module) -> torch.Tensor:
    """
    Computes the cross entropy loss over the given batch. This is a simple wrapper
    """
    return model(**batch).loss


def knowledge_distillation_loss(batch: Any, model: nn.Module) -> torch.Tensor:
    """
    Implementation of the Knowledge Distillation Loss function.
    The sparse model logits are matched to the dense counterpart using KL loss.
    """
    teacher_logits = _original_forward(model=model, batch=batch).logits
    logits = model(**batch).logits

    with torch.no_grad():
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
    student_logprobs = F.log_softmax(logits, dim=-1)
    loss = F.kl_div(
        input=student_logprobs.flatten(0, -2),
        target=teacher_logprobs.flatten(0, -2),
        log_target=True,
        reduction="batchmean",
    ).mean()
    return loss


LOSSES = {
    "ce": cross_entropy_loss,
    "kd": knowledge_distillation_loss,
}


def load_adapters(
    sparse_model: nn.Module,
    adapter_conf: DictConfig,
    adapter_path: str,
):
    """
    Loads adapters for a given path and configuration.
    Args:
        sparse_model: Model to which the adapters are applied
        adapter_conf: Configuration of the adapters to load
        adapter_path: Path to the adapters
    """

    log.info(f"Loading adapter from {adapter_path}")
    # Check the configuration matches
    config_file = os.path.join(adapter_path, ADAPTER_CONF_FILE)
    weight_file = os.path.join(adapter_path, ADAPTER_WEIGHT_FILE)

    with open(config_file, "r") as f:
        saved_adapters_conf = OmegaConf.load(f)

    # Remove the load key if any
    if "load" in adapter_conf:
        with open_dict(adapter_conf):
            del adapter_conf.load

    # If the models are inconsistent, there is no possible way to load
    if adapter_conf.model != saved_adapters_conf.model:
        log.warning(f"Incompatible adapters config in {adapter_path}. Trying to load them anyway.")
    # If the training configuration or models are different, we can try loading
    if adapter_conf != saved_adapters_conf:
        log.warning(
            "The adapter configuration is not compatible with the current configuration! "
            "The current configuration will be overwritten."
        )
    sparse_model.adapters.load_state_dict(torch.load(weight_file))
    sparse_model.adapters.eval()
    return sparse_model


def save_adapters(sparse_model: nn.Module, adapter_weights_path: str):
    """
    Saves adapters attached to the given model on the specified path.

    Args:
        sparse_model: Model to which the adapters are applied
        adapter_weights_path: Path corresponding to the location in which the adapters weights are saved.
    """
    log.info(f"Adapter weights saved to {adapter_weights_path}")
    torch.save(sparse_model.adapters.state_dict(), adapter_weights_path)


def train_adapters(
    sparse_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    adapter_conf: DictConfig,
):
    """
    Trains adapters for the given model using the procedure specified in the configuration.

    Args:
        sparse_model: Model to which the adapters are applied
        tokenizer: tokenizer used to tokenize the train and validation set.
        adapter_conf: Configuration of the adapters to train
    """
    train_loader = instantiate(
        adapter_conf.training.data.train_on,
        tokenizer=tokenizer,
    )
    valid_loader = instantiate(
        adapter_conf.training.data.valid_on,
        tokenizer=tokenizer,
    )
    preprocess_batch = instantiate(adapter_conf.training.data.preprocess_batch)

    # Determine paths to save configuration and adapter weights
    adapter_weights_path = os.path.abspath(os.path.join("", ADAPTER_WEIGHT_FILE))
    adapter_conf_path = os.path.abspath(os.path.join("", ADAPTER_CONF_FILE))
    adapter_train_log_path = os.path.abspath(os.path.join("", ADAPTER_TRAIN_LOG))

    # Store the initial adapters and configurations
    save_adapters(sparse_model=sparse_model, adapter_weights_path=adapter_weights_path)
    OmegaConf.save(
        adapter_conf,
        adapter_conf_path,
    )

    # Save the original datatype and move the model to bf16
    dtype = next(iter(sparse_model.adapters.parameters())).dtype
    sparse_model.to(dtype=torch.bfloat16)

    #############################
    # Instantiate the optimizer #
    #############################
    optimizer_class = instantiate(adapter_conf.training.optimization.optimizer)
    patience = adapter_conf.training.optimization.patience
    n_epochs = adapter_conf.training.optimization.n_epochs
    gradient_accumulation_steps = adapter_conf.training.optimization.gradient_accumulation_steps
    assert gradient_accumulation_steps > 0

    for parameter in sparse_model.parameters():
        parameter.requires_grad = False

    for parameter in sparse_model.adapters.parameters():
        parameter.requires_grad = True

    opt = optimizer_class(sparse_model.adapters.parameters())

    if "lr_scheduler" in adapter_conf.training.optimization:
        lr_scheduler_builder = instantiate(
            adapter_conf.training.optimization.lr_scheduler,
        )
        lr_scheduler = lr_scheduler_builder(
            optimizer=opt,
            total_iterations=len(train_loader) * n_epochs,
        )
    else:
        lr_scheduler = None

    #################
    # Loss function #
    #################
    if adapter_conf.training.loss not in LOSSES:
        raise ValueError(f"Loss is not supported. Please choose one of {LOSSES.keys()}")
    loss_fn = LOSSES[adapter_conf.training.loss]

    # Setting training variables
    epochs_pbar = tqdm(total=n_epochs)
    training = True
    epoch = 0
    sub_iteration = 0
    iteration = 0
    iteration_loss = 0
    max_epochs = adapter_conf.training.optimization.n_epochs
    validate_every = adapter_conf.training.optimization.validate_every
    if patience is None:
        patience = n_epochs
    original_patience = patience

    ##########
    # Logger #
    ##########
    log.info(f"Adapter train logs saved to {adapter_train_log_path}")
    train_logger = CSVLogger(csv_filepath=adapter_train_log_path)

    val_loss = validation_loss(
        sparse_model=sparse_model,
        dataloader=valid_loader,
        preprocess_batch=preprocess_batch,
        loss_fn=loss_fn,
    )

    train_logger.log(epoch=epoch, iteration=iteration, split="valid", loss=val_loss)
    best_loss = val_loss
    log.info(f"Initial validation Loss: {val_loss}\n")

    #################
    # Training Loop #
    #################
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    sparse_model.eval()
    log.info("Training the adapters...")

    while training:
        sparse_model.adapters.train()

        for batch in train_loader:
            # Training sub-iteration
            batch = preprocess_batch(batch)
            loss = loss_fn(batch=batch, model=sparse_model) / float(gradient_accumulation_steps)
            iteration_loss += loss.item()
            loss.backward()
            sub_iteration += 1

            # Every gradient_accumulation_steps perform an optimizer step and increase the iteration counter
            if sub_iteration % gradient_accumulation_steps == 0:
                opt.step()
                opt.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                train_logger.log(
                    epoch=epoch, iteration=iteration, split="train", loss=iteration_loss
                )
                iteration_loss = 0
                sub_iteration = 0
                iteration += 1

            # Validation is performed every validate_every on the first sub-iteration
            if iteration % validate_every == 0 and sub_iteration == 0:
                epoch += 1
                val_loss = validation_loss(
                    sparse_model=sparse_model,
                    dataloader=valid_loader,
                    preprocess_batch=preprocess_batch,
                    loss_fn=loss_fn,
                )
                sparse_model.adapters.train()
                epochs_pbar.set_postfix({"Validation Loss": val_loss})
                train_logger.log(epoch=epoch, iteration=iteration, split="valid", loss=val_loss)
                epochs_pbar.update(1)
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_adapters(
                        sparse_model=sparse_model,
                        adapter_weights_path=adapter_weights_path,
                    )
                    patience = original_patience
                    log.info(f"Best validation Loss: {val_loss}\n")
                else:
                    patience -= 1

                # If the patience reaches 0, stop training
                if patience == 0:
                    log.info(f"Early stopping at epoch {epoch}")
                    training = False
                    break

            # Stop when max_epochs is reached
            if epoch >= max_epochs:
                training = False
                break

    # Set the grad_enabled to the same state it was before starting the training
    torch.set_grad_enabled(grad_enabled)
    # Restore the original dtype
    sparse_model.to(dtype)

    # Load the adapters from disk
    load_adapters(
        sparse_model=sparse_model,
        adapter_path="",
        adapter_conf=adapter_conf,
    )
    sparse_model.eval()
