# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from copy import deepcopy
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from contextual_sparsity.masking_hooks.trained.loss import PredictorLoss
from contextual_sparsity.utils.misc import get_batch_size, move_to_device

log = logging.getLogger(__name__)

LOSS = "loss"


def optimize(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: Union[str, torch.device],
    n_epochs: int,
    patience: Optional[int],
    optimizer: Callable[[Any], Optimizer],
    loss_func: PredictorLoss,
) -> pd.DataFrame:
    """
    Train the parameters of a given loss function on a given train_loader. Validate on valid_valid to determine
    early stopping, otherwise train for n_epochs. The function returns a dataframe containing the log of training
    and validation losses.
    """

    opt = optimizer([param for param in loss_func.parameters() if param.requires_grad])
    train_log = []
    iteration = 0
    best_loss = float("inf")
    best_weights = None
    if patience is None:
        patience = n_epochs
    original_patience = patience
    loss_func = loss_func.to(device)

    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)

    # Progress bars
    epochs_pbar = tqdm(total=n_epochs)

    for epoch in range(n_epochs):
        # Train iteration
        loss_func.train()
        for batch in train_loader:
            batch = move_to_device(batch, device)

            batch_out = loss_func(**batch)

            if not isinstance(batch_out, dict):
                batch_out = {LOSS: batch_out}
            else:
                assert LOSS in batch_out
            loss = batch_out[LOSS]

            opt.zero_grad()
            loss.backward()
            opt.step()

            log_entry = {
                k: v.item() if torch.is_tensor(v) else v.detach() for k, v in batch_out.items()
            }
            log_entry.update({"epoch": epoch, "iteration": iteration, "split": "train"})
            train_log.append(log_entry)
            iteration += 1

        # Validation
        val_loss = 0.0
        n = 0.0
        loss_func.eval()
        for batch in valid_loader:
            batch = move_to_device(batch, device)
            batch_size = get_batch_size(batch)

            batch_out = loss_func(**batch)
            if not isinstance(batch_out, dict):
                batch_out = {LOSS: batch_out}
            else:
                assert LOSS in batch_out
            loss = batch_out[LOSS]

            val_loss += loss.item() * batch_size
            n += batch_size
            log_entry = {
                k: v.item() if torch.is_tensor(v) else v.detach() for k, v in batch_out.items()
            }
            log_entry.update({"epoch": epoch, "iteration": iteration, "split": "valid"})
            train_log.append(log_entry)

        val_loss /= n
        epochs_pbar.set_postfix({"valid_loss": val_loss})
        epochs_pbar.update(1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = deepcopy(loss_func.predictor.state_dict())
            patience = original_patience
            log.info(f"Best validation loss: {val_loss}")
        else:
            patience -= 1

        if patience == 0:
            log.info(f"Early stopping at epoch {epoch}")
            break

    torch.set_grad_enabled(grad_enabled)
    if best_weights is not None:
        loss_func.predictor.load_state_dict(best_weights)
    return pd.DataFrame(train_log)
