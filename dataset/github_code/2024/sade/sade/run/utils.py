import logging
import os

import ants
import numpy as np
import torch

from sade.models.ema import ExponentialMovingAverage


def restore_checkpoint(
    ckpt_dir, state, device, restore_optim_states=True, raise_error=False
):
    if not os.path.exists(ckpt_dir):
        if raise_error:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}")
        else:
            logging.warning(
                f"No checkpoint found at {ckpt_dir}. "
                f"Returned the same state as input"
            )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]

        if restore_optim_states:
            if state.get("optimizer") is not None:
                state["optimizer"].load_state_dict(loaded_state["optimizer"])
                state["optimizer"].param_groups[0]["capturable"] = True

            if "scheduler" in loaded_state and state["scheduler"] is not None:
                state["scheduler"].load_state_dict(loaded_state["scheduler"])

            if "grad_scaler" in loaded_state and "grad_scaler" in state:
                state["grad_scaler"].load_state_dict(loaded_state["grad_scaler"])

        logging.info(f"Loaded model state at step {state['step']} from {ckpt_dir}")
        return state


def restore_pretrained_weights(ckpt_dir, state, device):
    assert (
        state["step"] == 0
    ), "Can only load pretrained weights when starting a new run"
    assert os.path.exists(
        ckpt_dir
    ), f"Pretrain weights directory {ckpt_dir} does not exist"
    assert (
        state["model"].training
    ), "Model must be in training mode to appropriately load pretrained weights"

    loaded_state = torch.load(ckpt_dir, map_location=device)
    state["model_checkpoint_step"] = loaded_state["step"]
    dummy_ema = ExponentialMovingAverage(state["model"].parameters(), decay=0.999)
    dummy_ema.load_state_dict(loaded_state["ema"])
    dummy_ema.lazy_copy_to(state["model"].parameters())
    logging.info(
        f"Loaded pretrained EMA weights from {ckpt_dir} at {loaded_state['step']}"
    )

    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }

    if state.get("scheduler") is not None:
        saved_state["scheduler"] = state["scheduler"].state_dict()

    if state.get("grad_scaler") is not None:
        saved_state["grad_scaler"] = state["grad_scaler"].state_dict()

    torch.save(saved_state, ckpt_dir)
    return


def plot_slices(x, fname, channels_first=False):
    """Plot slices of a 5D tensor."""

    if channels_first:
        if isinstance(x, np.ndarray):
            x = np.transpose(x, axes=(0, 2, 3, 4, 1))

        if isinstance(x, torch.Tensor):
            x = x.permute(0, 2, 3, 4, 1).detach().cpu().numpy()

    # Get alternating channels per sample
    c = x.shape[-1]
    x_imgs = [ants.from_numpy(sample[..., i % c]) for i, sample in enumerate(x)]
    ants.plot_ortho_stack(
        x_imgs,
        orient_labels=False,
        dpi=100,
        filename=fname,
        transparent=True,
        crop=True,
        scale=(0.01, 0.99),
    )

    return


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_flow_rundir(config, workdir):
    hparams = f"psz{config.flow.local_patch_config.kernel_size}"
    hparams += f"-globalpsz{config.flow.global_patch_config.kernel_size}"
    # haparams += f"-nb{config.flow.num_blocks}"
    hparams += (
        f"-nb{config.flow.num_blocks}-lr{config.flow.lr}-bs{config.training.batch_size}"
    )
    hparams += (
        f"-np{config.flow.patches_per_train_step}-kimg{config.flow.training_kimg}"
    )
    rundir = os.path.join(workdir, "flow", hparams)
    return rundir
