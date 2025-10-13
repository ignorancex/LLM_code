from .gui_eval import action_schema_check, action_args_check, action_type_check,react_check
from .process import _prepare_messages,_process_inputs,_create_inputs
from .dataloader import GlobalDistributed0MQDataLoader
from .dataset import GUIRFTDataset,GUIMTRFTDataset
from .dataloader import GlobalDistributed0MQDataLoader

__all__ = [
    "GUIRFTDataset","GUIMTRFTDataset",
    "action_schema_check","action_args_check","action_type_check","react_check",
    "_prepare_messages","_process_inputs","_create_inputs",
    "GlobalDistributed0MQDataLoader",
    "no_sync","Timer","logger"
    ]

import os
import time
import torch
import logging
from contextlib import contextmanager,nullcontext
from accelerate import Accelerator

import warnings
import functools
from accelerate.utils.fsdp_utils import is_compiled_module, get_module_children_bottom_up,fsdp2_prepare_auto_wrap_policy
from accelerate.utils import FullyShardedDataParallelPlugin
import torch.distributed as dist

@contextmanager
def Timer(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    # torch.cuda.synchronize()
    logger.info(f"[TIMER] {name}: {(end - start)*1000:.2f} ms")

logger = logging.getLogger("ARL")    
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@contextmanager
def no_sync(self:Accelerator, model):
    '''For FSPD2, disable gradient synchronization for all model parameters.'''
    context = nullcontext
    if self.use_distributed:
        context = getattr(model, "no_sync", context)
        if self.is_fsdp2 and os.environ.get("ENABLE_FSDP2_NOSYNC","False") == "True":
            model.set_requires_gradient_sync(False)
            yield
            model.set_requires_gradient_sync(True)
            return

    with context():
        yield




def fsdp2_prepare_model(model: torch.nn.Module,mesh:dist.device_mesh.DeviceMesh) -> torch.nn.Module:
    """Prepares the model for FSDP2 in-place. Also returns the model to avoid misuse of the original model.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to prepare

    Returns:
        `torch.nn.Module`: Prepared model
    """
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    is_type_fsdp = isinstance(model, FSDPModule) or (
        is_compiled_module(model) and isinstance(model._orig_mod, FSDPModule)
    )
    if is_type_fsdp:
        return model

    fsdp2_plugin = FullyShardedDataParallelPlugin()
    
    original_sd = model.state_dict()

    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

    # We need the `auto_wrap_policy` original type to create a custom poilicy function for sharding
    # This is because `fully_shard` doesn't support old auto wrap policies, rather we have to imitate the behaviour
    auto_wrap_policy_type = None
    if fsdp2_plugin.auto_wrap_policy is transformer_auto_wrap_policy:
        auto_wrap_policy_type = "transformer"
    elif fsdp2_plugin.auto_wrap_policy is size_based_auto_wrap_policy:
        auto_wrap_policy_type = "size"

    # We set `auto_wrap_policy` to `functools.partial` to avoid creating it again
    # This is because of `apply_activation_checkpointing` which will can reuse this function
    fsdp2_plugin.set_auto_wrap_policy(model)

    if fsdp2_plugin.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        # Apply activation checkpointing before applying `fully_shard`
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            auto_wrap_policy=fsdp2_plugin.auto_wrap_policy,
        )
    fsdp2_kwargs = {
        "reshard_after_forward": fsdp2_plugin.reshard_after_forward,
        "offload_policy": fsdp2_plugin.cpu_offload,
        "mesh": mesh,
        # `fully_shard` doesn't accept `None` in case of `MixedPrecisionPolicy`
        "mp_policy": fsdp2_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
    }

    auto_wrap_policy = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, auto_wrap_policy_type, model)
    if auto_wrap_policy is not None:
        # We skip the model itself, as that one is always wrapped
        for module in get_module_children_bottom_up(model)[:-1]:
            if auto_wrap_policy(module):
                fully_shard(module, **fsdp2_kwargs)

    fully_shard(model, **fsdp2_kwargs)

    if fsdp2_plugin.cpu_ram_efficient_loading:
        # If `cpu_ram_efficient_loading` is enabled, only rank 0 loads the weights
        # Other ranks have an empty model on `meta` device, so we need to distribute the weights properly
        # fsdp2_load_full_state_dict(model, original_sd)
        assert False, "Currently not support `cpu_ram_efficient_loading` with Tensor Parallel."

    if model.dtype != torch.float32:
        # We upcast the model according to `deepspeed`'s implementation
        # More info about this can be found in `accelerator.py:prepare_model`s FSDP1 section
        model = model.to(torch.float32)
        # if accelerator.is_main_process:
        #     # TODO(siro1): Add a warning for each parameter that was upcasted
        #     warnings.warn(
        #         "FSDP upcast of low precision parameters to fp32 (since mixed_precision != 'no') may affect the precision of model checkpoints."
        #     )
    return model