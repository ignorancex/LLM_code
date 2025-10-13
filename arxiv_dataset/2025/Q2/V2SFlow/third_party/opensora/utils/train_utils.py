import math
import random
from collections import OrderedDict

import torch
import torch.distributed as dist
from colossalai.booster.plugin import LowLevelZeroPlugin

from ..acceleration.parallel_states import set_data_parallel_group, set_sequence_parallel_group
from ..acceleration.plugin import ZeroSeqParallelPlugin


def create_colossalai_plugin(plugin, dtype, grad_clip, sp_size, reduce_bucket_size_in_m: int = 20):
    if plugin == "zero2":
        assert sp_size == 1, "Zero2 plugin does not support sequence parallelism"
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif plugin == "zero2-seq":
        assert sp_size > 1, "Zero2-seq plugin requires sequence parallelism"
        plugin = ZeroSeqParallelPlugin(
            sp_size=sp_size,
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
):
    """
    Step the EMA model towards the current model.

    Args:
        ema_model (torch.nn.Module): The EMA model.
        model (torch.nn.Module): The current model.
        optimizer (torch.optim.Optimizer): The optimizer.
        decay (float): The decay rate.
        sharded (bool): Whether the model is sharded.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.get_working_to_master_map()[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
