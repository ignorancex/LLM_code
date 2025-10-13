from typing import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from moe import RTMoE, make_expert_vgrid
from routing_net import FiringRateEncoder, RoutingNet

from .misc import EasyDict


class Model(eqx.Module):
    input_block: Callable
    output_block: Callable
    rtmoe: RTMoE

    def __init__(self, cfg: EasyDict, key: PRNGKeyArray):
        key = jr.PRNGKey(cfg.seed)
        keys = jr.split(key, 6)
        self.input_block = create_input_block(cfg, keys[0])
        self.output_block = create_output_block(cfg, keys[1])
        experts_params, experts_struct = create_experts(cfg, keys[2])
        router = create_routing_net(cfg, keys[3])
        self.rtmoe = RTMoE(
            routing_net=router,
            expert_params=experts_params,
            expert_struct=experts_struct,
            res=cfg.res,
        )


def create_routing_net(cfg: EasyDict, key: PRNGKeyArray) -> RoutingNet:
    keys = jr.split(key, num=3)
    fr_encoder = FiringRateEncoder(
        in_size=cfg.hdim,
        num_gates=cfg.num_gates,
        num_experts=cfg.n_exp_per_l,
        depth=0,
        key=keys[1],
    )
    syn_net = RoutingNet(
        n_exps=cfg.n_exp_per_l,
        fr_encoder=fr_encoder,
        n_layers=cfg.n_layers,
        num_gates=cfg.num_gates,
        key=keys[2],
    )
    return syn_net


def create_experts(cfg: EasyDict, key: PRNGKeyArray):
    def _expert_recipe(k: PRNGKeyArray) -> eqx.nn.MLP:
        return eqx.nn.MLP(
            in_size=cfg.hdim,
            out_size=cfg.hdim,
            width_size=cfg.hdim,
            depth=2,
            key=k,
        )

    experts_params, experts_struct = make_expert_vgrid(
        expert_recipe=_expert_recipe,
        n_layers=cfg.n_layers,
        n_experts=cfg.n_exp_per_l,
        key=key,
    )
    return experts_params, experts_struct


def create_output_block(cfg: EasyDict, key: PRNGKeyArray):
    ob_dim = cfg.hdim
    ob = eqx.nn.Linear(ob_dim, cfg.odim, key=key)
    return ob


def create_input_block(cfg: EasyDict, key: PRNGKeyArray):
    if cfg.input_block == "conv":
        ckeys = jr.split(key, 4)
        ib = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(
                    in_channels=1 if cfg.dset in ("mnist", "fashion_mnist") else 3,
                    out_channels=4,
                    kernel_size=3,
                    padding="VALID",
                    stride=1,
                    key=ckeys[0],
                ),
                eqx.nn.MaxPool2d(
                    kernel_size=2,
                    stride=1,
                ),
                eqx.nn.Lambda(jnn.relu),
                eqx.nn.Conv2d(
                    in_channels=4,
                    out_channels=8,
                    kernel_size=3,
                    padding="VALID",
                    stride=1,
                    key=ckeys[1],
                ),
                eqx.nn.MaxPool2d(
                    kernel_size=2,
                    stride=1,
                ),
                eqx.nn.Lambda(jnn.relu),
                eqx.nn.Conv2d(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=3,
                    padding="VALID",
                    stride=1,
                    key=ckeys[2],
                ),
                eqx.nn.Lambda(jnn.relu),
                eqx.nn.AdaptiveMaxPool2d(
                    target_shape=(4, 4),
                ),
                eqx.nn.Lambda(jnp.ravel),
                eqx.nn.Linear(
                    in_features=16 * 4 * 4,
                    out_features=cfg.hdim,
                    key=ckeys[3],
                ),
            ]
        )
    elif cfg.input_block == "fc":
        ib = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(jnp.ravel),
                eqx.nn.Linear(cfg.idim, cfg.hdim, key=key),
            ]
        )
    return ib
