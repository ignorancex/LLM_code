from typing import List

import einops as ein
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar

from moe import WeightedSkipMoE


def l2_aux_loss(
    model: WeightedSkipMoE,
) -> Float[Scalar, ""]:
    """L2 norm on routing network weights."""
    model_params = eqx.filter(
        jax.tree.leaves(model.syn_net), eqx.is_inexact_array, replace=0
    )
    l2_aux_loss = sum(jnp.sum(p**2) for p in model_params)
    return l2_aux_loss


def synaptic_variance_aux_loss(syns: List[Float[Array, "..."]]) -> Float[Scalar, ""]:
    """Synaptic strength variance."""
    syns_var = jax.tree.map(lambda s: jnp.var(s, axis=0), syns)
    syns_var_sum = jax.flatten_util.ravel_pytree(syns_var)[0].sum()
    variance_loss = 1 / (syns_var_sum + 1e-6)
    return variance_loss


def skip_connection_aux_loss(
    syns: List[Float[Array, "..."]],
    actmasks: Bool[Array, "..."],
) -> Float[Scalar, ""]:
    "Promoting skip connections."
    s_skip = jax.tree.map(
        lambda s: jnp.mean(jnp.sum(s[:, :, 1:], axis=2) ** 2), syns[1:-1]
    )
    mean_skip = jnp.sum(jnp.array(s_skip)) / actmasks.shape[2]
    return mean_skip


def importance_aux_loss(
    syns: List[Float[Array, "..."]],
    batch_size: int,
) -> Float[Scalar, ""]:
    # compute importance -> sum over batch
    importance = jax.tree.map(lambda s: jnp.sum(jnp.abs(s), axis=0), syns[:-1])
    std_importance = jnp.std(jax.flatten_util.ravel_pytree(importance)[0])
    cv = std_importance / batch_size
    return cv**2


def throughput_proxy_loss(Fts: Float[Array, "n t d w"]) -> Float[Scalar, ""]:
    fts = ein.rearrange(Fts, "n t d w -> n t (d w)")
    overlaps = ein.reduce(fts[:, None] * fts[None, ...], "n1 n2 t dw -> n1 n2", "sum")
    return jnp.mean(overlaps)
