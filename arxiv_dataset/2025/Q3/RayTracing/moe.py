# Layer-wise vectorized MoEs
import functools as fts
from collections.abc import Callable
from typing import List, Sequence, Tuple

import einops as ein
import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jax.tree import map as tmap
from jaxtyping import Array, Float, PRNGKeyArray

from routing_net import FiringRateEncoder, RoutingNet

UNROLL = True  # set to True for better perf

def make_expert_vgrid(
    expert_recipe: Callable,
    n_layers: int,
    n_experts: int,
    key: PRNGKeyArray,
) -> Tuple[eqx.Module, eqx.Module]:
    """Produces a pair of PyTrees, one with the dynamic part of the network
    (i.e, the weights) and the other with the static part of the network (i.e.,
    functions and so on).
    """
    exp_keys = jr.split(key, n_experts * n_layers)
    exp_keys = ein.rearrange(exp_keys, "(l e) k -> l e k", l=n_layers, e=n_experts)
    experts = eqx.filter_vmap(eqx.filter_vmap(expert_recipe))(exp_keys)
    return eqx.partition(experts, eqx.is_array)


class RTMoE(eqx.Module):
    routing_net: RoutingNet
    expert_params: eqx.Module
    expert_struct: eqx.Module
    res: bool

    def expert_net(
        self,
        h_0: Float[Array, "c ..."],
        softmask: Float[Array, "l e"],
    ) -> Float[Array, "le c ..."]:
        # define layerwise pass
        def _layer(H: Float[Array, "e c ..."], pnm):
            params, mask = pnm
            mod = eqx.combine(params, self.expert_struct)
            outs = eqx.filter_vmap(lambda m: m(H))(mod)
            H = ein.einsum(outs, mask, "e c ..., e -> c ...")
            return H, outs

        xs = (self.expert_params, softmask)
        _, outs = jax.lax.scan(_layer, init=h_0, xs=xs, unroll=UNROLL)
        h_out = ein.einsum(outs, softmask, "l e c ..., l e -> c ...")
        return h_out + self.res * h_0

    def act_sequence(self, h_0, temp, key, *args):
        return self.routing_net.act_sequence(h_0, temp, key, *args)

    def __call__(
        self,
        h_0: Float[Array, "c ..."],
        temp: float,
        key: PRNGKeyArray,
        *args,
    ) -> Float[Array, "c ..."]:
        sequence = self.routing_net.act_sequence(h_0, temp, key)
        fr_out, fr_nodes, nodes_mask, frs, syns, out_node_mask = sequence
        magic_idx = jnp.argmax(out_node_mask, axis=-1)
        masks = nodes_mask[magic_idx]
        return self.expert_net(h_0, masks)


def get_out_shapes(
    experts: Sequence[eqx.Module], dummy_inputs: List
) -> List[Tuple[int]]:
    """Takes experts, dummy inputs and computes their output shapes. Useful"""
    dyn_exps, stat_exps = eqx.partition(experts, eqx.is_array)
    out_shapes = []
    for d, s, i in zip(dyn_exps, stat_exps, dummy_inputs):
        f = fts.partial(lambda d, i, s: eqx.combine(d, s)(*i), s=s)
        out_shapes.append(jax.eval_shape(eqx.filter_vmap(f), d, i).shape)
    return out_shapes


def construct_shape_diagram(
    experts: Sequence[eqx.Module], context: Float[Array, "e ..."]
) -> List[Tuple[int]]:
    """To be used to interactively find the shapes of the experts."""
    dyn_exps, stat_exps = eqx.partition(experts, eqx.is_array)
    out_shapes = []
    dummy_input = jnp.zeros_like(context)
    i = (dummy_input, context)
    for d, s in zip(dyn_exps, stat_exps):
        f = fts.partial(lambda d, i, s: eqx.combine(d, s)(*i), s=s)
        out_shapes.append(jax.eval_shape(eqx.filter_vmap(f), d, i).shape)
        # here we are concatenating the expert axis (0) with the channel axis (1)
        # and leaving the "structure" (i.e., widht and height for CNNs) untouched
        single_expert_shape = (
            out_shapes[-1][0] * out_shapes[-1][1],
            *out_shapes[-1][2:],
        )
        i = (jnp.zeros((context.shape[0], *single_expert_shape)), context)
    return out_shapes


if __name__ == "__main__":
    SEED = 1
    RNG = jr.PRNGKey(SEED)
    N_EXPS = 8
    N_LAYERS = 4
    CDIM = 16  # common dimension
    EXP_WIDTH = 512
    EXP_DEPTH = 2
    NUM_GATES = N_EXPS + 1

    # ----------------------------------------------------------------------------------
    # Test: does the code run?
    RNG, frkey, rkey = jr.split(RNG, num=3)
    fr_encoder = FiringRateEncoder(
        in_size=CDIM,
        num_gates=NUM_GATES,
        num_experts=N_EXPS,
        depth=0,
        key=frkey,
    )
    router = RoutingNet(
        n_exps=N_EXPS,
        n_layers=N_LAYERS,
        num_gates=NUM_GATES,
        fr_encoder=fr_encoder,
        key=rkey,
    )

    def make_expert(k: PRNGKeyArray) -> eqx.nn.MLP:
        return eqx.nn.MLP(CDIM, CDIM, EXP_WIDTH, EXP_DEPTH, key=k)

    RNG, exp_key = jr.split(RNG)
    exp_params, exp_struct = make_expert_vgrid(make_expert, N_LAYERS, N_EXPS, exp_key)
    rtmoe = RTMoE(
        routing_net=router,
        expert_params=exp_params,
        expert_struct=exp_struct,
        res=True,
    )

    h_0 = jnp.ones(CDIM)
    h_out = rtmoe(h_0, RNG)
    assert len(h_out.shape) == 1
    assert h_out.shape[0] == CDIM
    # ----------------------------------------------------------------------------------
