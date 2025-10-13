# %% Routing by synaptic net
import itertools as its
import math
from typing import Any, Callable, List, Tuple

import einops as ein
import equinox as eqx
import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jr
from jax.tree import map as tmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Scalar

from utils.misc import triplewise


class FiringRateEncoder(eqx.Module):
    backbone: Callable
    num_gates: int
    num_experts: int

    def __init__(
        self,
        in_size: int,
        num_gates: int,
        num_experts: int,
        depth: int = 0,
        *,
        key: PRNGKeyArray,
    ):
        self.backbone = eqx.nn.MLP(
            in_size=in_size,
            out_size=num_gates * num_experts,
            width_size=num_gates * num_experts,
            depth=depth,
            final_activation=jnn.relu,
            key=key,
        )
        self.num_gates = num_gates
        self.num_experts = num_experts

    def __call__(self, h0: Float[Array, " in"], *args, **kwargs) -> Float[Array, "g e"]:
        fr0 = self.backbone(h0, *args, **kwargs)
        return ein.rearrange(fr0, "(g e) -> g e", g=self.num_gates, e=self.num_experts)


# STRAIGHT-THROUGH ESTIMATOR: custom vjp
# --------------------------------------------------------------------------------------
@jax.custom_vjp
def ste(ps: Float[Array, " k"], gs: Float[Array, " k"], temp: Float[Scalar, ""]):
    """Straight through estimator (hopefully)"""
    logits = (jnp.log(ps) + gs) / temp
    return jnn.one_hot(jnp.argmax(logits), len(logits))


def ste_fwd(ps, gs, temp):
    hard = ste(ps, gs, temp)
    safe_logits = jnp.where(ps, (jnp.log(ps) + gs) / temp, 0.0)
    safe_soft = jnn.softmax(safe_logits, where=ps>0.)
    return hard, (safe_logits, safe_soft, temp)


def ste_bwd(res, grads):
    safe_logits, safe_soft, temp = res
    # tempdiff = -(safe_soft / temp) * (safe_logits - jnp.sum(safe_logits * safe_soft))
    return (
        grads * safe_soft - jnp.sum(grads * safe_soft) * safe_soft,
        None,
        #jnp.sum(tempdiff * grads),
        None
    )


ste.defvjp(ste_fwd, ste_bwd)
# --------------------------------------------------------------------------------------


class RoutingNet(eqx.Module):
    fr_encoder: FiringRateEncoder
    wgs: list[Array]
    n_exps: int
    n_layers: int

    def __init__(
        self,
        n_exps: int,
        n_layers: int,
        num_gates: int,
        fr_encoder: FiringRateEncoder,
        *,
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, num=2 * n_layers)
        shapes = list(triplewise([num_gates] + [n_exps for _ in range(n_layers)]))
        wshapes = [(n, o + 1, i) for i, n, o in shapes]
        lims = [math.sqrt(o) for i, n, o in shapes]
        self.wgs = [
            jr.normal(k, wsh) * lim
            for layer, (k, wsh, lim) in enumerate(zip(keys[:n_layers], wshapes, lims))
        ]
        self.fr_encoder = fr_encoder
        self.n_exps = n_exps
        self.n_layers = n_layers

    def _synapses(
        self,
        fr_0: Float[Array, "g e"],
        mask: Float[Array, "l e"],
        *args,
        **kwargs,
    ) -> Tuple[List[Float[Array, "..."]], List[Float[Array, "..."]]]:
        """Compute the firing rates and the synaptic strengths, given the encoded inputs."""
        fr = jnn.softmax(fr_0) / fr_0.shape[0]
        firing_rates, syns = [fr], [fr]
        for wg, m in zip(self.wgs, mask[:-1]):
            p_n = m * ein.reduce(fr, "prev curr -> curr", "sum")
            synstr = jnn.softmax(ein.einsum(wg, fr, "c n p, p c -> c n"))
            synstr = ein.einsum(synstr, m, "curr next, curr -> curr next")
            fr = ein.einsum(synstr, p_n, "curr next, curr -> curr next")
            firing_rates.append(fr)
            syns.append(synstr)
            fr = fr[:, 1:]
        syns.append(mask[-1][:, None])
        firing_rates.append(syns[-1] * ein.reduce(fr, "prev curr -> curr 1", "sum"))
        return firing_rates, syns

    def act_sequence(
        self,
        h_0: Float[Array, " n"],
        temp: float,
        key: PRNGKeyArray,
        *args,
        **kwargs,
    ) -> Any:
        fr_0 = self.fr_encoder(h_0)
        # initialize vector of probabilities: fill the first few elements
        tot_nodes = self.n_exps * self.n_layers + 1
        Pt = jnp.zeros(tot_nodes, dtype=float)
        Pt = Pt.at[: self.n_exps].set(jnn.softmax(fr_0).mean(0))
        mask = jnp.zeros_like(Pt)
        fr_out = 0.0

        def _step(carry, x=None):
            pt, mask, fr_out, key = carry
            key, newkey = jr.split(key)
            # Cfr. https://arxiv.org/pdf/1611.01144
            gs = jr.gumbel(key, shape=pt.shape)
            candidate = ste(pt, gs, temp)
            mask = mask + candidate
            m_ = ein.rearrange(mask[:-1], "(l e) -> l e", l=self.n_layers)
            frs, syns = self._synapses(fr_0, m_)
            fr_out = sum([f[:, 0].sum() for f in frs[1:]])
            fr_first_layer = [frs[0].sum(0)]
            fr_next_layers = tmap(lambda fr: fr[:, 1:].sum(0), frs[1:-1])
            fr_nodes = jnp.concatenate(fr_first_layer + fr_next_layers)
            all_frs = jnp.array((*fr_nodes, fr_out))
            pt = all_frs * (1 - mask)  # same as jnp.where(mask, 0., frs_t)
            carry = (pt, mask, fr_out, newkey)
            outs = (pt, mask, frs, syns, fr_nodes, fr_out)
            return carry, outs

        _, outs = jax.lax.scan(_step, init=(Pt, mask, fr_out, key), length=tot_nodes)
        pt, mask, frs, syns, fr_nodes, fr_out = outs
        # some reshaping
        fr_nodes = ein.rearrange(fr_nodes, "s (l e) -> s l e", l=self.n_layers)
        nodes_mask, out_node_mask = mask[..., :-1], mask[..., -1]
        nodes_mask = ein.rearrange(nodes_mask, "s (l e) -> s l e", l=self.n_layers)
        return (fr_out, fr_nodes, nodes_mask, frs, syns, out_node_mask)


def s_to_f(syns: List[Float[Array, ""]]) -> List[Float[Array, ""]]:
    """Turns synaptic strengths into firing rates."""
    frs = [syns[0]]
    f = frs[-1]
    for s in syns[1:]:
        pn = f.sum(0)
        frs.append(pn[:, None] * s)
        f = frs[-1][:, 1:]
    return frs


if __name__ == "__main__":
    SEED = 3
    RNG = jr.PRNGKey(SEED)
    skey, RNG = jr.split(RNG)
    inkey, RNG = jr.split(RNG)
    IN_SHAPE = 100
    N_EXPS = 5
    N_GATES = 6
    N_LAYERS = 4
    TEMP = 50.

    fr_encoder = FiringRateEncoder(
        in_size=IN_SHAPE,
        num_gates=N_GATES,
        num_experts=N_EXPS,
        depth=0,
        key=skey,
    )
    fake_input = jr.normal(key=inkey, shape=(IN_SHAPE,))
    router = RoutingNet(
        n_exps=N_EXPS,
        n_layers=N_LAYERS,
        num_gates=N_GATES,
        fr_encoder=fr_encoder,
        key=skey,
    )

    fr0 = router.fr_encoder(fake_input)
    means = jnp.ones((N_LAYERS, N_EXPS)) * 0.5
    means = means.at[0, 0].set(1.0)
    mask = jr.bernoulli(RNG, p=means).astype(float)
    frs, syn = router._synapses(fr0, mask)
    for f in frs:
        print(f)

    fr_out, fr_nodes, nodes_mask, frs, syns, out_node_mask = router.act_sequence(
        fake_input, TEMP, skey
    )

    @eqx.filter_value_and_grad
    def dummy_loss(model, x, k):
        stuff = model.act_sequence(x, k)
        fr_out, fr_nodes, *other, out_node_mask = stuff
        magic_idx = jnp.argmax(out_node_mask)
        return jnp.sum(fr_out**2)

    val, grad = dummy_loss(router, fake_input, RNG)
    print(f"Loss: {val}")
    print(grad.wgs)
    print()
