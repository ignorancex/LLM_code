from typing import Any, Callable, Tuple, Sequence
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from jaxtyping import Array, Float, Bool, PRNGKeyArray
import functools

# from sklearn.datasets import make_classification


class TopKMoeLayer(eqx.Module):
    router: Callable
    experts: Tuple[Callable]
    top_k: int

    def __call__(self, x, biases) -> Float[Array, "..."]:
        probs = self.router(x) + biases
        probs = jnn.softmax(probs, axis=-1)
        top_k_probs, top_k_indices = jax.lax.top_k(probs, self.top_k)
        mask = jnp.zeros_like(probs).at[top_k_indices].set(top_k_probs)
        outs = jnp.array(
            [e(x) for e in self.experts]
        )  # TODO: use vmapped initialization as in equinox https://docs.kidger.site/equinox/tricks/
        weighted_outs = (outs * mask[:, None]).sum(axis=0)
        return weighted_outs, mask, probs


class ThresholdMoeLayer(eqx.Module):
    """
    Mixture of Experts with threshold on the total probability of the experts.
    The number of experts is chosen based on the cumulative probability of the routing predictions.
    The experts are weighted by the routing predictions and the output is a weighted sum of the experts.
    """
    router: Callable
    experts: Tuple[Callable]
    threshold: float

    def syn_net(self, x, biases):
        """
        Decides the routing
        """
        probs = self.router(x) + biases
        probs = jnn.softmax(probs, axis=-1)
        sorting_indices = jnp.argsort(probs, axis=None, descending=True)
        sorted_probs = probs[sorting_indices]
        sorted_probs = jnp.cumsum(sorted_probs)
        magic_idx = jnp.searchsorted(sorted_probs, self.threshold)
        return magic_idx, sorting_indices, probs

    def __call__(self, x, biases, *args, **kwds):
        magic_idx, sorting_indices, probs = self.syn_net(x, biases)
        # Step 1: Get top-k indices
        mask = jnp.arange(len(self.experts)) <= magic_idx
        unsorting_indices = jnp.empty_like(sorting_indices).at[sorting_indices].set(jnp.arange(len(sorting_indices))) # indices that would permute the sorted probailities array back to the original position
        mask = mask[unsorting_indices]
        probs = probs * mask
        weighted_outs = (jnp.array([e(x) for e in self.experts]) * probs[:, None]).sum(axis=0)
        return weighted_outs, mask, probs


class MoE(eqx.Module):
    input_block:eqx.Module
    layers:Sequence[TopKMoeLayer]
    output_block:eqx.Module
    num_layers:int
    def __init__(self, input_block, layers, output_block):
        self.input_block = input_block
        self.layers = layers
        self.output_block = output_block
        self.num_layers = len(self.layers)
    def __call__(self, x, biases):
        masks = []
        probs = []
        x = self.input_block(x)
        for i, l in enumerate(self.layers):
            x, mask, prob = l(x, biases[i])
            masks.append(mask)
            probs.append(prob)
        x = self.output_block(x)
        return x, masks, probs


class MLP(eqx.Module):
    input_block:eqx.Module
    mlp:eqx.nn.MLP
    output_block:eqx.Module
    num_layers:int
    def __init__(
        self,
        input_block:eqx.Module,
        mlp:eqx.nn.MLP,
        output_block:eqx.Module
    ):
        self.input_block = input_block
        self.output_block = output_block
        self.mlp = mlp
        self.num_layers = mlp.depth

    def __call__(self, x, *args, **kwargs):
        x = self.input_block(x)
        x = self.mlp(x)
        x= self.output_block(x)
        return x, [0], [0] # to make it compatible with the training script


def get_experts(n_experts, expert_recipe, keys, *args, **kwargs) -> Tuple[Callable]:
    experts = []
    for k in keys:
        experts.append(expert_recipe(*args, key=k))
    return tuple(experts)


def get_router(dim_h, n_experts, *args, key) -> Callable:
    return eqx.nn.Linear(dim_h, n_experts, *args, key=key)


def get_deepseek_bias(last_batch_utilization, last_biases, k, gamma=0.01, target_utilization=None):
    """
    Updates expert biases based on their utilization in the last batch.

    Args:
        last_batch_utilization (List[Array[float]]): Utilization ratios for each expert.
        last_biases (List[array[float]]): Current bias values for each expert.
        gamma (float): Bias update speed.

    Returns:
        List[float]: Updated bias values for each expert.
    """
    num_experts = len(last_batch_utilization)
    if target_utilization is None:
        target_utilization =  k / num_experts
    updated_biases = []

    for util, bias in zip(last_batch_utilization, last_biases):
        delta = gamma * (target_utilization - util)
        updated_biases.append(bias + delta)

    return updated_biases


def get_deepseek_bias_threshold(last_batch_utilization, last_biases, threshold, gamma=0.01, target_utilization=None):
    """
    Updates expert biases based on their utilization in the last batch.

    Args:
        last_batch_utilization (List[Array[float]]): Utilization ratios for each expert.
        last_biases (List[array[float]]): Current bias values for each expert.
        gamma (float): Bias update speed.

    Returns:
        List[float]: Updated bias values for each expert.
    """
    num_experts = len(last_batch_utilization)
    if target_utilization is None:
        target_utilization =  threshold
    updated_biases = []

    for util, bias in zip(last_batch_utilization, last_biases):
        delta = gamma * (target_utilization - util)
        updated_biases.append(bias + delta)

    return updated_biases


def entropy_loss(probs):

    return - jnp.mean(jax.vmap(jnp.dot)(probs.flatten() + 1e-6, jnp.log(probs.flatten()+1e-6)))


def balancing_loss(
    masks:Bool[Array, "batch_size experts layers"],
    probs:Float[Array, "batch_size experts layers"]
):
    f =jnp.mean(masks, axis=0).flatten()
    q = jnp.mean(probs, axis=0).flatten()
    return len(f) * jnp.dot(f, q)





if __name__ == "__main__":
    def test_topkmoe_layer():
        key = jr.PRNGKey(0)
        x = jr.normal(key, (4, 8))  # batch of 4, input dim 8
        dim_h = x.shape[-1]
        n_experts = 3
        top_k = 2

        # Create dummy experts (simple linear transforms)
        expert_keys = jr.split(key, n_experts)
        experts = get_experts(n_experts, eqx.nn.Linear, expert_keys, dim_h, dim_h)

        # Create router
        router_key = jr.PRNGKey(42)
        router = get_router(dim_h, n_experts, key=router_key)

        layer = TopKMoeLayer(router=router, experts=experts, top_k=top_k)

        out, mask = jax.vmap(layer)(x)

        print("Output:", out)
        print("Mask:", mask)

        assert out.shape == (x.shape[0], x.shape[1]), "Output shape mismatch"
        assert mask.shape == (x.shape[0], n_experts), "Mask shape mismatch"
        assert jnp.allclose(mask.sum(axis=-1), jnp.sum(jax.lax.top_k(jnn.softmax(jax.vmap(router)(x), axis=-1), top_k)[0], axis=-1)), "Mask values do not match top-k probs"

    def test_topkmoe_model():
        key = jr.PRNGKey(1)
        x = jr.normal(key, (4, 8))
        dim_h = x.shape[-1]
        n_experts = 4
        top_k = 2
        depth = 2

        expert_keys = jax.random.split(key, depth * n_experts)
        routers_keys = jax.random.split(key, depth)

        layers = []
        for d in range(depth):
            experts = get_experts(n_experts, eqx.nn.Linear, expert_keys[d*n_experts:(d+1)*n_experts], dim_h, dim_h)
            router = get_router(dim_h, n_experts, key=routers_keys[d])
            layers.append(TopKMoeLayer(router=router, experts=experts, top_k=top_k))

        input_block = eqx.nn.Linear(dim_h, dim_h, key=jr.PRNGKey(10))
        output_block = eqx.nn.Linear(dim_h, dim_h, key=jr.PRNGKey(20))

        model = MoE(input_block=input_block, layers=layers, output_block=output_block)

        dummy_biases = [jnp.zeros(n_experts) for _ in range(depth)]  # unused in forward pass right now
        y, masks = jax.vmap(model, in_axes=(0, None))(x, dummy_biases)

        print("Final output:", y)
        print("Masks:", masks)

        assert y.shape == (x.shape[0], x.shape[1]), "Final output shape mismatch"
        assert len(masks) == depth, "Incorrect number of masks"
        for m in masks:
            assert m.shape == (x.shape[0], n_experts), "Incorrect mask shape in one of the layers"

    def test_deepseek_bias():
        last_utilization = [0.5, 0.2, 0.1]
        last_biases = [0.0, 0.0, 0.0]
        k = 1
        gamma = 0.05

        updated = get_deepseek_bias(last_utilization, last_biases, k, gamma=gamma)

        target = k / len(last_utilization)
        expected = [gamma * (target - u) for u in last_utilization]

        for u, e in zip(updated, expected):
            assert jnp.allclose(u, e), "Bias update mismatch"
        print(updated)
        print("Bias update test passed.")

    def test_threshold_moe():
        dim_h = 10
        input_block = eqx.nn.Linear(dim_h, dim_h, key=jr.PRNGKey(11))
        output_block = eqx.nn.Linear(dim_h, dim_h, key=jr.PRNGKey(20))
        expert_keys = jr.split(jr.key(30), 5)
        experts = get_experts(5, eqx.nn.Linear, expert_keys, dim_h, dim_h)
        router = get_router(dim_h, 5, key = jr.key(69))
        th_moe = ThresholdMoeLayer(
            input_block,
            router,
            output_block,
            experts,
            .5,
        )
        print(th_moe)
        x = jr.normal(key=jr.key(420), shape=32)
        #print(th_moe(x))
        x_batch = jr.normal(key=jr.key(777), shape=(1, dim_h))
        out = jax.vmap(th_moe)(x_batch)
        for o in out:
            print(o)

    def test_aux_losses():
        n_exps = 5
        def get_mask_probs(key):
            logits = jr.uniform(key, (n_exps,))
            probs = jnn.softmax(logits, axis=0)
            print(probs.shape)
            sorted_probs_idx =jnp.argsort(probs)
            print(probs[sorted_probs_idx].shape)
            magic_idx = jnp.searchsorted(jnp.cumsum(probs[sorted_probs_idx]), 0.5)
            mask = mask = jnp.arange(n_exps) >= magic_idx
            unsorting_indices = jnp.empty_like(sorted_probs_idx).at[sorted_probs_idx].set(jnp.arange(len(sorted_probs_idx)))
            mask = mask[unsorting_indices]
            return logits, probs, mask
        keys = jr.split(jr.key(0), 32)
        logits, probs, masks = jax.vmap(get_mask_probs)(keys)
        probs=probs[..., None] 
        masks=masks[..., None]     


        # test entropy loss
        print("entropy loss")
        print(entropy_loss(probs))
        # case with small entropy
        print("entropy loss with small entropy")
        probs_ = jnp.zeros_like(probs).at[:, 0, :].set(0.99)
        probs_ = jnn.softmax(probs_.at[:, 1:].set(jr.normal(keys[0], shape=probs_[:, 1:].shape)  - 10) ,axis=1)
        print(entropy_loss(probs_))
        # case with max entropy
        print("entropy loss with max entropy")
        probs_ = 1/n_exps * jnp.ones_like(probs)
        print(entropy_loss(probs_))
        
        # test balancing loss
        print("balancing loss")
        print("balanced case")
        print(jnp.sum(masks, axis=0))
        print(balancing_loss(masks, probs))
        # highly unbalanced case
        probs_ = jnp.zeros_like(probs).at[:, 0, :].set(0.99)
        probs_ = jnn.softmax(probs_.at[:, 1:].set(jr.normal(keys[0], shape=probs[:, 1:].shape)-10) , axis=1)
        print(masks.shape)
        masks_ = jnp.zeros_like(masks).at[:, 0].set(True)
    
        print("balancing loss, highly unbalanced")
        print(balancing_loss(masks_, probs_))

    # Run all tests
    # test_topkmoe_layer()
    # test_topkmoe_model()
    # test_deepseek_bias()
    # test_threshold_moe()
    test_aux_losses()