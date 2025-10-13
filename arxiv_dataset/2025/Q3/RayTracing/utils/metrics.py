import einops as ein
import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float

import wandb


def get_parameter_count(model: eqx.Module) -> int:
    """Produces a parameter count for a given model."""
    params = jax.tree.leaves(model)
    params = eqx.filter(params, eqx.is_inexact_array, replace=jnp.empty(0))
    return sum(x.size for x in params)


def make_wandb_histlike(metric, bins, range=None):
    """Creates a wandb.Table which can then be used to display a hist-like bar
    plot. NB: this is not really a histogram, but a bar plot. For our use case,
    it is fine enough."""
    counts, bins = np.histogram(metric, bins=bins, range=range, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    table = wandb.Table(
        columns=["bin", "count"], data=[[bc, c] for bc, c in zip(bin_centers, counts)]
    )
    return table


def expert_overlap_min(act_mask: Bool[np.ndarray, "batch experts"]) -> Float[Array, ""]:
    def _ov(row):
        num = (row[None, :] * act_mask).sum(-1)
        den = jnp.minimum(row[None, :].sum(-1), act_mask.sum(-1))
        return num / den

    idxs = jnp.triu_indices(len(act_mask), 1)
    return jax.lax.map(_ov, act_mask)[idxs]


def expert_overlap_max(act_mask: Bool[np.ndarray, "batch experts"]) -> Float[Array, ""]:
    def _ov(row):
        num = (row[None, :] * act_mask).sum(-1)
        den = jnp.maximum(row[None, :].sum(-1), act_mask.sum(-1))
        return num / den

    idxs = jnp.triu_indices(len(act_mask), 1)
    return jax.lax.map(_ov, act_mask)[idxs]


def efficient_frontier(actives, preds, true):
    """Computes the efficient frontier, i.e., accuracy vs num experts."""
    pred_is_correct = (preds == true).astype(float)
    idxs_sorted = actives.argsort()
    num_actives = np.unique(actives)
    splits = np.split(
        pred_is_correct[idxs_sorted],
        np.unique(pred_is_correct[idxs_sorted], return_index=True)[1][1:],
    )
    splits_means = np.array([np.mean(s) for s in splits])
    splits_stds = np.array([np.std(s) for s in splits])
    return np.stack((num_actives, splits_means, splits_stds))


def img_table(act_counts, imgs, labs, preds):
    n_img_to_log = 5
    img_log_columns = ["image", "true_label", "pred_label", "n_exp"]
    table = wandb.Table(columns=img_log_columns)
    easy_to_hard = jnp.argsort(act_counts.astype(int))
    easy_idxs = easy_to_hard[:n_img_to_log]
    hard_idxs = easy_to_hard[-n_img_to_log:]
    imgs_to_plot = jnp.concat((imgs[hard_idxs], imgs[easy_idxs]))
    labs_to_plot = jnp.concat((labs[hard_idxs], labs[easy_idxs]))
    predictions = jnp.concat((preds[hard_idxs], preds[easy_idxs]))
    num_experts = jnp.concat([act_counts[hard_idxs], act_counts[easy_idxs]])
    for img, true, pred, nexp in zip(
        imgs_to_plot, labs_to_plot, predictions, num_experts
    ):
        row = [
            wandb.Image(np.array(img)),
            true.item(),
            pred.item(),
            nexp.item(),
        ]
        table.add_data(*row)
    return table


if __name__ == "__main__":
    seed = 0
    rng = jr.PRNGKey(seed)

    b, e = 81, 9

    # overlap computations for random mask
    experts = jr.uniform(key=rng, shape=(b, e)) > 0.5
    overlap_min = expert_overlap_min(experts)
    overlap_max = expert_overlap_max(experts)
    print("Overlap on random usage")
    print(f"Max overlap (mean): {jnp.mean(overlap_min):.3f}")
    print(f"Min Overlap (mean): {jnp.mean(overlap_max):.3f}")
    print()

    # theoretical maximum overlap
    experts = jr.uniform(key=rng, shape=(e,)) > 0.5
    experts = ein.repeat(experts, "e -> b e", b=100)
    overlap_max = expert_overlap_max(experts)
    overlap_min = expert_overlap_min(experts)
    print("Theoretical max overlap (should be 1)")
    print(f"Max overlap (mean): {jnp.mean(overlap_min):.3f}")
    print(f"Min Overlap (mean): {jnp.mean(overlap_max):.3f}")
    print()

    # minimum overlap: should be 0 (ideally), but really b/e
    row1 = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    expert_batch = jax.vmap(jnp.roll, in_axes=(None, 0))(row1, jnp.arange(len(row1)))
    experts = ein.repeat(expert_batch, "b e -> (b m) e", m=b // e)
    overlap_max = expert_overlap_max(experts)
    overlap_min = expert_overlap_min(experts)
    print(f"Around min. overlap (should be roughly 0, or {1 / 9:.3f})")
    print(f"Max overlap (mean): {jnp.mean(overlap_min):.3f}")
    print(f"Min Overlap (mean): {jnp.mean(overlap_max):.3f}")
    print()
