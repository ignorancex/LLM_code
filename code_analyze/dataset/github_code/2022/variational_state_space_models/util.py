import wandb
import jax.numpy as jnp
import jax
import contextlib
import argparse
import airsim

from haiku._src.stateful import temporary_internal_state, \
        internal_state, update_internal_state, InternalState, difference as state_diff
from jax.tree_util import tree_map, tree_multimap, tree_flatten
import functools

def first(x):
    return tree_map(lambda x: x[0], x)

def environment_setup():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.config.experimental.set_visible_devices([], "GPU")

def gallery(array, ncols=4):
    array = array
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

from haiku._src.data_structures import FlatMap

def recursive_items(obj):
    if hasattr(obj, '_asdict'):
        obj = obj._asdict()
    if isinstance(obj, FlatMap):
        obj = obj._to_mapping()

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            for (sk, sv) in recursive_items(v):
                if sk is None:
                    yield (f'{i}', sv)
                else:
                    yield (f'{i}/{sk}', sv)
    elif isinstance(obj, dict):
        for (k, v) in obj.items():
            for (sk, sv) in recursive_items(v):
                if sk is None:
                    yield (k, sv)
                else:
                    yield (f'{k}/{sk}', sv)
    elif obj is None:
        return
    else:
        yield (None, obj)

# static_unroll, but with reverse support
def static_unroll(core, input_sequence, initial_state, time_major=True, reverse=False):
  output_sequence = []
  time_axis = 0 if time_major else 1
  num_steps = jax.tree_leaves(input_sequence)[0].shape[time_axis]
  state = initial_state
  for t in range(num_steps - 1, -1, -1) if reverse else range(num_steps):
    if time_major:
      inputs = jax.tree_map(lambda x, _t=t: x[_t], input_sequence)
    else:
      inputs = jax.tree_map(lambda x, _t=t: x[:, _t], input_sequence)
    outputs, state = core(inputs, state)
    output_sequence.append(outputs)

  # Stack outputs along the time axis.
  output_sequence = jax.tree_multimap(
      lambda *args: jnp.stack(args, axis=time_axis),
      *output_sequence)
  return output_sequence, state

from jax._src.api import _mapped_axis_size, flatten_axes
from jax.interpreters import batching
from haiku._src.stateful import internal_state, update_internal_state, difference

# vmap_rng auto-splits the rng based on the relevant batch dimension
def vmap_rng(fun, in_axes=0, out_axes=0, axis_name=None):
    """Equivalent to :func:`jax.vmap` with module parameters/state not mapped."""

    # TODO(tomhennigan): Allow configuration of params/state/rng mapping.
    in_axes_orig = in_axes
    in_axes = 0, in_axes, None
    out_axes = out_axes, None

    @functools.wraps(fun)
    def pure_fun(rng, args, state_in):
        with temporary_internal_state(state_in):
            out = fun(rng, *args)
            state_out = difference(state_in, internal_state())
            return out, state_out

    mapped_pure_fun = jax.vmap(pure_fun, in_axes=in_axes, out_axes=out_axes,
                                axis_name=axis_name)
    @functools.wraps(fun)
    def mapped_fun(rng, *args):
        args_flat, in_tree  = tree_flatten((args, {}))
        in_axes_flat = flatten_axes("vmap in_axes", in_tree, (in_axes_orig, 0), kws=True)
        axis_size_ = _mapped_axis_size(in_tree, args_flat, 
                            in_axes_flat, "vmap", kws=True)

        rngs = jax.random.split(rng, axis_size_)
        state = internal_state()
        out, state = mapped_pure_fun(rngs, args, state)
        update_internal_state(state)
        return out

    return mapped_fun


def plot_mean_std(samples, title):
    means_ar = jnp.mean(samples, tuple(range(samples.ndim - 1)))
    stds_ar = jnp.sqrt(jnp.square(samples - means_ar).mean(tuple(range(samples.ndim - 1))))

    means = [[str(i), v] for i, v in enumerate(means_ar)]
    stds = [[str(i), v] for i, v in enumerate(stds_ar)]

    means = wandb.Table(data=means, columns=['dim', 'mean'])
    stds = wandb.Table(data=stds, columns=['dim', 'std'])

    means = wandb.plot.bar(means, "dim", "mean", title=f"{title} Mean")
    stds = wandb.plot.bar(stds, "dim", "std", title=f"{title} Std")

    return means, stds, means_ar, stds_ar

def increase_contrast(img, factor=2):
    mean = jnp.mean(img, tuple(range(img.ndim - 1))) # find per-channel averages
    diff = img - mean
    diff = factor*diff
    return jnp.clip(mean + diff, 0, 1)

def grad_norm(grad, norm_type):
    grad_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(jnp.ravel(x), norm_type), grad)
    norms = jnp.array(jax.tree_util.tree_leaves(grad_norms))
    return jnp.linalg.norm(jnp.ravel(norms), norm_type)

@contextlib.contextmanager
def disable_params_grad():
    orig_state = internal_state()
    tmp = InternalState(params=tree_map(jax.lax.stop_gradient, orig_state.params),
            state=orig_state.state, rng=orig_state.rng)

    with temporary_internal_state(tmp):
        yield
        new_state = internal_state()

    # patch back to remove the gradients
    # so it doesn't get picked up by the diff()
    new_params = tree_multimap(
        lambda orig, ng, new: orig if ng is new else new,
        orig_state.params, tmp.params, new_state.params)

    new_state = InternalState(params=new_params,
        state=new_state.state, rng=new_state.rng)
    diff = state_diff(new_state, orig_state)
    update_internal_state(diff)

from jax import custom_vjp

@custom_vjp
def clip_gradient(lo, hi, x):
    return x  # identity function

def clip_gradient_fwd(lo, hi, x):
    return x, (lo, hi)  # save bounds as residuals

def clip_gradient_bwd(res, g):
  lo, hi = res
  return (None, None, jnp.clip(g, lo, hi))  # use None to indicate zero cotangents for lo and hi

@custom_vjp
def scale_gradient(factor, x):
    return x  # identity function

def scale_gradient_fwd(factor, x):
    return x, factor  # save bounds as residuals

def scale_gradient_bwd(res, g):
  factor = res
  return (None, factor*g) # No gradient wrt factor

@custom_vjp
def scale_gradient_norm(factor, x):
    return x  # identity function

def scale_gradient_norm_fwd(factor, x):
    return x, factor  # save bounds as residuals

def scale_gradient_norm_bwd(res, g):
  factor = res
  return (None, safe_clip_grads(g, factor))

from jax.example_libraries.optimizers import l2_norm

def safe_clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  eps = 1e-9
  normalize = lambda g: jnp.where(norm < max_norm, g, g * max_norm / (norm + eps))
  return tree_map(normalize, grad_tree)

from jax.experimental.host_callback import id_print

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)
scale_gradient.defvjp(scale_gradient_fwd, scale_gradient_bwd)
scale_gradient_norm.defvjp(scale_gradient_norm_fwd, scale_gradient_norm_bwd)

from optax._src.base import GradientTransformation
from optax._src.alias import _scale_by_learning_rate
from optax import chain
from typing import NamedTuple, Any

class AdadeltaState(NamedTuple):
    sq_avg: Any
    acc_delta: Any 

def scale_by_adadelta(rho = 0.9, eps=1e-6):
    def init_fn(params):
        sq_avg = jax.tree_map(lambda t: jnp.full_like(t, 0), params)
        acc_delta = jax.tree_map(lambda t: jnp.full_like(t, 0), params)
        return AdadeltaState(sq_avg, acc_delta)

    def update_fn(updates, state, params=None):
        del params
        sq_avg = jax.tree_multimap(lambda s, u: rho*s + jnp.square(u)*(1-rho), state.sq_avg, updates)
        delta = jax.tree_multimap(lambda d, s, u: (jnp.sqrt(d + eps)/jnp.sqrt(s + eps))*u, state.acc_delta, sq_avg, updates)
        acc_delta = jax.tree_multimap(lambda a, d: rho*a + jnp.square(d)*(1-rho), state.acc_delta, delta)
        new_state = AdadeltaState(sq_avg, acc_delta)
        return delta, new_state
    
    return GradientTransformation(init_fn, update_fn)

def optim_adadelta(learning_rate=1, rho=0.9, eps=1e-6):
    return chain(
        scale_by_adadelta(rho=rho, eps=eps),
        _scale_by_learning_rate(learning_rate)
    )