from collections.abc import Callable
from typing import Collection, Literal

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PRNGKeyArray


class MultiHeadMLP(eqx.Module):
    """Multi headed multi-layer perceptron architecture.

    See e.g. Three-headed DragonNet architecture
    See Figure 1 in Shi et al. (2019).

    References
    ----------
    Shi, C., Blei, D., & Veitch, V. (2019). Adapting Neural Networks for the Estimation
    of Treatment Effects. Advances in Neural Information Processing Systems, 32.
    """

    shared_mlp: eqx.nn.MLP
    nonshared_mlps: list[eqx.nn.MLP]

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        width_size: int,
        shared_depth: int,
        nonshared_depths: Collection[int],
        activation: Callable,
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Multiheaded MLP layers.

        Inputs:
            in_size: The input size.
            width_size: The size of each hidden layer in the shared component.
            shared_depth: The number of hidden layers in the shared componenet.
            nonshared_depths: The number of hidden layers for each nonshared componenet.
                The number of outputs is inferred from the number of depths.
            activation: The activation function after each hidden layer.
            use_bias: Whether to add on a bias to internal layers.
            use_final_bias: Whether to add on a bias to the final layers.
            key: A `jax.random.key`. Keyword Only.
        """
        keys = jr.split(key, len(nonshared_depths) + 1)
        self.shared_mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=width_size,
            width_size=width_size,
            depth=shared_depth,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_bias,
            key=keys[0],
        )
        self.nonshared_mlps = [
            eqx.nn.MLP(
                in_size=width_size,
                out_size="scalar",
                width_size=width_size // 2,
                depth=depth,
                activation=activation,
                use_bias=use_bias,
                use_final_bias=use_final_bias,
                key=key,
            )
            for depth, key in zip(nonshared_depths, keys[1:])
        ]

    def __call__(self, x: Array) -> Array:
        _x = self.shared_mlp(x)
        outputs = [mlp(_x) for mlp in self.nonshared_mlps]
        return jnp.hstack(outputs)
