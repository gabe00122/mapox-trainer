import functools
from collections.abc import Callable

import jax
from flax import nnx
from jax import numpy as jnp
from jax.typing import DTypeLike

from mapox_trainer.config import FeedForwardConfig


def parse_activation_fn(activation_name: str) -> Callable[[jax.Array], jax.Array]:
    match activation_name:
        case "relu":
            return jax.nn.relu
        case "relu2":
                # squared relu (primer, arXiv 2109.08668)
                return lambda x: jnp.square(jax.nn.relu(x))
        case "mish":
            return jax.nn.mish
        case "gelu":
            return jax.nn.gelu
        case "silu":
            return jax.nn.silu
        case _:
            raise ValueError(f"Activation function {activation_name} not recognized")


class FFBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        config: FeedForwardConfig,
        *,
        kernel_init: nnx.Initializer,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ):
        self._config = config
        self.d_model = d_model

        linear = functools.partial(
            nnx.Linear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.activation = parse_activation_fn(config.activation)
        self.up_proj = linear(d_model, config.size, kernel_init=kernel_init)
        self.down_proj = linear(config.size, d_model, kernel_init=nnx.initializers.zeros_init())

    def __call__(self, inputs):
        x = self.up_proj(inputs)
        x = self.activation(x)
        out = self.down_proj(x)
        return out


class GLUBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        config: FeedForwardConfig,
        *,
        kernel_init: nnx.Initializer,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ):
        self._config = config
        self.d_model = d_model

        linear = functools.partial(
            nnx.Linear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.activation = parse_activation_fn(config.activation)
        self.up_proj = linear(d_model, config.size, kernel_init=kernel_init)
        self.up_gate = linear(d_model, config.size, kernel_init=kernel_init,)
        self.down_proj = linear(config.size, d_model, kernel_init=nnx.initializers.zeros_init())

    def __call__(self, inputs):
        x = self.up_proj(inputs)
        gate = self.up_gate(inputs)
        x = self.activation(x) * gate
        out = self.down_proj(x)
        return out
