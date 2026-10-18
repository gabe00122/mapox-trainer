from functools import partial

import jax
from einops import rearrange
from flax import nnx
from flax.nnx.nn.linear import default_embed_init
from jax import numpy as jnp
from mapox import ObservationSpec

from mapox_trainer.config import (
    FlattenedObsEncoderConfig,
    GridCnnObsEncoderConfig,
    LinearObsEncoderConfig,
)
from mapox_trainer.util import FixedParam


# table copies the embedding gradient is spread across before they're summed
_GRAD_COPIES = 64


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def embedding_lookup(table: jax.Array, ids: jax.Array, dtype) -> jax.Array:
    """`table[ids]` cast to `dtype`, with a privatized scatter for its gradient.

    The default gradient of a gather is a scatter-add into the table. Every
    tile of every observation is a lookup and a few ids dominate (floor tiles
    are half of them), so its atomics pile up on a handful of rows. Here
    consecutive lookups add into different copies of the table, which are
    summed afterwards: the contention drops by the number of copies and the
    cost stays independent of the vocabulary size.
    """
    return jnp.take(table, ids, axis=0).astype(dtype)


def _embedding_lookup_fwd(table, ids, dtype):
    return embedding_lookup(table, ids, dtype), (table, ids)


def _embedding_lookup_bwd(dtype, residuals, grad):
    table, ids = residuals
    num_classes, features = table.shape
    ids = ids.reshape(-1).astype(jnp.int32)
    grad = grad.reshape(-1, features).astype(jnp.float32)

    copy = jnp.arange(ids.shape[0], dtype=jnp.int32) % _GRAD_COPIES
    # out-of-range ids get no gradient, as with the gather's own scatter,
    # rather than landing in a neighbouring copy
    slots = jnp.where(
        (ids >= 0) & (ids < num_classes),
        copy * num_classes + ids,
        _GRAD_COPIES * num_classes,
    )
    copies = jnp.zeros((_GRAD_COPIES * num_classes, features), jnp.float32)
    copies = copies.at[slots].add(grad, mode="drop")
    table_grad = copies.reshape(_GRAD_COPIES, num_classes, features).sum(0)
    return table_grad.astype(table.dtype), None


embedding_lookup.defvjp(_embedding_lookup_fwd, _embedding_lookup_bwd)


class LinearObsEncoder(nnx.Module):
    def __init__(
        self,
        config: LinearObsEncoderConfig,
        obs_spec: ObservationSpec,
        output_size: int,
        *,
        dtype,
        params_dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.linear = nnx.Linear(
            obs_spec.shape[0],
            output_size,
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )

    def __call__(self, x) -> jax.Array:
        return self.linear(x)


class GridCnnObsEncoder(nnx.Module):
    def __init__(
        self,
        config: GridCnnObsEncoderConfig,
        obs_spec: ObservationSpec,
        output_size: int,
        *,
        dtype,
        params_dtype,
        rngs: nnx.Rngs,
    ) -> None:
        assert obs_spec.max_value is not None, (
            "max_value must be specified in the observation spec"
        )

        self._config = config
        self.dtype = dtype
        self.params_dtype = params_dtype

        self._one_hot_sizes = obs_spec.max_value
        self.num_classes = (
            int(obs_spec.max_value)
            if isinstance(obs_spec.max_value, int)
            else sum(obs_spec.max_value)
        )

        embeddings = default_embed_init(
            rngs.params(), (self.num_classes, config.embedding_dim), params_dtype
        )
        self.embedding = (
            nnx.Param(embeddings)
            if config.learned_embeddings
            else FixedParam(embeddings)
        )
        channels = [*config.channels, output_size]

        in_channel = config.embedding_dim
        layers = []
        for kernel_size, strides, channel in zip(
            config.kernels, config.strides, channels
        ):
            layers.append(
                nnx.Conv(
                    in_features=in_channel,
                    out_features=channel,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="VALID",
                    dtype=dtype,
                    param_dtype=params_dtype,
                    rngs=rngs,
                )
            )
            in_channel = channel

        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        # only supports one channel for now
        assert x.shape[-1] == 1
        x = embedding_lookup(self.embedding[...], x[..., 0], self.dtype)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.gelu(x)

        x = x.squeeze((-3, -2))

        return x


class FlattenedObsEncoder(nnx.Module):
    def __init__(
        self,
        config: FlattenedObsEncoderConfig,
        obs_spec: ObservationSpec,
        output_size: int,
        *,
        dtype,
        params_dtype,
        rngs: nnx.Rngs,
    ) -> None:
        assert obs_spec.max_value is not None, (
            "max_value must be specified in the observation spec"
        )

        embed_features = 4

        self.params_dtype = params_dtype
        self.num_classes = (
            int(obs_spec.max_value)
            if isinstance(obs_spec.max_value, int)
            else sum(obs_spec.max_value)
        )
        in_features = embed_features * obs_spec.shape[0] * obs_spec.shape[1]

        self.embedding = nnx.Linear(
            self.num_classes,
            embed_features,
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )
        self.dense = nnx.Linear(
            in_features,
            output_size,
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )

    def __call__(self, x) -> jax.Array:
        x = jax.nn.one_hot(x, self.num_classes, dtype=self.params_dtype)
        x = self.embedding(x)
        x = rearrange(x, "... w h c -> ... (w h c)")
        x = self.dense(x)

        return x


def create_obs_encoder(
    config: LinearObsEncoderConfig
    | GridCnnObsEncoderConfig
    | FlattenedObsEncoderConfig,
    obs_spec: ObservationSpec,
    output_size: int,
    *,
    dtype,
    params_dtype,
    rngs: nnx.Rngs,
):
    match config.obs_type:
        case "linear":
            return LinearObsEncoder(
                config,
                obs_spec,
                output_size,
                dtype=dtype,
                params_dtype=params_dtype,
                rngs=rngs,
            )
        case "grid_cnn":
            return GridCnnObsEncoder(
                config,
                obs_spec,
                output_size,
                dtype=dtype,
                params_dtype=params_dtype,
                rngs=rngs,
            )
        case "grid_flattened":
            return FlattenedObsEncoder(
                config,
                obs_spec,
                output_size,
                dtype=dtype,
                params_dtype=params_dtype,
                rngs=rngs,
            )
