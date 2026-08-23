from flax.nnx.nn.linear import default_embed_init
import jax
from jax import numpy as jnp
from einops import rearrange
from flax import nnx
from mapox import ObservationSpec

from mapox_trainer.config import (
    FlattenedObsEncoderConfig,
    GridCnnObsEncoderConfig,
    LinearObsEncoderConfig,
)


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

        embeddings = default_embed_init(rngs.params(), (self.num_classes, config.embedding_dim), params_dtype)
        self.embedding = nnx.Param(embeddings) if config.learned_embeddings else nnx.Variable(embeddings)
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
        x = jnp.take(self.embedding[:], x[..., 0], axis=0)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.gelu(x)

        x = rearrange(x, "... w h c -> ... (w h c)")

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


class GridCnnObsDecoder(nnx.Module):
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

        self.dtype = dtype
        self.num_classes = self.num_classes = (
            int(obs_spec.max_value)
            if isinstance(obs_spec.max_value, int)
            else sum(obs_spec.max_value)
        )
        self.conv2 = nnx.ConvTranspose(
            in_features=output_size,
            out_features=16,
            kernel_size=(3, 3),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )
        self.conv1 = nnx.ConvTranspose(
            in_features=16,
            out_features=self.num_classes,
            kernel_size=(3, 3),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )

    def __call__(self, x) -> jax.Array:
        x = rearrange(x, "... c -> ... 1 1 c")

        x = self.conv2(x)
        x = jax.nn.gelu(x)
        x = self.conv1(x)

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
