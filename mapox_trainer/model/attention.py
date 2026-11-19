from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from mapox_trainer.model import positional_embeddings
from mapox_trainer.config import AttentionConfig

class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        config: AttentionConfig,
        *,
        max_seq_length: int,
        attention_impl: Literal["xla", "cudnn"] | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer | None = None,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self._config = config

        if kernel_init is None:
            kernel_init = nnx.initializers.normal()

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.attention_impl: Literal["xla", "cudnn"] = (
            attention_impl
            if attention_impl is not None
            else ("cudnn" if jax.default_backend() == "gpu" else "xla")
        )
        self.dtype = dtype
        self.param_dtype = param_dtype

        if self.d_model % self._config.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by "
                f"'num_heads' heads ({self._config.num_heads})."
            )

        self.qkv_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(self._config.num_heads + 2 * self._config.num_kv_heads, self._config.head_dim),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.out = nnx.LinearGeneral(
            in_features=(self._config.num_heads, self._config.head_dim),
            out_features=self.d_model,
            axis=(-2, -1),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

        if self._config.gate_dim is not None:
            self.gate = nnx.Linear(
                self._config.gate_dim,
                self._config.num_heads,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )


    def initialize_carry(self, batch_size: int, rngs: nnx.Rngs) -> KVCache:
        shape = (batch_size, self.max_seq_length, self._config.num_kv_heads, self._config.head_dim)
        key = jnp.zeros(shape, dtype=self.dtype)
        value = jnp.zeros(shape, dtype=self.dtype)
        return KVCache(key, value)

    def update_kv_cache(self, kv_cache: KVCache, seq_pos, key, value) -> KVCache:
        pos = seq_pos[0, 0] % self.max_seq_length

        key = jax.lax.dynamic_update_slice(kv_cache.key, key, (0, pos, 0, 0))
        value = jax.lax.dynamic_update_slice(kv_cache.value, value, (0, pos, 0, 0))

        return KVCache(key, value)

    def __call__(
        self, inputs: jax.Array, seq_pos: jax.Array, kv_cache: KVCache | None = None
    ) -> tuple[jax.Array, KVCache | None]:
        batch, seq, _ = inputs.shape

        qkv = self.qkv_proj(inputs)

        query = qkv[:, :, : self._config.num_heads]
        key = qkv[:, :, self._config.num_heads : self._config.num_heads + self._config.num_kv_heads]
        value = qkv[:, :, self._config.num_heads + self._config.num_kv_heads :]

        query = positional_embeddings.apply_rope(
            query, seq_pos, self._config.head_dim, self._config.rope_max_wavelength
        )
        key = positional_embeddings.apply_rope(
            key, seq_pos, self._config.head_dim, self._config.rope_max_wavelength
        )

        if kv_cache is not None:
            kv_cache = self.update_kv_cache(kv_cache, seq_pos, key, value)
            key = kv_cache.key
            value = kv_cache.value

            kv_length = jnp.full(
                (batch,), jnp.minimum(seq_pos[0, 0] + 1, self.max_seq_length)
            )
            x = jax.nn.dot_product_attention(
                query,
                key,
                value,
                key_value_seq_lengths=kv_length,
                implementation=self.attention_impl,
            )
        else:
            if self.max_seq_length < seq:
                # sliding window attention
                x = jax.nn.dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                    local_window_size=(self.max_seq_length - 1, 0),
                    implementation=self.attention_impl,
                )
            else:
                x = jax.nn.dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                    implementation=self.attention_impl,
                )

        if hasattr(self, "gate"):
            gate = 2 * jax.nn.sigmoid(
                self.gate(inputs[..., :self._config.gate_dim])
            )
            x = x * gate[..., None]

        out = self.out(x)

        return out, kv_cache
