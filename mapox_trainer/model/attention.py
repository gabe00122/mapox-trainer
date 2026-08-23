from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from mapox_trainer.model import positional_embeddings


class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        max_seq_length: int,
        rope_max_wavelength: float = 10_000,
        use_qk_norm: bool = False,
        attention_impl: str | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer = nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_qk_norm = use_qk_norm
        self.max_seq_length = max_seq_length
        self.rope_max_wavelength = rope_max_wavelength
        if attention_impl is None:
            self.attention_impl = "cudnn" if jax.default_backend() == "gpu" else "xla"
        else:
            self.attention_impl = attention_impl
        self.dtype = dtype
        self.param_dtype = param_dtype

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.qkv_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(self.num_heads + 2 * self.num_kv_heads, self.head_dim),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.out = nnx.LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.d_model,
            axis=(-2, -1),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        if self.use_qk_norm:
            self._query_norm = nnx.RMSNorm(
                self.head_dim, dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
            )
            self._key_norm = nnx.RMSNorm(
                self.head_dim, dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
            )

    def initialize_carry(self, batch_size: int, rngs: nnx.Rngs) -> KVCache:
        shape = (batch_size, self.max_seq_length, self.num_kv_heads, self.head_dim)
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

        query = qkv[..., :self.num_heads]
        key = qkv[..., self.num_heads:self.num_heads + self.num_kv_heads]
        value = qkv[..., self.num_heads + self.num_kv_heads:]

        if self.use_qk_norm:
            query = self._query_norm(query)
            key = self._key_norm(key)

        query = positional_embeddings.apply_rope(
            query, seq_pos, self.head_dim, self.rope_max_wavelength
        )
        key = positional_embeddings.apply_rope(
            key, seq_pos, self.head_dim, self.rope_max_wavelength
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

        out = self.out(x)

        return out, kv_cache
