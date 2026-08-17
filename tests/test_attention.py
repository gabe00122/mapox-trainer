"""Tests for the attention block and KV cache.

KV cache bugs are insidious — they can cause the model to attend to stale
or wrong positions during inference while training looks fine.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from mapox_trainer.model.attention import AttentionBlock


def make_attention(
    d_model=32,
    head_dim=8,
    num_heads=4,
    num_kv_heads=4,
    max_seq_length=16,
    **kwargs,
) -> AttentionBlock:
    return AttentionBlock(
        d_model=d_model,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_length=max_seq_length,
        attention_impl="xla",  # Use XLA for testing (no CuDNN required)
        rngs=nnx.Rngs(default=0),
        **kwargs,
    )


class TestAttentionBlock:
    def test_output_shape(self):
        attn = make_attention()
        x = jnp.ones((2, 8, 32))
        pos = jnp.arange(8)[None, :].repeat(2, axis=0)

        out, _ = attn(x, pos)
        assert out.shape == (2, 8, 32)

    def test_causal_masking(self):
        """Earlier positions should not be affected by later positions.

        If we change a token at position t, positions < t should stay the same.
        """
        attn = make_attention()
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 32))
        pos = jnp.arange(8)[None, :]

        out1, _ = attn(x, pos)

        # Modify the last token
        x_modified = x.at[0, 7, :].set(99.0)
        out2, _ = attn(x_modified, pos)

        # Positions 0-6 should be identical (causal: can't see position 7)
        assert jnp.allclose(out1[0, :7], out2[0, :7], atol=1e-5)
        # Position 7 should differ
        assert not jnp.allclose(out1[0, 7], out2[0, 7], atol=1e-3)

    def test_grouped_query_attention(self):
        """GQA with fewer KV heads should still produce correct shapes."""
        attn = make_attention(num_heads=4, num_kv_heads=2)
        x = jnp.ones((2, 4, 32))
        pos = jnp.arange(4)[None, :].repeat(2, axis=0)

        out, _ = attn(x, pos)
        assert out.shape == (2, 4, 32)

    def test_single_kv_head(self):
        """Multi-query attention (1 KV head) should work."""
        attn = make_attention(num_heads=4, num_kv_heads=1)
        x = jnp.ones((1, 4, 32))
        pos = jnp.arange(4)[None, :]

        out, _ = attn(x, pos)
        assert out.shape == (1, 4, 32)


class TestKVCache:
    def test_initialize_carry_shape(self):
        attn = make_attention(max_seq_length=16, num_kv_heads=2, head_dim=8)
        cache = attn.initialize_carry(batch_size=3, rngs=nnx.Rngs(default=0))

        assert cache.key.shape == (3, 16, 2, 8)
        assert cache.value.shape == (3, 16, 2, 8)

    def test_kv_cache_stores_at_correct_position(self):
        attn = make_attention(max_seq_length=8)
        cache = attn.initialize_carry(batch_size=1, rngs=nnx.Rngs(default=0))

        # Simulate writing at position 3
        new_k = jnp.ones((1, 1, 4, 8))  # batch=1, seq=1, heads=4, dim=8
        new_v = jnp.ones((1, 1, 4, 8)) * 2.0

        seq_pos = jnp.array([[3]])
        updated = attn.update_kv_cache(cache, seq_pos, new_k, new_v)

        # Position 3 should have the new values
        assert jnp.allclose(updated.key[0, 3], 1.0)
        assert jnp.allclose(updated.value[0, 3], 2.0)
        # Position 0 should still be zeros
        assert jnp.allclose(updated.key[0, 0], 0.0)

    def test_kv_cache_wraps_around(self):
        """Positions beyond max_seq_length should wrap (modulo)."""
        attn = make_attention(max_seq_length=4)
        cache = attn.initialize_carry(batch_size=1, rngs=nnx.Rngs(default=0))

        new_k = jnp.ones((1, 1, 4, 8)) * 5.0
        new_v = jnp.ones((1, 1, 4, 8)) * 5.0

        # Position 6 should wrap to position 6 % 4 = 2
        seq_pos = jnp.array([[6]])
        updated = attn.update_kv_cache(cache, seq_pos, new_k, new_v)

        assert jnp.allclose(updated.key[0, 2], 5.0)
        assert jnp.allclose(updated.key[0, 0], 0.0)  # Other positions untouched

    def test_inference_matches_training_first_step(self):
        """First inference step should match the first position of a full forward pass."""
        attn = make_attention(d_model=32, max_seq_length=8)
        x_full = jax.random.normal(jax.random.PRNGKey(1), (1, 1, 32))
        pos_full = jnp.array([[0]])

        # Training mode (no cache)
        out_train, _ = attn(x_full, pos_full, kv_cache=None)

        # Inference mode (with cache)
        cache = attn.initialize_carry(batch_size=1, rngs=nnx.Rngs(default=0))
        out_infer, _ = attn(x_full, pos_full, kv_cache=cache)

        assert jnp.allclose(out_train, out_infer, atol=1e-5)
