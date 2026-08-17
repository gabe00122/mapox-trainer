"""Tests for Rotary Position Embeddings (RoPE).

RoPE is easy to get wrong (sin/cos split, dimension ordering, dtype).
These tests verify the key mathematical properties rather than exact values.
"""

import jax
import jax.numpy as jnp

from mapox_trainer.model.positional_embeddings import apply_rope


class TestApplyRope:
    def test_output_shape_matches_input(self):
        """RoPE should not change the tensor shape."""
        batch, seq, heads, dim = 2, 8, 4, 16
        inputs = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, heads, dim))
        positions = jnp.arange(seq)[None, :].repeat(batch, axis=0)

        out = apply_rope(inputs, positions, head_dim=dim)
        assert out.shape == inputs.shape

    def test_preserves_dtype(self):
        """Output dtype should match input dtype."""
        for dtype in [jnp.float32, jnp.bfloat16]:
            inputs = jnp.ones((1, 4, 2, 8), dtype=dtype)
            positions = jnp.arange(4)[None, :]
            out = apply_rope(inputs, positions, head_dim=8)
            assert out.dtype == dtype

    def test_preserves_norm(self):
        """RoPE is a rotation — it should preserve vector norms."""
        batch, seq, heads, dim = 2, 8, 4, 16
        inputs = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, heads, dim))
        positions = jnp.arange(seq)[None, :].repeat(batch, axis=0)

        out = apply_rope(inputs, positions, head_dim=dim)

        input_norms = jnp.linalg.norm(inputs, axis=-1)
        output_norms = jnp.linalg.norm(out, axis=-1)
        assert jnp.allclose(input_norms, output_norms, atol=1e-5)

    def test_position_zero_is_identity(self):
        """At position 0, sin=0 and cos=1, so RoPE should be identity."""
        inputs = jax.random.normal(jax.random.PRNGKey(2), (1, 1, 2, 8))
        positions = jnp.zeros((1, 1), dtype=jnp.int32)

        out = apply_rope(inputs, positions, head_dim=8)
        assert jnp.allclose(out, inputs, atol=1e-6)

    def test_different_positions_give_different_outputs(self):
        """Same vector at different positions should produce different embeddings."""
        inputs = jnp.ones((1, 2, 1, 8))
        positions = jnp.array([[0, 5]])

        out = apply_rope(inputs, positions, head_dim=8)
        # The two positions should differ
        assert not jnp.allclose(out[0, 0], out[0, 1], atol=1e-3)

    def test_relative_position_property(self):
        """The dot product between RoPE'd vectors should depend on relative position.

        If q is at position p and k is at position p+d, their dot product
        should be the same regardless of absolute p (for same q, k vectors).
        """
        dim = 16
        q = jax.random.normal(jax.random.PRNGKey(3), (1, 1, 1, dim))
        k = jax.random.normal(jax.random.PRNGKey(4), (1, 1, 1, dim))

        # Case 1: q at pos 2, k at pos 5 (distance = 3)
        q1 = apply_rope(q, jnp.array([[2]]), head_dim=dim)
        k1 = apply_rope(k, jnp.array([[5]]), head_dim=dim)
        dot1 = (q1 * k1).sum()

        # Case 2: q at pos 10, k at pos 13 (distance = 3)
        q2 = apply_rope(q, jnp.array([[10]]), head_dim=dim)
        k2 = apply_rope(k, jnp.array([[13]]), head_dim=dim)
        dot2 = (q2 * k2).sum()

        assert jnp.allclose(dot1, dot2, atol=1e-4)
