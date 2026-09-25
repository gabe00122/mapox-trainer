"""Tests for utility functions.

lerp and format_count are used everywhere — lerp drives entropy coefficient
scheduling, so getting it wrong means wrong exploration behavior.
"""

from typing import Any, cast

import jax.numpy as jnp
import pytest

from mapox_trainer.util import format_count, lerp


class TestLerp:
    def test_at_zero(self):
        assert jnp.allclose(lerp(1.0, 5.0, 0.0), 1.0)

    def test_at_one(self):
        assert jnp.allclose(lerp(1.0, 5.0, 1.0), 5.0)

    def test_midpoint(self):
        assert jnp.allclose(lerp(0.0, 10.0, 0.5), 5.0)

    def test_extrapolation(self):
        """lerp should work beyond [0, 1] — no clamping."""
        assert jnp.allclose(lerp(0.0, 10.0, 2.0), 20.0)

    def test_same_values(self):
        assert jnp.allclose(lerp(3.0, 3.0, 0.7), 3.0)

    def test_jax_arrays(self):
        result = lerp(jnp.array(0.01), jnp.array(0.0), jnp.array(0.5))
        assert jnp.allclose(result, 0.005)


class TestFormatCount:
    def test_small_numbers(self):
        assert format_count(0) == "0"
        assert format_count(999) == "999"

    def test_thousands(self):
        assert format_count(1000) == "1.00K"
        assert format_count(1500) == "1.50K"
        assert format_count(999_999) == "1000.00K"

    def test_millions(self):
        assert format_count(1_000_000) == "1.00M"
        assert format_count(10_500_000) == "10.50M"

    def test_billions(self):
        assert format_count(1_000_000_000) == "1.00B"

    def test_rejects_non_numbers(self):
        with pytest.raises(TypeError):
            format_count(cast(Any, "abc"))
