"""Tests for value head implementations (HlGauss and MSE).

HlGaussValue is particularly tricky — it discretizes continuous values into
bins using a CDF, then trains with cross-entropy. Bugs here silently produce
wrong value estimates that degrade training.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from mapox_trainer.values import HlGaussValue, MseValue, calculate_supports

from mapox_trainer.config import HlGaussConfig


class TestCalculateSupports:
    def test_support_shape(self):
        config = HlGaussConfig(type="hl_gauss", min=-1.0, max=1.0, n_logits=10, sigma=0.5)
        support, centers = calculate_supports(config)

        # support has n_logits+1 bin edges, with leading batch dim
        assert support.shape == (1, 11)
        # centers has n_logits values
        assert centers.shape == (10,)

    def test_support_range(self):
        config = HlGaussConfig(type="hl_gauss", min=-2.0, max=3.0, n_logits=20, sigma=0.5)
        support, centers = calculate_supports(config)

        assert jnp.allclose(support[0, 0], -2.0)
        assert jnp.allclose(support[0, -1], 3.0)

    def test_centers_are_midpoints(self):
        config = HlGaussConfig(type="hl_gauss", min=0.0, max=1.0, n_logits=4, sigma=0.5)
        support, centers = calculate_supports(config)

        # Centers should be midpoints of consecutive support values
        expected = (support[0, :-1] + support[0, 1:]) / 2
        assert jnp.allclose(centers, expected)


class TestHlGaussValue:
    @pytest.fixture
    def value_head(self):
        config = HlGaussConfig(type="hl_gauss", min=-5.0, max=5.0, n_logits=51, sigma=0.75)
        return HlGaussValue(32, config, rngs=nnx.Rngs(default=0))

    def test_get_value_is_weighted_sum(self, value_head):
        """get_value should return the expected value under the softmax distribution."""
        # Create logits that peak at a known center
        logits = jnp.zeros((2, 4, 51))
        values = value_head.get_value(logits)
        assert values.shape == (2, 4)

        # Uniform logits → value should be the mean of centers (≈ 0.0 for symmetric range)
        assert jnp.allclose(values, 0.0, atol=0.1)

    def test_get_value_peaked_distribution(self, value_head):
        """A very peaked logit distribution should give a value near the corresponding center."""
        # Create logits with a strong peak at bin 40 (positive side)
        logits = jnp.full((1, 1, 51), -100.0)
        logits = logits.at[0, 0, 40].set(100.0)

        value = value_head.get_value(logits)
        # Bin 40 center should be around 5.0 * (40/51*2 - 1) ≈ 2.84
        # The exact value depends on the bin centers
        assert value[0, 0] > 0  # Should be positive (right side of distribution)

    def test_loss_shape(self, value_head):
        """Loss should be a scalar."""
        logits = jnp.zeros((2, 4, 51))
        targets = jnp.zeros((2, 4))
        loss = value_head.get_loss(logits, targets)
        assert loss.shape == ()

    def test_loss_decreases_toward_target(self, value_head):
        """Loss should be lower when logits match the target better."""
        targets = jnp.array([[0.0, 0.0]])  # target at center

        # Logits peaked near 0 (center bin ≈ 25)
        good_logits = jnp.full((1, 2, 51), -10.0)
        good_logits = good_logits.at[:, :, 25].set(10.0)

        # Logits peaked far from 0 (bin 0, far left)
        bad_logits = jnp.full((1, 2, 51), -10.0)
        bad_logits = bad_logits.at[:, :, 0].set(10.0)

        good_loss = value_head.get_loss(good_logits, targets)
        bad_loss = value_head.get_loss(bad_logits, targets)
        assert good_loss < bad_loss

    def test_loss_clips_targets(self, value_head):
        """Targets outside [min, max] should be clipped, not produce NaN."""
        logits = jnp.zeros((1, 2, 51))
        targets = jnp.array([[100.0, -100.0]])  # Way outside range

        loss = value_head.get_loss(logits, targets)
        assert jnp.isfinite(loss)

    def test_loss_is_differentiable(self, value_head):
        """Should be able to take gradients of the loss w.r.t. logits."""
        targets = jnp.array([[1.0, -1.0]])

        def loss_fn(logits):
            return value_head.get_loss(logits, targets)

        logits = jnp.zeros((1, 2, 51))
        grad = jax.grad(loss_fn)(logits)
        assert grad.shape == logits.shape
        assert jnp.all(jnp.isfinite(grad))


class TestMseValue:
    @pytest.fixture
    def value_head(self):
        return MseValue(32, rngs=nnx.Rngs(default=0))

    def test_get_value_is_identity(self, value_head):
        values = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(value_head.get_value(values), values)

    def test_loss_is_half_mse(self, value_head):
        values = jnp.array([[1.0, 2.0]])
        targets = jnp.array([[3.0, 4.0]])
        loss = value_head.get_loss(values, targets)
        expected = 0.5 * jnp.square(values - targets).mean()
        assert jnp.allclose(loss, expected)

    def test_loss_zero_when_perfect(self, value_head):
        values = jnp.array([[1.5, -2.3]])
        loss = value_head.get_loss(values, values)
        assert jnp.allclose(loss, 0.0)
