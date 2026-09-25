"""Tests for value head implementations (HlGauss and MSE).

HlGauss is particularly tricky — it discretizes continuous values into bins
using a CDF, then trains with cross-entropy. Bugs here silently produce wrong
value estimates that degrade training.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from mapox_trainer.config import HlGaussConfig
from mapox_trainer.model.value import (
    HlGaussHead,
    HlGaussValueRepresentation,
    MseHead,
    MseValueRepresentation,
    calculate_supports,
)


class TestCalculateSupports:
    def test_support_shape(self):
        config = HlGaussConfig(
            type="hl_gauss", min=-1.0, max=1.0, n_logits=10, sigma=0.5
        )
        support, centers = calculate_supports(config)

        # support has n_logits+1 bin edges, with leading batch dim
        assert support.shape == (1, 11)
        assert centers.shape == (10,)

    def test_support_range(self):
        config = HlGaussConfig(
            type="hl_gauss", min=-2.0, max=3.0, n_logits=20, sigma=0.5
        )
        support, centers = calculate_supports(config)

        assert jnp.allclose(support[0, 0], -2.0)
        assert jnp.allclose(support[0, -1], 3.0)
        assert len(centers) == config.n_logits

    def test_centers_are_midpoints(self):
        config = HlGaussConfig(type="hl_gauss", min=0.0, max=1.0, n_logits=4, sigma=0.5)
        support, centers = calculate_supports(config)

        # Centers should be midpoints of consecutive support values
        expected = (support[0, :-1] + support[0, 1:]) / 2
        assert jnp.allclose(centers, expected)


@pytest.fixture
def hl_gauss_config():
    return HlGaussConfig(type="hl_gauss", min=-5.0, max=5.0, n_logits=51, sigma=0.75)


class TestHlGaussValueRepresentation:
    def test_value_is_weighted_sum(self, hl_gauss_config):
        """value() should return the expected value under the softmax distribution."""
        # Uniform logits -> value should be the mean of centers (0.0 here)
        logits = jnp.zeros((2, 4, 51))
        value = HlGaussValueRepresentation(hl_gauss_config, logits).value()
        assert value.shape == (2, 4)
        assert jnp.allclose(value, 0.0, atol=1e-3)

    def test_value_peaked_distribution(self, hl_gauss_config):
        """A very peaked distribution should give a value near its center."""
        # Strong peak on the right side of the range
        logits = jnp.full((1, 1, 51), -100.0)
        logits = logits.at[0, 0, 40].set(100.0)

        value = HlGaussValueRepresentation(hl_gauss_config, logits).value()
        assert value[0, 0] > 0
        assert jnp.allclose(value[0, 0], calculate_supports(hl_gauss_config)[1][40])

    def test_loss_shape(self, hl_gauss_config):
        """Loss is per (batch, time) element, not reduced."""
        logits = jnp.zeros((2, 4, 51))
        targets = jnp.zeros((2, 4))
        loss = HlGaussValueRepresentation(hl_gauss_config, logits).loss(targets)
        assert loss.shape == (2, 4)

    def test_loss_decreases_toward_target(self, hl_gauss_config):
        """Loss should be lower when logits put mass on the target bin."""
        targets = jnp.zeros((1, 2))  # target at the center bin (25)

        good_logits = jnp.full((1, 2, 51), -10.0)
        good_logits = good_logits.at[:, :, 25].set(10.0)

        bad_logits = jnp.full((1, 2, 51), -10.0)
        bad_logits = bad_logits.at[:, :, 0].set(10.0)

        good_loss = HlGaussValueRepresentation(hl_gauss_config, good_logits).loss(
            targets
        )
        bad_loss = HlGaussValueRepresentation(hl_gauss_config, bad_logits).loss(targets)
        assert good_loss.mean() < bad_loss.mean()

    def test_loss_clips_targets(self, hl_gauss_config):
        """Targets outside [min, max] should be clipped, not produce NaN."""
        logits = jnp.zeros((1, 2, 51))
        targets = jnp.array([[100.0, -100.0]])  # Way outside range

        loss = HlGaussValueRepresentation(hl_gauss_config, logits).loss(targets)
        assert jnp.all(jnp.isfinite(loss))

    def test_loss_is_differentiable(self, hl_gauss_config):
        """Should be able to take gradients of the loss w.r.t. logits."""
        targets = jnp.array([[1.0, -1.0]])

        def loss_fn(logits):
            return (
                HlGaussValueRepresentation(hl_gauss_config, logits).loss(targets).mean()
            )

        logits = jnp.zeros((1, 2, 51))
        grad = jax.grad(loss_fn)(logits)
        assert grad.shape == logits.shape
        assert jnp.all(jnp.isfinite(grad))


class TestMseValueRepresentation:
    def test_value_is_identity(self):
        values = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(MseValueRepresentation(values).value(), values)

    def test_loss_is_half_squared_error(self):
        values = jnp.array([[1.0, 2.0]])
        targets = jnp.array([[3.0, 4.0]])
        loss = MseValueRepresentation(values).loss(targets)
        assert jnp.allclose(loss, 0.5 * jnp.square(values - targets))

    def test_loss_zero_when_perfect(self):
        values = jnp.array([[1.5, -2.3]])
        loss = MseValueRepresentation(values).loss(values)
        assert jnp.allclose(loss, 0.0)


class TestValueHeads:
    def test_hl_gauss_head_output(self, hl_gauss_config):
        head = HlGaussHead(32, hl_gauss_config, rngs=nnx.Rngs(default=0))
        representation = head(jnp.ones((2, 4, 32)))

        assert representation.value().shape == (2, 4)
        assert representation.loss(jnp.zeros((2, 4))).shape == (2, 4)

    def test_mse_head_output(self):
        head = MseHead(32, rngs=nnx.Rngs(default=0))
        representation = head(jnp.ones((2, 4, 32)))

        assert representation.value().shape == (2, 4)
        assert representation.loss(jnp.zeros((2, 4))).shape == (2, 4)
