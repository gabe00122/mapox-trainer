"""Tests for GAE (Generalized Advantage Estimation) in rollout.py.

The advantage calculation is one of the most critical pieces — an off-by-one
or wrong discount mask silently produces garbage training signal.
"""

import jax
import jax.numpy as jnp
from mapox import ActionSpec, ObservationSpec

from mapox_trainer.rollout import Rollout


def make_rollout(batch_size: int = 2, trajectory_length: int = 4) -> Rollout:
    obs_spec = ObservationSpec(shape=(3, 3, 2), dtype=jnp.uint16, max_value=(5, 5))
    action_spec = ActionSpec(n=4)
    return Rollout(batch_size, trajectory_length, obs_spec, action_spec)


def manual_gae(rewards, values, terminated, discount, gae_lambda):
    """Reference implementation: simple loop, no scan tricks."""
    T = len(rewards)
    advantages = [0.0] * T
    targets = [0.0] * T

    # values has T+1 entries (includes bootstrap)
    acc = values[T]
    for t in reversed(range(T)):
        d = 0.0 if terminated[t] else discount
        acc = rewards[t] + d * ((1 - gae_lambda) * values[t + 1] + gae_lambda * acc)
        targets[t] = acc
        advantages[t] = targets[t] - values[t]

    return advantages, targets


class TestCalculateAdvantage:
    def test_vocab_buffers_are_uint16(self):
        state = make_rollout().create_state()

        assert state.obs.dtype == jnp.uint16
        assert state.actions.dtype == jnp.uint16
        assert state.last_actions.dtype == jnp.uint16

    def test_single_step_no_termination(self):
        """Simplest case: one timestep, no episode boundary."""
        rollout = make_rollout(batch_size=1, trajectory_length=1)
        state = rollout.create_state()

        rewards = jnp.array([[1.0]])
        values = jnp.array([[0.5, 0.8]])  # V(t=0), V(t=1) bootstrap
        terminated = jnp.array([[False]])

        state = state._replace(
            rewards=rewards,
            values=values,
            next_terminated=terminated,
        )

        result = rollout.calculate_advantage(state, discount=0.99, gae_lambda=0.95, norm_adv=False)

        # Manual: target = r + γ((1-λ)V(1) + λ*V(1)) = r + γ*V(1) = 1.0 + 0.99*0.8
        expected_target = 1.0 + 0.99 * 0.8
        expected_adv = expected_target - 0.5

        assert jnp.allclose(result.targets[0, 0], expected_target, atol=1e-5)
        assert jnp.allclose(result.advantages[0, 0], expected_adv, atol=1e-5)

    def test_termination_zeros_discount(self):
        """When an episode terminates, future rewards should not bleed back."""
        rollout = make_rollout(batch_size=1, trajectory_length=3)
        state = rollout.create_state()

        rewards = jnp.array([[1.0, 2.0, 3.0]])
        values = jnp.array([[0.5, 0.5, 0.5, 0.5]])
        # Episode terminates AFTER step 1 (next_terminated at step 1 is True)
        terminated = jnp.array([[False, True, False]])

        state = state._replace(
            rewards=rewards,
            values=values,
            next_terminated=terminated,
        )

        result = rollout.calculate_advantage(state, discount=0.99, gae_lambda=0.95, norm_adv=False)

        # Step 2 (last): target = r2 + γ*((1-λ)*V3 + λ*V3) = 3 + 0.99*0.5
        # Step 1 (terminated): target = r1 + 0*(...) = 2.0 (discount is 0)
        # Step 0: target = r0 + γ*((1-λ)*V1 + λ*target1) = 1 + 0.99*(0.05*0.5 + 0.95*2.0)
        ref_adv, ref_targets = manual_gae(
            rewards[0].tolist(), values[0].tolist(), terminated[0].tolist(), 0.99, 0.95
        )

        for t in range(3):
            assert jnp.allclose(result.targets[0, t], ref_targets[t], atol=1e-5), (
                f"target mismatch at t={t}: got {result.targets[0, t]}, expected {ref_targets[t]}"
            )

    def test_matches_reference_implementation(self):
        """Multi-step trajectory should match naive loop implementation."""
        rollout = make_rollout(batch_size=1, trajectory_length=5)
        state = rollout.create_state()

        rewards = jnp.array([[1.0, -0.5, 2.0, 0.0, 1.5]])
        values = jnp.array([[0.3, 0.7, 0.1, 0.9, 0.4, 0.6]])
        terminated = jnp.array([[False, False, False, False, False]])

        state = state._replace(rewards=rewards, values=values, next_terminated=terminated)
        result = rollout.calculate_advantage(state, discount=0.97, gae_lambda=0.9, norm_adv=False)

        ref_adv, ref_targets = manual_gae(
            rewards[0].tolist(), values[0].tolist(), terminated[0].tolist(), 0.97, 0.9
        )

        for t in range(5):
            assert jnp.allclose(result.targets[0, t], ref_targets[t], atol=1e-4)
            assert jnp.allclose(result.advantages[0, t], ref_adv[t], atol=1e-4)

    def test_batch_dimension_independent(self):
        """Each batch element should be computed independently."""
        rollout = make_rollout(batch_size=2, trajectory_length=3)
        state = rollout.create_state()

        rewards = jnp.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        values = jnp.array([[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]])
        terminated = jnp.zeros((2, 3), dtype=jnp.bool_)

        state = state._replace(rewards=rewards, values=values, next_terminated=terminated)
        result = rollout.calculate_advantage(state, discount=0.99, gae_lambda=0.95, norm_adv=False)

        # Compare each batch element against its own reference
        for b in range(2):
            ref_adv, ref_targets = manual_gae(
                rewards[b].tolist(), values[b].tolist(), terminated[b].tolist(), 0.99, 0.95
            )
            for t in range(3):
                assert jnp.allclose(result.targets[b, t], ref_targets[t], atol=1e-4)

    def test_advantage_normalization(self):
        """norm_adv=True should produce zero-mean, unit-variance advantages."""
        rollout = make_rollout(batch_size=4, trajectory_length=8)
        state = rollout.create_state()

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        rewards = jax.random.normal(k1, (4, 8))
        values = jax.random.normal(k2, (4, 9))
        terminated = jnp.zeros((4, 8), dtype=jnp.bool_)

        state = state._replace(rewards=rewards, values=values, next_terminated=terminated)
        result = rollout.calculate_advantage(state, discount=0.99, gae_lambda=0.95, norm_adv=True)

        assert jnp.allclose(result.advantages.mean(), 0.0, atol=1e-5)
        assert jnp.allclose(result.advantages.std(), 1.0, atol=0.1)

    def test_discount_zero_is_one_step(self):
        """With discount=0, target should just be the immediate reward."""
        rollout = make_rollout(batch_size=1, trajectory_length=3)
        state = rollout.create_state()

        rewards = jnp.array([[5.0, 10.0, 15.0]])
        values = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        terminated = jnp.zeros((1, 3), dtype=jnp.bool_)

        state = state._replace(rewards=rewards, values=values, next_terminated=terminated)
        result = rollout.calculate_advantage(state, discount=0.0, gae_lambda=0.95, norm_adv=False)

        # With discount=0, target_t = reward_t
        assert jnp.allclose(result.targets, rewards, atol=1e-5)
