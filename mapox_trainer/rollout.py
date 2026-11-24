from typing import NamedTuple

import jax
from einops import rearrange
from jax import numpy as jnp
from mapox import ActionSpec, ObservationSpec, TimeStep

from mapox_trainer.constants import index_type


class RolloutState(NamedTuple):
    # observation gets plus one because we need to store the next trailing observation
    obs: jax.Array
    action_mask: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    next_terminated: jax.Array

    log_prob: jax.Array
    values: jax.Array

    # these are calculated in the rollout
    advantages: jax.Array
    targets: jax.Array

    # these are used as part of the observation
    last_actions: jax.Array
    last_rewards: jax.Array
    task_ids: jax.Array


def lambda_returns(
    rewards: jax.Array,
    next_terminated: jax.Array,
    values: jax.Array,
    discount: float,
    gae_lambda: float,
) -> jax.Array:
    """TD(lambda) return for every turn of a (batch, time) rollout.

    `values` carries one extra trailing entry, the estimate that bootstraps the
    turns after the rollout ends. A termination zeroes the discount, so nothing
    leaks back across an episode boundary.
    """

    def _body(acc, xs):
        rewards, discount, v_tp1 = xs
        acc = rewards + discount * ((1 - gae_lambda) * v_tp1 + gae_lambda * acc)
        return acc, acc

    # swap to time major
    rewards = jnp.swapaxes(rewards, 0, 1)
    terminated = jnp.swapaxes(next_terminated, 0, 1)
    values = jnp.swapaxes(values, 0, 1)

    _, returns = jax.lax.scan(
        _body,
        values[-1],
        (rewards, jnp.where(terminated, 0.0, discount), values[1:]),
        reverse=True,
    )

    return jnp.swapaxes(returns, 0, 1)


def discounted_episode_return(
    rewards: jax.Array, next_terminated: jax.Array, discount: float
) -> jax.Array:
    """Mean discounted return measured from the first turn of each episode.

    Built from rewards alone: lambda=1 over a zero value function, so nothing is
    bootstrapped. An episode starts on turn 0 and on every turn that follows a
    termination, so a rollout holding several episodes per agent averages over
    all of them.
    """
    batch_size, trajectory_length = rewards.shape
    returns = lambda_returns(
        rewards,
        next_terminated,
        jnp.zeros((batch_size, trajectory_length + 1), dtype=rewards.dtype),
        discount,
        gae_lambda=1.0,
    )
    starts = jnp.concatenate(
        [jnp.ones_like(next_terminated[:, :1]), next_terminated[:, :-1]], axis=1
    )
    return jnp.sum(returns, where=starts) / jnp.sum(starts)


class Rollout:
    def __init__(
        self,
        batch_size: int,
        trajectory_length: int,
        obs_spec: ObservationSpec,
        action_spec: ActionSpec,
    ):
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def create_state(self) -> RolloutState:
        return RolloutState(
            # observation gets plus one because we need to store the next trailing observation
            obs=jnp.zeros(
                (self.batch_size, self.trajectory_length, *self.obs_spec.shape),
                dtype=self.obs_spec.dtype,
            ),
            action_mask=jnp.ones(
                (self.batch_size, self.trajectory_length, self.action_spec.n),
                dtype=jnp.bool_,
            ),
            actions=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=index_type
            ),
            rewards=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            terminated=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.bool_
            ),
            next_terminated=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.bool_
            ),
            log_prob=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            values=jnp.zeros(
                (self.batch_size, self.trajectory_length + 1), dtype=jnp.float32
            ),
            # these are calculated in the rollout
            advantages=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            targets=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            last_actions=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=index_type
            ),
            last_rewards=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            task_ids=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.int32
            ),
        )

    def store(
        self,
        state: RolloutState,
        step: jax.Array,
        timestep: TimeStep,
        next_timestep: TimeStep,
        log_prob: jax.Array,
        value: jax.Array,
    ) -> RolloutState:
        return state._replace(
            obs=state.obs.at[:, step].set(timestep.obs),
            action_mask=(
                state.action_mask.at[:, step].set(timestep.action_mask)
                if timestep.action_mask is not None
                else state.action_mask
            ),
            actions=state.actions.at[:, step].set(next_timestep.last_action),
            log_prob=state.log_prob.at[:, step].set(log_prob),
            values=state.values.at[:, step].set(value),
            rewards=state.rewards.at[:, step].set(next_timestep.reward),
            terminated=state.terminated.at[:, step].set(timestep.terminated),
            next_terminated=state.next_terminated.at[:, step].set(
                next_timestep.terminated
            ),
            last_actions=state.last_actions.at[:, step].set(timestep.last_action),
            last_rewards=state.last_rewards.at[:, step].set(timestep.reward),
            task_ids=(
                state.task_ids.at[:, step].set(timestep.task_ids)
                if timestep.task_ids is not None
                else state.task_ids
            ),
        )

    def calculate_advantage(
        self,
        state: RolloutState,
        discount: float,
        gae_lambda: float,
        norm_adv: bool,
    ) -> RolloutState:
        targets = lambda_returns(
            state.rewards, state.next_terminated, state.values, discount, gae_lambda
        )
        advantages = targets - state.values[:, :-1]

        # rollout norm
        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return state._replace(
            advantages=advantages,
            targets=targets,
        )

    def _shuffle(self, state: RolloutState, rng_key: jax.Array) -> RolloutState:
        indices = jax.random.permutation(rng_key, self.batch_size)
        return jax.tree.map(lambda x: x[indices], state)

    def create_minibatches(
        self, state: RolloutState, minibatches: int, rng_key: jax.Array
    ) -> RolloutState:
        if minibatches > 1:
            state = self._shuffle(state, rng_key)

        return jax.tree.map(
            lambda x: rearrange(x, "(m b) ... -> m b ...", m=minibatches), state
        )
