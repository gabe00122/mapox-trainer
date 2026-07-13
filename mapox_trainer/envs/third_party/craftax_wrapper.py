from functools import cached_property
from typing import Any, NamedTuple, Literal

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict

from mapox import Environment, TimeStep
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.renderer import GridRenderSettings, GridRenderState


PREPROCESS_SHAPE = (65, 55, 1)


class CraftaxWrapperState(NamedTuple):
    cstate: Any
    time: jax.Array
    rewards: jax.Array
    total_rewards: jax.Array
    total_episodes: jax.Array


def rgb2gray(rgb):
    return jnp.dot(rgb[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))[..., None]


class CraftaxEnvironment(Environment[CraftaxWrapperState]):
    def __init__(self, config: "CraftaxConfig", length: int) -> None:
        super().__init__()

        # craftax is an optional dependency (`uv sync --extra craftax`)
        from craftax.craftax_env import make_craftax_env_from_name

        self._symbolic = True

        self._env = make_craftax_env_from_name(
            "Craftax-Symbolic-v1" if self._symbolic else "Craftax-Pixels-v1",
            auto_reset=True,
        )
        self._env_params = self._env.default_params

        self._n_actions = self._env.action_space().n
        self._n_obs = self._env.observation_space(self._env_params).shape

    def reset(self, rng_key):
        obs, cstate = self._env.reset(rng_key, self._env_params)

        actions = jnp.zeros(1, dtype=jnp.int32)
        rewards = jnp.zeros(1)
        time = jnp.zeros(1, dtype=jnp.int32)

        state = CraftaxWrapperState(
            cstate, time, jnp.float32(0.0), jnp.float32(0), jnp.int32(0)
        )

        return state, self._encode_timestep(
            obs, jnp.array(False, dtype=jnp.bool_), actions, rewards, time
        )

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            shape=self._n_obs if self._symbolic else PREPROCESS_SHAPE,
            dtype=jnp.bfloat16,
        )

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(n=self._n_actions)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return 1

    def step(
        self, state, action: jax.Array, rng_key: jax.Array
    ) -> tuple[Any, TimeStep]:
        obs, cstate, reward, done, info = self._env.step(
            rng_key, state.cstate, action.squeeze(-1), self._env_params
        )

        rewards = state.rewards + jnp.squeeze(reward)

        total_rewards = jnp.where(
            done, state.total_rewards + rewards, state.total_rewards
        )
        total_episodes = jnp.where(done, state.total_episodes + 1, state.total_episodes)
        rewards = jnp.where(done, 0, rewards)

        state = state._replace(
            cstate=cstate,
            time=state.time + 1,
            rewards=rewards,
            total_rewards=total_rewards,
            total_episodes=total_episodes,
        )

        return state, self._encode_timestep(obs, done, action, reward[None], state.time)

    def _encode_timestep(self, obs, terminated, actions, rewards, time):
        if not self._symbolic:
            obs = jax.image.resize(obs, (65, 55, 3), jax.image.ResizeMethod.LINEAR)
            obs = rgb2gray(obs)
        obs = obs[None, ...]

        return TimeStep(
            obs=obs.astype(jnp.bfloat16),
            time=time,
            terminated=terminated[None],
            last_action=actions,
            reward=rewards,
            action_mask=None,
        )

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0), "episodes": jnp.float32(0.0)}

    def create_logs(self, state: CraftaxWrapperState):
        reward = jnp.where(
            state.total_episodes > 0,
            state.total_rewards / state.total_episodes,
            state.rewards,
        )
        return {"rewards": reward, "episodes": state.total_episodes.astype(jnp.float32)}

    def get_render_settings(self) -> GridRenderSettings:
        raise NotImplementedError("Craftax uses its own renderer (see play_craftax.py)")

    def get_render_state(self, state: CraftaxWrapperState) -> GridRenderState:
        raise NotImplementedError("Craftax uses its own renderer (see play_craftax.py)")


class CraftaxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["craftax"] = "craftax"
