from typing import Any, cast

import jax
from flax import nnx
from mapox import Environment, TimeStep
from mapox.agent import Agent
from mapox.envs.rust_env import RustEnv
from mapox.rust_play import rust_enjoy

# from mapox.play import enjoy
from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.constants import index_type
from mapox_trainer.envs import create_env_factory
from mapox_trainer.experiment import Experiment
from mapox_trainer.model.network import TransformerActorCritic
from mapox_trainer.util import add_seq_dim


@jax.jit(static_argnums=(0,), donate_argnums=(2, 3))
def _act(
    model_def, model_params, agent_state, rng_key: jax.Array, timestep: TimeStep
) -> tuple[jax.Array, Any, jax.Array]:
    model = nnx.merge(model_def, model_params)

    _, policy, agent_state = model(add_seq_dim(timestep), agent_state)

    sample_rng, rng_key = jax.random.split(rng_key)
    actions = policy.sample(seed=sample_rng).squeeze(axis=-1).astype(index_type)

    return actions, agent_state, rng_key

@jax.jit(static_argnums=(0, 2), donate_argnums=(1,))
def _reset(model_def, model_params, num_agents: int) -> Any:
    model = nnx.merge(model_def, model_params)
    return model.initialize_carry(num_agents, None)

class MapoxAgent(Agent):
    def __init__(
        self,
        experiment: Experiment,
        env: Environment,
        max_steps: int,
        task_count: int,
        rngs: nnx.Rngs,
    ):
        model = TransformerActorCritic(
            experiment.config.learner.model,
            env.observation_spec,
            env.action_spec.n,
            max_seq_length=max_steps,
            task_count=task_count,
            rngs=rngs,
        )

        with Checkpointer(experiment.checkpoints_url) as checkpointer:
            model = checkpointer.restore_latest(model)

        self._agent_state = None
        self._model_def, self._model_params = nnx.split(model)
        self._rng_key = rngs.policy()

    def act(self, timestep: TimeStep) -> jax.Array:
        actions, self._agent_state, self._rng_key = _act(
            self._model_def,
            self._model_params,
            self._agent_state,
            self._rng_key,
            timestep,
        )
        return actions

    def reset(self, num_agents: int, seed: int) -> None:
        self._agent_state = _reset(self._model_def, self._model_params, num_agents)

def play_from_run(
    name: str,
    human_control: bool,
    pov: bool,
    seed: int,
    env_name: str | None = None,
    video_path: str | None = None,
    size: int = 500,
    fps: int = 15,
):
    experiment = Experiment.load(name, "results")
    config = experiment.config
    rngs = nnx.Rngs(default=experiment.default_seed)

    env, task_count = create_env_factory().create_env(config.environment, config.max_env_steps, 1, env_name)

    agent = MapoxAgent(experiment, env, config.max_env_steps, task_count, rngs)
    rust_enjoy(cast(RustEnv, env), experiment.config.max_env_steps, seed, agent)
    # enjoy(
    #     env,
    #     agent,
    #     rngs.env(),
    #     video_path,
    #     size,
    #     fps,
    #     human_control,
    #     pov,
    # )
