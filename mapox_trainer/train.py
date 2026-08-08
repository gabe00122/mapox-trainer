import math
import random
import time
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from flax import nnx
from mapox import Environment, TimeStep
from rich.console import Console
from rich.progress import track

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.config import Config, PPOConfig
from mapox_trainer.constants import index_type
from mapox_trainer.envs import create_env_factory
from mapox_trainer.experiment import Experiment
from mapox_trainer.logger import JaxLogger
from mapox_trainer.model.network import TransformerActorCritic
from mapox_trainer.optimizer import create_optimizer
from mapox_trainer.rollout import Rollout, RolloutState
from mapox_trainer.util import add_seq_dim, count_parameters, format_count, lerp


def create_training_logs() -> dict[str, jax.Array]:
    return {
        "rewards": jnp.array(0.0),
        "value_loss": jnp.array(0.0),
        "actor_loss": jnp.array(0.0),
        "entropy_loss": jnp.array(0.0),
        "total_loss": jnp.array(0.0),
        "entropy_coef": jnp.array(0.0),
    }


def evaluate(
    model: TransformerActorCritic,
    rollout: Rollout,
    rngs: nnx.Rngs,
    env: Environment,
    hypers: PPOConfig,
):
    reset_key = rngs.env()
    env_state, timestep = env.reset(reset_key)

    rollout_state = rollout.create_state()
    carry = model.initialize_carry(rollout.batch_size, rngs)

    def _step(i, x):
        rollout_state, rngs, env_state, env_timestep, carry = x

        action_key = rngs.action()
        env_key = rngs.env()

        value_rep, policy, carry = model(add_seq_dim(env_timestep), carry)

        action = policy.sample(seed=action_key)
        log_prob = cast(jax.Array, policy.log_prob(action)).squeeze(axis=-1)
        action = action.squeeze(axis=-1).astype(index_type)
        value = model.get_value(value_rep).squeeze(axis=-1)

        env_state, next_timestep = env.step(env_state, action, env_key)

        rollout_state = rollout.store(
            rollout_state,
            step=i,
            timestep=env_timestep,
            next_timestep=next_timestep,
            log_prob=log_prob,
            value=value,
        )

        return (
            rollout_state,
            rngs,
            env_state,
            next_timestep,
            carry,
        )

    rollout_state, rngs, env_state, timestep, carry = nnx.fori_loop(
        0,
        rollout.trajectory_length,
        _step,
        init_val=(
            rollout_state,
            rngs,
            env_state,
            timestep,
            carry,
        ),
    )

    # save the last value
    value_rep, _, _ = model(add_seq_dim(timestep), carry)
    value = model.get_value(value_rep).squeeze(axis=-1)
    rollout_state = rollout_state._replace(
        values=rollout_state.values.at[:, -1].set(value)
    )

    rollout_state = rollout.calculate_advantage(
        rollout_state,
        discount=hypers.discount,
        gae_lambda=hypers.gae_lambda,
        norm_adv=hypers.normalize_advantage,
    )

    env_logs = env.create_logs(env_state)

    return rollout_state, env_logs, rngs


def ppo_loss(
    model: TransformerActorCritic,
    rollout: RolloutState,
    hypers: PPOConfig,
    progress: jax.Array,
):
    batch_obs = rollout.obs
    batch_target = rollout.targets
    batch_log_prob = rollout.log_prob
    batch_actions = rollout.actions
    batch_action_masks = rollout.action_mask
    batch_advantage = rollout.advantages
    batch_terminated = rollout.terminated

    batch_last_actions = rollout.last_actions
    batch_last_rewards = rollout.last_rewards
    batch_task_ids = rollout.task_ids

    positions = jnp.arange(batch_obs.shape[1], dtype=jnp.int32)[None, :]

    value_rep, policy, _ = model(
        TimeStep(
            obs=batch_obs,
            time=positions,
            terminated=batch_terminated,
            last_action=batch_last_actions,
            reward=batch_last_rewards,
            action_mask=batch_action_masks,
            task_ids=batch_task_ids,
        ),
    )
    log_probs = policy.log_prob(batch_actions)

    value_loss = model.get_value_loss(value_rep, batch_target)

    ratio = jnp.exp(log_probs - batch_log_prob)

    pg_loss1 = ratio * batch_advantage
    pg_loss2 = (
        jnp.clip(ratio, 1.0 - hypers.vf_clip, 1.0 + hypers.vf_clip) * batch_advantage
    )

    actor_loss = -jnp.minimum(pg_loss1, pg_loss2).mean()

    entropy_loss = -policy.entropy().mean()

    entropy_coef_value = lerp(hypers.entropy_coef, hypers.entropy_coef_end, progress)

    total_loss = (
        hypers.vf_coef * value_loss + actor_loss + entropy_coef_value * entropy_loss
    )

    logs = {
        "rewards": jnp.mean(rollout.rewards),
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "entropy_loss": entropy_loss,
        "total_loss": total_loss,
        "entropy_coef": entropy_coef_value,
    }

    return total_loss, logs


def train(
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    step: jax.Array,
    rollout: Rollout,
    env: Environment,
    config: Config,
):
    hypers = config.learner.trainer

    @partial(nnx.grad, has_aux=True)
    def _vec_grad(model, rollout_state, progress):
        return ppo_loss(model, rollout_state, hypers, progress)

    def _minibatch_step(carry, rollout_state):
        optimizer, logs, progress = carry
        grad, step_logs = _vec_grad(optimizer.model, rollout_state, progress)

        optimizer.update(grad)
        logs = jax.tree.map(lambda x, y: x + y, logs, step_logs)

        return (optimizer, logs, progress)

    def _epoch_step(i, x):
        optimizer, rollout_state, logs, rngs, progress = x

        minibatch_rng = rngs.shuffle()
        minibatch_rollout_state = rollout.create_minibatches(
            rollout_state, hypers.minibatch_count, minibatch_rng
        )
        optimizer, logs, progress = nnx.scan(
            _minibatch_step, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry
        )((optimizer, logs, progress), minibatch_rollout_state)

        return optimizer, rollout_state, logs, rngs, progress

    def _global_step(i, x):
        optimizer, logs, env_logs, rngs, step_value = x
        progress = step_value / config.update_steps

        rollout_state, env_log_update, rngs = evaluate(
            optimizer.model,
            rollout,
            rngs,
            env,
            hypers,
        )
        optimizer, rollout_state, logs, rngs, _ = nnx.fori_loop(
            0,
            hypers.epoch_count,
            _epoch_step,
            init_val=(
                optimizer,
                rollout_state,
                logs,
                rngs,
                progress,
            ),
        )

        env_logs = jax.tree.map(lambda x, y: x + y, env_logs, env_log_update)

        return optimizer, logs, env_logs, rngs, step_value + 1

    logs = create_training_logs()
    env_logs = env.create_placeholder_logs()

    optimizer, logs, env_logs, rngs, step = nnx.fori_loop(
        0,
        config.updates_per_jit,
        _global_step,
        init_val=(optimizer, logs, env_logs, rngs, step),
    )

    logs = jax.tree.map(
        lambda x: x
        / (config.updates_per_jit * hypers.epoch_count * hypers.minibatch_count),
        logs,
    )
    env_logs = jax.tree.map(lambda x: x / config.updates_per_jit, env_logs)

    return optimizer, rngs, step, {"algo": logs._asdict(), "env": env_logs}


def replicate_model(optimizer, sharding):
    state = nnx.state(optimizer)
    state = jax.device_put(state, sharding)
    nnx.update(optimizer, state)


def block_all(xs):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)


def train_run(
    experiment: Experiment,
    profile: bool = False,
    rust_env: bool = False,
):
    console = Console()

    max_steps = experiment.config.max_env_steps

    logger = JaxLogger(experiment, console)
    checkpointer = Checkpointer(experiment.checkpoints_url)

    if rust_env:
        if experiment.config.environment.env_type != "find_return":
            raise ValueError("The Rust environment currently supports find_return only")

        from mapox.envs.rust_env import RustEnv

        env = RustEnv(
            experiment.config.environment, num_envs=experiment.config.num_envs
        )
        task_count = 1
    else:
        env_factory = create_env_factory()
        env, task_count = env_factory.create_env(
            experiment.config.environment, max_steps, experiment.config.num_envs
        )

    batch_size = env.num_agents

    rollout_size = batch_size

    rngs = nnx.Rngs(default=experiment.default_seed)
    rollout = Rollout(rollout_size, max_steps, env.observation_spec, env.action_spec)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=max_steps,
        task_count=task_count,
        rngs=rngs,
    )

    optimizer = nnx.ModelAndOptimizer(
        model=model,
        tx=create_optimizer(
            experiment.config.learner.optimizer,
            experiment.config.update_steps
            * experiment.config.learner.trainer.minibatch_count
            * experiment.config.learner.trainer.epoch_count,
        ),
        wrt=nnx.Param,
    )

    rngs = nnx.Rngs(default=jax.random.PRNGKey(experiment.default_seed))

    jitted_train = nnx.jit(train, static_argnums=(3, 4, 5))

    env_steps_per_update = (
        batch_size
        * max_steps
        # * device_rngs.shape[0]
        * experiment.config.updates_per_jit
    )
    outer_updates = experiment.config.update_steps // experiment.config.updates_per_jit

    console.print(f"Starting Training: {experiment.unique_token}")
    console.print(f"Parameter Count: {count_parameters(model)}")
    console.print(f"Agent Count: {env.num_agents}")

    checkpoint_interval: int | None = None
    if experiment.config.num_checkpoints > 0 and outer_updates > 0:
        checkpoint_interval = max(
            1,
            math.ceil(outer_updates / experiment.config.num_checkpoints),
        )

    logs = None
    step = jnp.asarray(0, dtype=jnp.int32)
    for i in track(range(outer_updates), description="Training", console=console):
        start_time = time.time()

        optimizer, rngs, step, logs = jitted_train(
            optimizer, rngs, step, rollout, env, experiment.config
        )

        if profile and i >= 4:
            with jax.profiler.trace("/tmp/jax-trace"):
                optimizer, rngs, step, logs = jitted_train(
                    optimizer, rngs, step, rollout, env, experiment.config
                )
                block_all(nnx.state(optimizer))

            break

        # this should be delayed n-1 for jax to use async dispatch
        logger.log(logs, i)

        stop_time = time.time()

        delta_time = stop_time - start_time
        console.print(
            f"Steps per second: {format_count(int(env_steps_per_update / delta_time))}"
        )
        console.print(
            f"Updates per second: {experiment.config.updates_per_jit / delta_time}"
        )

        if checkpoint_interval is not None and (i + 1) % checkpoint_interval == 0:
            completed_updates = (i + 1) * experiment.config.updates_per_jit
            checkpointer.save(optimizer.model, completed_updates)

    checkpointer.save(optimizer.model, experiment.config.update_steps)

    logger.close()
    checkpointer.close()

    return -1.0
