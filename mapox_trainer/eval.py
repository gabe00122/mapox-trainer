from dataclasses import dataclass
import random
from typing import Any, Optional
import jax
from jax import numpy as jnp
from flax import nnx
import pandas as pd

import trueskill

from functools import partial

import typer
from rich import progress
from rich.console import Console

from mapox import Environment, TimeStep

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.envs import create_env_factory
from mapox_trainer.experiment import Experiment
from mapox_trainer.model.network import TransformerActorCritic
from mapox_trainer.train import add_seq_dim
from mapox_trainer.utils.ranking_plot import save_ranking_plot


console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass
class PolicyRecord:
    name: str
    step: int
    rating: trueskill.Rating
    model: Any
    task_id: int


@partial(nnx.jit, static_argnums=(0, 3, 4, 5))
def _round(
    env: Environment,
    policy1: TransformerActorCritic,
    policy2: TransformerActorCritic,
    policy1_task_id: int,
    policy2_task_id: int,
    max_steps: int,
    rngs: nnx.Rngs,
):
    teams = env.teams
    team_size = env.num_agents // 2

    policy1_carry = policy1.initialize_carry(team_size, rngs)
    policy2_carry = policy2.initialize_carry(team_size, rngs)

    policy1_idx = jnp.where(teams == 0, fill_value=-1, size=team_size)
    policy2_idx = jnp.where(teams == 1, fill_value=-1, size=team_size)

    state, ts = env.reset(rngs.env())

    def _step(_, x):
        (
            state,
            ts,
            policy1_carry,
            policy2_carry,
            policy1_reward,
            policy2_reward,
            rngs,
        ) = x

        ts = add_seq_dim(ts)

        policy1_ts: TimeStep = jax.tree.map(lambda item: item[policy1_idx], ts)
        policy2_ts: TimeStep = jax.tree.map(lambda item: item[policy2_idx], ts)

        # override the task id's in case they are different for different models
        policy1_ts = policy1_ts._replace(
            task_ids=jnp.full((policy1_ts.obs.shape[0],), policy1_task_id)
        )
        policy2_ts = policy2_ts._replace(
            task_ids=jnp.full((policy2_ts.obs.shape[0],), policy2_task_id)
        )

        _, a1, policy1_carry = policy1(policy1_ts, policy1_carry)
        _, a2, policy2_carry = policy2(policy2_ts, policy2_carry)

        a1 = a1.sample(seed=rngs.action()).squeeze(-1)
        a2 = a2.sample(seed=rngs.action()).squeeze(-1)

        actions = jnp.zeros((env.num_agents,), jnp.int32)
        actions = actions.at[policy1_idx].set(a1)
        actions = actions.at[policy2_idx].set(a2)

        state, ts = env.step(state, actions, rngs.env())

        policy1_reward = policy1_reward + policy1_ts.reward.sum()
        policy2_reward = policy2_reward + policy2_ts.reward.sum()

        return (
            state,
            ts,
            policy1_carry,
            policy2_carry,
            policy1_reward,
            policy2_reward,
            rngs,
        )

    _, _, _, _, policy1_reward, policy2_reward, rngs = nnx.fori_loop(
        0,
        max_steps,
        _step,
        (
            state,
            ts,
            policy1_carry,
            policy2_carry,
            jnp.float32(0.0),
            jnp.float32(0.0),
            rngs,
        ),
    )

    return policy1_reward, policy2_reward, rngs


def load_policy(
    experiment: Experiment,
    env,
    env_name,
    max_steps: int,
    task_count: int,
    rngs: nnx.Rngs,
) -> list[PolicyRecord]:
    model_template = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=max_steps,
        task_count=task_count,
        rngs=rngs,
    )

    task_id = 0

    if experiment.config.environment.env_type == "multi":
        for i, task in enumerate(experiment.config.environment.envs):
            if task.name == env_name:
                task_id = i
                break

    policies = []

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        for step in checkpointer.mngr.all_steps():
            checkpointer.restore(model_template, step)
            model = nnx.clone(model_template)
            policy = PolicyRecord(
                experiment.unique_token, step, trueskill.Rating(), model, task_id
            )
            policies.append(policy)

    return policies


def evaluate(
    run_tokens: list[str],
    env_name: Optional[str],
    seed: int,
    rounds: int,
    output_name: str,
):
    console = Console()

    experiment = Experiment.load(run_tokens[0], base_dir="results")
    max_steps = experiment.config.max_env_steps

    env_factory = create_env_factory()

    env, task_count = env_factory.create_env(
        experiment.config.environment, max_steps, vec_count=32, env_name=env_name
    )
    rngs = nnx.Rngs(default=seed)

    league: list[PolicyRecord] = []
    run_tokens = list(dict.fromkeys(run_tokens))

    for name in run_tokens:
        console.print(f"Loading: {name}")
        experiment = Experiment.load(name, base_dir="results")
        policies = load_policy(experiment, env, env_name, max_steps, task_count, rngs)
        league.extend(policies)

    for _ in progress.track(
        range(rounds), description="Ranking rounds", console=console
    ):
        agents = random.choices(league, k=2)

        policy1_reward, policy2_reward, rngs = _round(
            env,
            agents[0].model,
            agents[1].model,
            agents[0].task_id,
            agents[1].task_id,
            max_steps,
            rngs,
        )
        winner, loser = (
            agents if policy1_reward > policy2_reward else (agents[1], agents[0])
        )

        winner_rating, loser_rating = trueskill.rate_1vs1(winner.rating, loser.rating)
        winner.rating = winner_rating
        loser.rating = loser_rating

    df = pd.DataFrame(
        [
            (policy.name, policy.step, policy.rating.mu, policy.rating.sigma)
            for policy in league
        ],
        columns=["run", "step", "mu", "sigma"],
    )

    df.to_csv(output_name + ".csv")
    save_ranking_plot(df, output_name + ".png")


@app.command()
def main(
    run: list[str] = typer.Option(
        ...,
        help="Existing experiment run token (under results/)",
        rich_help_panel="Input",
    ),
    env: Optional[str] = typer.Option(
        None, help="Select a specific env when using a multi env config."
    ),
    seed: int = typer.Option(0, help="Random seed for RNGs."),
    rounds: int = typer.Option(1000, help="Number of head-to-head rounds to run."),
    out: str = typer.Option(help="The path and name of the results to save"),
):
    evaluate(run, env_name=env, seed=seed, rounds=rounds, output_name=out)


if __name__ == "__main__":
    app()
