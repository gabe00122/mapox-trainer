"""League evaluation: every checkpoint of every run plays in one shared batch.

The env's agent slots are dealt out evenly between the checkpoints by one
shuffle of all of them, redone every round, so a checkpoint's teammates and
opponents are random draws from the pool. A checkpoint is scored by the mean
return of the first life of the slots it played, per task.

Every checkpoint has the same number of seats, enough between them for all the
slots. When the slots don't fill them, the seats left over stay empty for the
round, at most one per checkpoint, and which checkpoints are a slot short is
redrawn every round. How many of a checkpoint's slots are of each task is up
to the shuffle too, and evens out over the rounds.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, Any

import fsspec
import jax
import numpy as np
import pandas as pd
import typer
from flax import nnx
from jax import numpy as jnp
from mapox import Environment, EnvironmentConfig, TimeStep
from rich.console import Console
from rich.table import Table

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.config import load_config
from mapox_trainer.constants import index_type
from mapox_trainer.envs import create_env_factory
from mapox_trainer.experiment import Experiment
from mapox_trainer.model.network import TransformerActorCritic
from mapox_trainer.utils.eval_plot import save_eval_plot

app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass(frozen=True)
class Checkpoint:
    run: str
    step: int


@dataclass
class PolicyGroup:
    """Checkpoints of one architecture, their weights stacked on a leading axis.

    Only weights of the same shapes stack, so each architecture in the pool is
    its own group and its own forward pass.
    """

    template: TransformerActorCritic
    state: nnx.State
    count: int

    def initialize_carry(self, seats: int):
        carry = self.template.initialize_carry(seats, nnx.Rngs(0))
        return jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.count, *x.shape)), carry
        )

    def act(self, state: nnx.State, ts: TimeStep, carry, key: jax.Array):
        """Sample an action for every seat of every checkpoint in the group."""
        graphdef = nnx.graphdef(self.template)

        def _act(state, ts, carry, key):
            _, policy, carry = nnx.merge(graphdef, state)(ts, carry)
            action = policy.sample(seed=key).squeeze(axis=-1).astype(index_type)
            return action, carry

        return jax.vmap(_act)(state, ts, carry, jax.random.split(key, self.count))


def seats_per_checkpoint(num_slots: int, num_checkpoints: int) -> int:
    """How many seats each checkpoint has in a round: the fewest that hold all
    of the env's agent slots between them."""
    return -(-num_slots // num_checkpoints)


def agents_per_checkpoint(num_slots: int, num_checkpoints: int) -> str:
    """The slots a checkpoint plays in a round, which is one of two counts when
    the slots leave seats empty."""
    most = seats_per_checkpoint(num_slots, num_checkpoints)
    fewest = num_slots // num_checkpoints
    return str(most) if fewest == most else f"{fewest} or {most}"


def assign_seats(key: jax.Array, num_slots: int, num_checkpoints: int) -> jax.Array:
    """Deal the env's slots out between the checkpoints at random, as evenly as
    they go.

    Returns the env slot behind each seat, shaped (checkpoints, seats): a row is
    the slots one checkpoint plays, of whichever tasks the shuffle gave it. The
    seats the slots don't fill are empty, which is marked by `num_slots`, one
    past the last slot.
    """
    seats = seats_per_checkpoint(num_slots, num_checkpoints)
    deal_key, row_key = jax.random.split(key)

    deck = jnp.full((num_checkpoints * seats,), num_slots)
    deck = deck.at[:num_slots].set(jax.random.permutation(deal_key, num_slots))
    # dealt a seat at a time, so the empty seats at the end of the deck go to
    # as many checkpoints, one each
    slot_of_seat = deck.reshape(seats, num_checkpoints).T
    # and those are a different draw of the checkpoints every round
    return jax.random.permutation(row_key, slot_of_seat)


def make_league_eval(
    env: Environment,
    groups: list[PolicyGroup],
    max_steps: int,
    rounds: int,
):
    """Build the jitted league: `(group states, key) -> (sum, sum of squares,
    count)` of the first-life returns, each shaped (rounds, checkpoints, tasks).
    """
    num_checkpoints = sum(group.count for group in groups)
    seats = seats_per_checkpoint(env.num_agents, num_checkpoints)
    group_starts = np.cumsum([0, *(group.count for group in groups)])

    def _rows(tree, start: int, stop: int):
        return jax.tree.map(lambda x: x[start:stop], tree)

    def _round(states: tuple[nnx.State, ...], key: jax.Array):
        seat_key, reset_key, rollout_key = jax.random.split(key, 3)

        slot_of_seat = assign_seats(seat_key, env.num_agents, num_checkpoints)
        # an empty seat still takes a place in its checkpoint's batch. It reads
        # slot 0, and neither its action nor its return goes anywhere
        occupied = slot_of_seat < env.num_agents
        read_slot = jnp.where(occupied, slot_of_seat, 0)

        env_state, timestep = env.reset(reset_key)
        # a slot plays one task for the whole round; an env of one task names none
        if timestep.task_ids is None:
            task_of_slot = jnp.zeros((env.num_agents,), jnp.int32)
        else:
            task_of_slot = jnp.asarray(timestep.task_ids)
        # (checkpoints, seats, tasks): the task each seat's return counts toward,
        # which for an empty seat is none of them
        seat_task = jax.nn.one_hot(task_of_slot[read_slot], env.num_tasks)
        seat_task = seat_task * occupied[:, :, None]
        carries = tuple(group.initialize_carry(seats) for group in groups)
        returns = jnp.zeros(slot_of_seat.shape, jnp.float32)
        first_life = occupied

        def _step(carry, key):
            env_state, timestep, carries, returns, first_life = carry
            action_key, env_key = jax.random.split(key)

            # (checkpoints, seats, 1, ...): each checkpoint sees its own seats
            seated = jax.tree.map(lambda x: x[read_slot][:, :, None], timestep)

            actions, next_carries = [], []
            group_keys = jax.random.split(action_key, len(groups))
            for group, state, group_carry, group_key, (start, stop) in zip(
                groups, states, carries, group_keys, pairwise(group_starts)
            ):
                action, group_carry = group.act(
                    state, _rows(seated, start, stop), group_carry, group_key
                )
                actions.append(action)
                next_carries.append(group_carry)

            # every slot is some seat's, so this fills all of them. An empty
            # seat's slot is out of bounds, which drops its action
            env_actions = jnp.zeros((env.num_agents,), index_type)
            env_actions = env_actions.at[slot_of_seat].set(
                jnp.concatenate(actions), mode="drop"
            )
            env_state, timestep = env.step(env_state, env_actions, env_key)

            # the reward of the step that ends a life still belongs to it
            reward = jnp.asarray(timestep.reward)[read_slot]
            returns = returns + jnp.where(first_life, reward, 0.0)
            first_life = first_life & ~jnp.asarray(timestep.terminated)[read_slot]

            return (env_state, timestep, tuple(next_carries), returns, first_life), None

        (_, _, _, returns, _), _ = jax.lax.scan(
            _step,
            (env_state, timestep, carries, returns, first_life),
            jax.random.split(rollout_key, max_steps),
        )

        def _per_task(values):
            return jnp.einsum("cs,cst->ct", values, seat_task)

        return (
            _per_task(returns),
            _per_task(jnp.square(returns)),
            _per_task(jnp.ones_like(returns)),
        )

    @jax.jit
    def _league(states: tuple[nnx.State, ...], key: jax.Array):
        _, stats = jax.lax.scan(
            lambda _, round_key: (None, _round(states, round_key)),
            None,
            jax.random.split(key, rounds),
        )
        return stats

    return _league


def load_league(
    experiments: list[Experiment], env: Environment, max_steps: int
) -> tuple[list[Checkpoint], list[PolicyGroup]]:
    """Restore every checkpoint of every run, grouped by architecture.

    The checkpoints come back in group order, which is the order of the league's
    checkpoint axis.
    """
    by_model: dict[Any, list[Experiment]] = {}
    for experiment in experiments:
        by_model.setdefault(experiment.config.learner.model, []).append(experiment)

    checkpoints: list[Checkpoint] = []
    groups: list[PolicyGroup] = []

    for model_config, group_experiments in by_model.items():
        template = TransformerActorCritic(
            model_config,
            env.observation_spec,
            env.action_spec.n,
            max_seq_length=max_steps,
            task_count=env.num_tasks,
            rngs=nnx.Rngs(0),
        )
        treedef = jax.tree.structure(nnx.state(template))

        weights = []
        for experiment in group_experiments:
            with Checkpointer(experiment.checkpoints_url) as checkpointer:
                for step in checkpointer.mngr.all_steps():
                    checkpointer.restore(template, step)
                    # onto the host: a pool's worth of loose checkpoints on the
                    # gpu doubles the weights there and fragments what is left
                    leaves = jax.tree.leaves(nnx.state(template))
                    weights.append([np.asarray(leaf) for leaf in leaves])
                    checkpoints.append(Checkpoint(experiment.unique_token, step))

        if weights:
            stacked = [jnp.asarray(np.stack(leaves)) for leaves in zip(*weights)]
            state = jax.tree.unflatten(treedef, stacked)
            groups.append(PolicyGroup(template, state, len(weights)))

    return checkpoints, groups


def _task_layout(node: Any) -> Any:
    """The part of an environment config that decides its tasks and vocabulary:
    the env types and task names, without their settings."""
    if isinstance(node, dict):
        return {
            key: _task_layout(value)
            for key, value in node.items()
            if key in ("env_type", "name") or isinstance(value, (dict, list, tuple))
        }
    if isinstance(node, (list, tuple)):
        return [_task_layout(value) for value in node]
    return node


def league_env(
    experiments: list[Experiment], env_config: str | None, console: Console
) -> tuple[EnvironmentConfig, int]:
    """The environment config and episode length the league is played with:
    those of the training config at `env_config`, or else of the first run.

    Every run has to have trained on the same tasks as it, since the tasks
    decide the vocabulary, but their settings are free to differ. That is how a
    league is scaled to the gpu: by a config with fewer games of each task.
    """
    if env_config is None:
        source = experiments[0].unique_token
        config = experiments[0].config
    else:
        source = env_config
        with fsspec.open(env_config, "r") as f:
            config = load_config(f.read())

    environment, max_steps = config.environment, config.max_env_steps
    layout = _task_layout(environment.model_dump())

    for experiment in experiments:
        trained = experiment.config
        if _task_layout(trained.environment.model_dump()) != layout:
            raise ValueError(
                f"{experiment.unique_token} trained on different tasks than "
                f"{source}, so they don't share a vocabulary"
            )

        # a difference from a config that was asked for is no surprise
        if env_config is None and (
            trained.environment != environment or trained.max_env_steps != max_steps
        ):
            console.print(
                f"[yellow]{experiment.unique_token} trained with different env "
                f"settings; it is evaluated with those of {source}"
            )

    return environment, max_steps


def summarize(
    checkpoints: list[Checkpoint],
    task_names: list[str],
    total: np.ndarray,
    total_squares: np.ndarray,
    count: np.ndarray,
) -> pd.DataFrame:
    """One row per checkpoint and task from the league's per-round sums."""
    total = total.astype(np.float64).sum(axis=0)
    total_squares = total_squares.astype(np.float64).sum(axis=0)
    count = count.astype(np.float64).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        variance = (total_squares / count - np.square(mean)) * count / (count - 1)
        sem = np.sqrt(np.maximum(variance, 0.0) / count)
    # one sample has no spread to estimate
    sem = np.where(count > 1, sem, np.nan)

    return pd.DataFrame(
        [
            (
                checkpoint.run,
                checkpoint.step,
                task_name,
                mean[i, task],
                sem[i, task],
                int(count[i, task]),
            )
            for i, checkpoint in enumerate(checkpoints)
            for task, task_name in enumerate(task_names)
        ],
        columns=["run", "step", "task", "mean", "sem", "count"],
    )


def print_summary(console: Console, df: pd.DataFrame) -> None:
    """A table per run of the mean first-life returns; the csv has the errors."""
    tasks = list(dict.fromkeys(df["task"]))

    for run, run_df in df.groupby("run", sort=False):
        table = Table(title=f"Mean first-life return: {run}")
        table.add_column("step", justify="right")
        for task in tasks:
            table.add_column(task, justify="right")

        means = run_df.pivot(index="step", columns="task", values="mean")
        for step, row in means.sort_index().iterrows():
            table.add_row(str(step), *(f"{row[task]:.2f}" for task in tasks))
        console.print(table)


def evaluate(
    run_tokens: list[str],
    seed: int,
    rounds: int,
    output_name: str,
    base_dir: str = "results",
    env_config: str | None = None,
) -> pd.DataFrame:
    console = Console()

    run_tokens = list(dict.fromkeys(run_tokens))
    experiments = [Experiment.load(name, base_dir=base_dir) for name in run_tokens]
    environment, max_steps = league_env(experiments, env_config, console)
    env = create_env_factory().create_env(environment, max_steps)

    checkpoints, groups = load_league(experiments, env, max_steps)
    if not checkpoints:
        raise ValueError("none of the runs has a checkpoint")

    task_names = env.task_names or ["all"]
    if len(task_names) != env.num_tasks:
        raise ValueError(
            f"the env names {len(task_names)} tasks but plays {env.num_tasks}"
        )

    console.print(
        f"{len(checkpoints)} checkpoints of {len(groups)} architecture(s), "
        f"{rounds} rounds of {max_steps} steps, "
        f"{agents_per_checkpoint(env.num_agents, len(checkpoints))} of the "
        f"{env.num_agents} agents per checkpoint"
    )

    league = make_league_eval(env, groups, max_steps, rounds)
    with console.status("Playing the league"):
        stats = league(tuple(group.state for group in groups), jax.random.key(seed))
    total, total_squares, count = (np.asarray(x) for x in stats)

    df = summarize(checkpoints, task_names, total, total_squares, count)
    print_summary(console, df)

    df.to_csv(output_name + ".csv", index=False)
    save_eval_plot(df, output_name + ".png")

    return df


@app.command()
def main(
    run: Annotated[
        list[str],
        typer.Option(
            ...,
            help="Existing experiment run token (under the base dir)",
            rich_help_panel="Input",
        ),
    ],
    out: Annotated[
        str, typer.Option(help="Path of the results to save, without an extension")
    ],
    seed: Annotated[int, typer.Option(help="Random seed for RNGs.")] = 0,
    rounds: Annotated[
        int,
        typer.Option(help="Rollouts of the whole env batch, reshuffled each time."),
    ] = 8,
    base_dir: Annotated[
        str, typer.Option(help="Directory the runs are stored under.")
    ] = "results",
    env_config: Annotated[
        str | None,
        typer.Option(
            help="Training config whose environment and max_env_steps the "
            "league is played with, in place of the first run's. It needs the "
            "same tasks as the runs. This is where the games of each task are set "
            "to fit the league to the GPU."
        ),
    ] = None,
):
    evaluate(
        run,
        seed=seed,
        rounds=rounds,
        output_name=out,
        base_dir=base_dir,
        env_config=env_config,
    )


if __name__ == "__main__":
    app()
