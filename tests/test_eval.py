"""Tests for the league evaluation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from mapox import Environment, ObservationSpec, TimeStep
from mapox.specs import DiscreteActionSpec
from rich.console import Console

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.config import (
    AttentionConfig,
    FeedForwardConfig,
    LayerConfig,
    LinearObsEncoderConfig,
    MseCriticConfig,
    TransformerActorCriticConfig,
    load_config,
)
from mapox_trainer.eval import (
    Checkpoint,
    PolicyGroup,
    assign_seats,
    league_env,
    load_league,
    make_league_eval,
    summarize,
)
from mapox_trainer.experiment import Experiment
from mapox_trainer.model.network import TransformerActorCritic

CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "return.json"

NUM_ACTIONS = 4
MAX_STEPS = 12


class LivesEnv(Environment):
    """Pays a slot 1 for each step it plays its own action, the only one its
    mask allows, and ends slot i's lives every `1 + i % 5` steps. A slot's
    first-life return is then that life's length, whichever policy plays it,
    as long as each slot is handed the action sampled for it."""

    def __init__(self, task_sizes: tuple[int, ...]):
        self._num_tasks = len(task_sizes)
        self._task_ids = jnp.repeat(
            jnp.arange(len(task_sizes)), jnp.array(task_sizes)
        ).astype(jnp.int32)
        slots = jnp.arange(self.num_agents)
        self._own_action = slots % NUM_ACTIONS
        self.life_length = 1 + np.arange(self.num_agents) % 5

    def _timestep(self, time, reward):
        return TimeStep(
            obs=jnp.zeros((self.num_agents, 3), jnp.float32),
            time=jnp.full((self.num_agents,), time, jnp.int32),
            terminated=(time > 0) & (time % jnp.asarray(self.life_length) == 0),
            last_action=jnp.zeros((self.num_agents,), jnp.uint16),
            reward=reward,
            action_mask=jax.nn.one_hot(self._own_action, NUM_ACTIONS, dtype=jnp.bool_),
            task_ids=self._task_ids,
        )

    def reset(self, rng_key):
        time = jnp.int32(0)
        return time, self._timestep(time, jnp.zeros((self.num_agents,), jnp.float32))

    def step(self, state, action, rng_key):
        time = state + 1
        reward = (action == self._own_action).astype(jnp.float32)
        return time, self._timestep(time, reward)

    @property
    def observation_spec(self):
        return ObservationSpec(dtype=jnp.float32, shape=(3,))

    @property
    def action_spec(self):
        return DiscreteActionSpec(NUM_ACTIONS)

    @property
    def num_agents(self) -> int:
        return self._task_ids.shape[0]

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    def create_placeholder_logs(self):
        return {}

    def create_logs(self, state):
        return {}

    def get_render_settings(self):
        raise NotImplementedError

    def get_render_state(self, state):
        raise NotImplementedError

    @property
    def obs_vocab(self):
        raise NotImplementedError

    @property
    def action_vocab(self):
        raise NotImplementedError


def model_config(num_layers: int) -> TransformerActorCriticConfig:
    return TransformerActorCriticConfig(
        obs_encoder=LinearObsEncoderConfig(),
        hidden_features=16,
        layer=LayerConfig(
            feed_forward=FeedForwardConfig(size=32, activation="gelu"),
            history=AttentionConfig(num_heads=2, num_kv_heads=1, head_dim=8),
        ),
        num_layers=num_layers,
        value=MseCriticConfig(),
        norm="rms_norm",
        # the cudnn attention a gpu run picks takes nothing wider
        dtype="bfloat16",
    )


def make_model(env: Environment, num_layers: int, seed: int) -> TransformerActorCritic:
    return TransformerActorCritic(
        model_config(num_layers),
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=MAX_STEPS,
        task_count=env.num_tasks,
        rngs=nnx.Rngs(seed),
    )


def make_group(env: Environment, num_layers: int, seeds: list[int]) -> PolicyGroup:
    models = [make_model(env, num_layers, seed) for seed in seeds]
    state = jax.tree.map(lambda *xs: jnp.stack(xs), *(nnx.state(m) for m in models))
    return PolicyGroup(models[0], state, len(models))


def test_every_slot_gets_one_seat():
    slot_of_seat = np.asarray(assign_seats(jax.random.key(0), 18, 3))

    assert slot_of_seat.shape == (3, 6)
    assert sorted(slot_of_seat.flat) == list(range(18))


def test_agents_the_pool_does_not_divide_are_refused():
    with pytest.raises(ValueError, match="17 agents don't divide evenly"):
        assign_seats(jax.random.key(0), 17, 3)


def test_the_seats_are_reshuffled_by_the_key():
    first = assign_seats(jax.random.key(0), 12, 3)
    second = assign_seats(jax.random.key(1), 12, 3)

    assert not np.array_equal(first, second)


def test_league_scores_the_first_life_of_every_slot():
    # the pool divides the 18 slots but neither task's, so a checkpoint's seats
    # of a task change from round to round
    env = LivesEnv((7, 11))
    # two architectures
    groups = [make_group(env, 1, [0, 1]), make_group(env, 2, [2])]
    task_of_slot = np.repeat([0, 1], [7, 11])
    rounds = 3

    league = make_league_eval(env, groups, MAX_STEPS, rounds)
    total, _, count = league(tuple(g.state for g in groups), jax.random.key(0))
    total, count = np.asarray(total), np.asarray(count)

    assert total.shape == count.shape == (rounds, 3, 2)
    # every checkpoint plays 6 seats a round, between the two tasks
    assert np.all(count.sum(axis=2) == 6)
    for task in range(2):
        slots = np.flatnonzero(task_of_slot == task)
        # each round every slot is played once, by one checkpoint or another
        assert np.all(count[:, :, task].sum(axis=1) == len(slots))
        assert np.all(total[:, :, task].sum(axis=1) == env.life_length[slots].sum())


def test_summarize_reports_the_mean_and_its_standard_error():
    returns = np.array([1.0, 2.0, 3.0, 6.0])
    # two rounds of two samples each, for one checkpoint on one task
    total = returns.reshape(2, 1, 1, 2).sum(axis=-1)
    squares = np.square(returns).reshape(2, 1, 1, 2).sum(axis=-1)
    count = np.full((2, 1, 1), 2.0)

    df = summarize([Checkpoint("run", 5)], ["task"], total, squares, count)

    row = df.iloc[0]
    assert (row["run"], row["step"], row["task"], row["count"]) == ("run", 5, "task", 4)
    assert row["mean"] == pytest.approx(returns.mean())
    assert row["sem"] == pytest.approx(returns.std(ddof=1) / 2.0)


def test_load_league_stacks_the_checkpoints_of_each_architecture(tmp_path):
    env = LivesEnv((4,))
    base_config = load_config(CONFIG_FILE.read_text())

    def save_run(name: str, num_layers: int, steps: dict[int, int]):
        learner = base_config.learner.model_copy(
            update={"model": model_config(num_layers)}
        )
        config = base_config.model_copy(update={"learner": learner})
        experiment = Experiment.from_config(name, config, str(tmp_path))
        models = {}
        with Checkpointer(experiment.checkpoints_url) as checkpointer:
            for step, seed in steps.items():
                models[step] = make_model(env, num_layers, seed)
                optimizer = nnx.Optimizer(models[step], optax.sgd(0.1), wrt=nnx.Param)
                checkpointer.save(models[step], optimizer, step)
        return experiment, models

    shallow, shallow_models = save_run("shallow", 1, {10: 0, 20: 1})
    deep, _ = save_run("deep", 2, {10: 2})
    shallow_too, _ = save_run("shallow-too", 1, {5: 3})

    checkpoints, groups = load_league([shallow, deep, shallow_too], env, MAX_STEPS)

    # runs of one architecture share a group, in the order they were first seen
    assert [(c.run, c.step) for c in checkpoints] == [
        ("shallow", 10),
        ("shallow", 20),
        ("shallow-too", 5),
        ("deep", 10),
    ]
    assert [group.count for group in groups] == [3, 1]

    stacked = groups[0].state["action_embedder"]["embedding_table"]
    for i, step in enumerate((10, 20)):
        saved = shallow_models[step].action_embedder.embedding_table[...]
        assert jnp.array_equal(stacked[i], saved)


def with_games(config, num: int):
    environment = config.environment.model_copy(update={"num": num})
    return config.model_copy(update={"environment": environment})


def test_league_env_is_the_first_runs_unless_a_config_is_given(tmp_path):
    base_config = load_config(CONFIG_FILE.read_text())
    first = Experiment.from_config("first", with_games(base_config, 64), str(tmp_path))
    second = Experiment.from_config("second", base_config, str(tmp_path))
    console = Console(record=True)

    environment, max_steps = league_env([first, second], None, console)

    assert environment == first.config.environment
    assert max_steps == first.config.max_env_steps
    assert "second trained with different env settings" in console.export_text()

    # the same tasks, scaled down and cut short
    small = with_games(base_config, 8).model_copy(update={"max_env_steps": 16})
    small_file = tmp_path / "small.json"
    small_file.write_text(small.model_dump_json())
    console = Console(record=True)

    environment, max_steps = league_env([first, second], str(small_file), console)

    assert environment == small.environment
    assert max_steps == 16
    assert console.export_text() == ""


def test_league_env_refuses_tasks_the_runs_did_not_train_on(tmp_path):
    base_config = load_config(CONFIG_FILE.read_text())
    run = Experiment.from_config("run", base_config, str(tmp_path))
    other_tasks = str(CONFIG_DIR / "koth.json")

    with pytest.raises(ValueError, match="run trained on different tasks"):
        league_env([run], other_tasks, Console())

    other = Experiment.from_config(
        "other", load_config(Path(other_tasks).read_text()), str(tmp_path)
    )
    with pytest.raises(ValueError, match="other trained on different tasks"):
        league_env([run, other], None, Console())
