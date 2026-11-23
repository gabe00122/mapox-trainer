"""Tests for saving and restoring the model and optimizer."""

from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.config import OptimizerConfig
from mapox_trainer.optimizer import create_optimizer


class TinyModel(nnx.Module):
    def __init__(self, seed: int):
        rngs = nnx.Rngs(seed)
        self.action_embedder = nnx.Linear(3, 4, rngs=rngs)
        # more than 10 layers, so a sort by layer name would reorder them
        self.layers = nnx.List([nnx.Linear(4, 4, rngs=rngs) for _ in range(12)])


def make(seed: int, optimizer_type: str) -> tuple[TinyModel, nnx.Optimizer]:
    config = OptimizerConfig(
        type=optimizer_type,
        learning_rate=1e-2,
        weight_decay=0.01,
        eps=1e-8,
        beta1=0.9,
        beta2=0.99,
        max_norm=1.0,
    )
    model = TinyModel(seed)
    return model, nnx.Optimizer(model, create_optimizer(config, 100), wrt=nnx.Param)


def step(model: TinyModel, optimizer: nnx.Optimizer, seed: int = 0):
    params = nnx.state(model, nnx.Param)
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 100))
    grads = jax.tree.map(lambda x: jax.random.normal(next(keys), x.shape), params)
    optimizer.update(model, grads)


def same(a, b) -> bool:
    a, b = jax.tree.leaves(nnx.state(a)), jax.tree.leaves(nnx.state(b))
    return len(a) == len(b) and all(jnp.array_equal(x, y) for x, y in zip(a, b))


@pytest.fixture(params=["adamw", "muon"])
def optimizer_type(request) -> str:
    return request.param


def save(directory: Path, model, optimizer, at: int = 1):
    with Checkpointer(str(directory)) as checkpointer:
        checkpointer.save(model, optimizer, at)


def test_round_trips_the_model_and_optimizer(tmp_path, optimizer_type):
    model, optimizer = make(0, optimizer_type)
    step(model, optimizer)
    step(model, optimizer)
    save(tmp_path, model, optimizer)

    restored_model, restored_optimizer = make(1, optimizer_type)
    with Checkpointer(str(tmp_path)) as checkpointer:
        checkpointer.restore(restored_model, 1, restored_optimizer)

    assert same(restored_model, model)
    assert same(restored_optimizer, optimizer)
    assert int(restored_optimizer.step[...]) == 2

    # the restored pair carries on exactly like the original
    step(model, optimizer, seed=1)
    step(restored_model, restored_optimizer, seed=1)
    assert same(restored_model, model)
    assert same(restored_optimizer, optimizer)


def test_restores_the_model_alone(tmp_path, optimizer_type):
    model, optimizer = make(0, optimizer_type)
    step(model, optimizer)
    save(tmp_path, model, optimizer)

    restored_model, fresh_optimizer = make(1, optimizer_type)
    untouched = nnx.clone(fresh_optimizer)
    with Checkpointer(str(tmp_path)) as checkpointer:
        checkpointer.restore_latest(restored_model)

    assert same(restored_model, model)
    assert same(fresh_optimizer, untouched)


def test_reads_checkpoints_saved_without_an_optimizer(tmp_path):
    model, _ = make(0, "adamw")
    # the layout written before the optimizer was checkpointed
    manager = ocp.CheckpointManager(tmp_path.absolute().as_posix())
    manager.save(1, args=ocp.args.StandardSave(nnx.state(model, nnx.Param)))
    manager.close()

    restored_model, optimizer = make(1, "adamw")
    with Checkpointer(str(tmp_path)) as checkpointer:
        checkpointer.restore(restored_model, 1)
        assert same(restored_model, model)

        with pytest.raises(KeyError, match="optimizer"):
            checkpointer.restore(restored_model, 1, optimizer)


def test_rejects_an_optimizer_of_another_type(tmp_path):
    model, optimizer = make(0, "adamw")
    save(tmp_path, model, optimizer)

    restored_model, muon = make(1, "muon")
    with Checkpointer(str(tmp_path)) as checkpointer, pytest.raises(ValueError):
        checkpointer.restore(restored_model, 1, muon)
