"""Tests for starting a run from another run's weights."""

from pathlib import Path

import jax.numpy as jnp
import pytest
from flax import nnx

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.config import Config, load_config
from mapox_trainer.experiment import Experiment
from mapox_trainer.train import load_init_weights

CONFIG_FILE = Path(__file__).parent.parent / "config" / "return.json"


class TinyModel(nnx.Module):
    def __init__(self, width: int, seed: int = 0):
        rngs = nnx.Rngs(seed)
        self.encoder = nnx.Linear(3, width, rngs=rngs)
        self.layers = nnx.List([nnx.Linear(width, width, rngs=rngs) for _ in range(2)])


@pytest.fixture
def config() -> Config:
    return load_config(CONFIG_FILE.read_text())


def make_source(base_dir: Path, config: Config, steps: dict[int, nnx.Module]):
    experiment = Experiment.from_config("source", config, str(base_dir))
    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        for step, model in steps.items():
            checkpointer.save(model, step)


def new_run(base_dir: Path, config: Config, **fields) -> Experiment:
    config = config.model_copy(update={"init_from": "source", **fields})
    return Experiment.from_config(
        "new", config, str(base_dir), create_directories=False
    )


def test_copies_the_latest_checkpoint(tmp_path, config):
    early, late = TinyModel(4, seed=1), TinyModel(4, seed=2)
    make_source(tmp_path, config, {5: early, 10: late})

    model = TinyModel(4, seed=3)
    load_init_weights(model, new_run(tmp_path, config))

    assert jnp.array_equal(model.encoder.kernel[...], late.encoder.kernel[...])
    assert jnp.array_equal(model.layers[1].bias[...], late.layers[1].bias[...])


def test_does_nothing_without_init_from(tmp_path, config):
    model = TinyModel(4, seed=3)
    before = model.encoder.kernel[...]

    load_init_weights(model, new_run(tmp_path, config, init_from=None))

    assert jnp.array_equal(model.encoder.kernel[...], before)


def test_rejects_a_different_model_config(tmp_path, config):
    make_source(tmp_path, config, {5: TinyModel(4)})
    model_config = config.learner.model
    deeper = config.learner.model_copy(
        update={
            "model": model_config.model_copy(
                update={"num_layers": model_config.num_layers + 1}
            )
        }
    )

    with pytest.raises(ValueError, match="learner.model differs"):
        load_init_weights(TinyModel(4), new_run(tmp_path, config, learner=deeper))


def test_rejects_parameters_of_a_different_shape(tmp_path, config):
    make_source(tmp_path, config, {5: TinyModel(4)})

    with pytest.raises(ValueError, match="not compatible with the stored shape"):
        load_init_weights(TinyModel(8), new_run(tmp_path, config))


def test_rejects_a_run_without_checkpoints(tmp_path, config):
    make_source(tmp_path, config, {})

    with pytest.raises(ValueError, match="no checkpoints"):
        load_init_weights(TinyModel(4), new_run(tmp_path, config))
