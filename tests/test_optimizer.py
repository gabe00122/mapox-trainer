"""Tests for the optimizer factory."""

from typing import Any, cast

import jax
import jax.numpy as jnp

from mapox_trainer.config import OptimizerConfig
from mapox_trainer.optimizer import create_optimizer


def make_config(**overrides) -> OptimizerConfig:
    fields = {
        "type": "muon",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "eps": 1e-8,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_norm": 1.0,
    }
    return OptimizerConfig.model_validate({**fields, **overrides})


def first_update(config: OptimizerConfig):
    # the action embedder stays on adamw, a layer's 2D kernel goes to muon
    key_embed, key_kernel = jax.random.split(jax.random.PRNGKey(0))
    params = {
        "action_embedder": {"embedding_table": jnp.ones((5, 8))},
        "layers": {"ffn": {"up_proj": {"kernel": jnp.ones((8, 16))}}},
    }
    grads = {
        "action_embedder": {
            "embedding_table": jax.random.normal(key_embed, (5, 8))
        },
        "layers": {
            "ffn": {"up_proj": {"kernel": jax.random.normal(key_kernel, (8, 16))}}
        },
    }

    tx = create_optimizer(config, update_steps=100)
    updates, _ = tx.update(grads, tx.init(params), params)
    updates = cast(dict[str, Any], updates)
    return (
        updates["action_embedder"]["embedding_table"],
        updates["layers"]["ffn"]["up_proj"]["kernel"],
    )


class TestMuonAdamLearningRate:
    def test_defaults_to_learning_rate(self):
        adam_default, muon_default = first_update(make_config())
        adam_same, muon_same = first_update(make_config(adam_learning_rate=1e-3))

        assert jnp.array_equal(adam_default, adam_same)
        assert jnp.array_equal(muon_default, muon_same)

    def test_only_scales_adam_parameters(self):
        adam_base, muon_base = first_update(make_config())
        adam_fast, muon_fast = first_update(make_config(adam_learning_rate=1e-2))

        assert jnp.allclose(adam_fast, 10 * adam_base, rtol=1e-5)
        assert jnp.array_equal(muon_fast, muon_base)
        assert jnp.abs(muon_base).max() > 0
