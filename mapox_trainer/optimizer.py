import jax
import optax

from mapox_trainer.config import OptimizerConfig


def create_optimizer(
    optimizer_config: OptimizerConfig, update_steps: int
) -> optax.GradientTransformation:
    match optimizer_config.type:
        case "adamw":
            return optax.chain(
                optax.clip_by_global_norm(optimizer_config.max_norm),
                optax.adamw(
                    optax.linear_schedule(
                        optimizer_config.learning_rate, 0, update_steps
                    ),
                    b1=optimizer_config.beta1,
                    b2=optimizer_config.beta2,
                    weight_decay=optimizer_config.weight_decay,
                    eps=optimizer_config.eps,
                ),
            )
        case "muon":
            return optax.contrib.muon(
                optax.linear_schedule(
                    optimizer_config.learning_rate, 0, update_steps
                ),
                weight_decay=optimizer_config.weight_decay,
                adam_weight_decay=optimizer_config.weight_decay,
                adam_b1=optimizer_config.beta1,
                adam_b2=optimizer_config.beta2,
                muon_weight_dimension_numbers=muon_dim_numbers_fn,
            )


EXCLUDE_TOP = {
    "embedding",
    "value_head",
    "action_encoder",
    "reward_encoder",
    "obs_encoder",
    "obs_decoder",
}


def muon_dim_numbers_fn(params):
    """Return a pytree mirroring `params` with:
    - MuonDimensionNumbers() for leaves you want Muon on
    - None for leaves you want AdamW on
    """

    def decide(path, x):
        # ignore whole subtrees by their top-level module name
        top = path[0].key if path else None
        if top in EXCLUDE_TOP:
            return None
        # Only 2D weights are eligible for Muon by default
        if hasattr(x, "ndim") and x.ndim == 2:
            return (
                optax.contrib.MuonDimensionNumbers()
            )  # default: treat as a matrix (no reshape)
        return None  # biases, vectors, embeddings, 3D/4D kernels -> AdamW

    return jax.tree.map_with_path(decide, params)
