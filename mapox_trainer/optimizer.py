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
            # no grad clipping: muon's orthogonalization is invariant to
            # gradient scale, so max_norm is unused here
            return optax.contrib.muon(
                optax.linear_schedule(optimizer_config.learning_rate, 0, update_steps),
                # beta1 drives both momentum terms: muon's and adam's
                beta=optimizer_config.beta1,
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay,
                adam_weight_decay=optimizer_config.weight_decay,
                adam_b1=optimizer_config.beta1,
                adam_b2=optimizer_config.beta2,
                muon_weight_dimension_numbers=muon_dim_numbers_fn,
            )


# top-level modules that stay on AdamW: embeddings (the action embedder is
# also the tied policy head), input encoders and the value head
EXCLUDE_TOP = {
    "action_embedder",
    "task_embedder",
    "reward_encoder",
    "obs_encoder",
    "value_head",
}

# attention kernels are 3D, so they need explicit axes to be treated as one
# matrix: qkv_proj is (d_model, heads, head_dim), out is (heads, head_dim, d_model)
ATTENTION_DIM_NUMBERS = {
    "qkv_proj": optax.contrib.MuonDimensionNumbers(
        reduction_axis=0, output_axis=(1, 2)
    ),
    "out": optax.contrib.MuonDimensionNumbers(reduction_axis=(0, 1), output_axis=2),
}


def _path_names(path) -> list:
    return [
        getattr(k, "key", getattr(k, "idx", getattr(k, "name", None))) for k in path
    ]


def muon_dim_numbers_fn(params):
    """Return a pytree mirroring `params` with:
    - MuonDimensionNumbers for leaves you want Muon on
    - None for leaves you want AdamW on
    """

    def decide(path, x):
        names = _path_names(path)
        # ignore whole subtrees by their top-level module name
        if names and names[0] in EXCLUDE_TOP:
            return None
        if not hasattr(x, "ndim"):
            return None
        if x.ndim == 3 and "history" in names and "kernel" in names:
            for module, dim_numbers in ATTENTION_DIM_NUMBERS.items():
                if module in names:
                    return dim_numbers
        if x.ndim == 2:
            return optax.contrib.MuonDimensionNumbers()
        return None  # biases, norm scales, conv kernels -> AdamW

    return jax.tree.map_with_path(decide, params)
