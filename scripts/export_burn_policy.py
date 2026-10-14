"""Export a trained TransformerActorCritic for the rust `mapox-burn` crate.

`uv run scripts/export_burn_policy.py <run>` loads the run's latest checkpoint
(like `demo_policy.py` does) and writes `results/<run>/burn/policy.safetensors`:
every `nnx.Param` leaf (float32, or `--dtype float16`/`bfloat16` to halve the
file for the web build) under its dotted state path, in the raw JAX layout (the rust loader owns all layout conversion), with the model config, env
config and specs embedded as JSON strings in the safetensors metadata:

    cargo run -p mapox-burn --example demo -- policy.safetensors

Only rust-env runs fit: the env config is embedded for the rust demo to
rebuild, and the obs encoder embeds a single observation channel. Multitask
(`rust_multi`) runs carry a task embedding, sized by `num_tasks` in the
metadata; the rust viewer plays one task at a time.
"""

import os

# before jax imports: a one-off export doesn't need to claim the GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
from flax import nnx
from mapox.envs.rust_env_numpy import RustEnvConfig
from pydantic import TypeAdapter
from safetensors.numpy import save_file

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.envs import create_env_factory
from mapox_trainer.experiment import Experiment
from mapox_trainer.model.network import TransformerActorCritic

FORMAT = "mapox-burn-v1"

# numpy has no native bfloat16; ml_dtypes (a jax dependency) provides it and
# safetensors writes it as BF16
DTYPES = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
}


def flatten_params(
    model: TransformerActorCritic, dtype: np.dtype
) -> dict[str, np.ndarray]:
    state = nnx.state(model, nnx.Param)
    tensors = {}
    for path, variable in state.flat_state():
        name = ".".join(str(part) for part in path)
        array = np.asarray(jax.device_get(variable.get_value()))
        stored = array.astype(dtype)
        if not np.isfinite(stored).all():
            raise SystemExit(f"{name} overflows {dtype.name}; export as float32")
        tensors[name] = stored
    return tensors


def export_run(args: argparse.Namespace) -> None:
    experiment = Experiment.load(args.name, args.base_dir)
    # revalidated as the rust config (as play.py does) so the dump carries
    # every field: the run's own config only holds the ones it set, and the
    # rust side has no defaults to fill in the rest
    env_config = TypeAdapter(RustEnvConfig).validate_python(
        experiment.config.environment.model_dump()
    )
    # the rust side always computes in f32
    model_config = experiment.config.learner.model.model_copy(
        update={"dtype": "float32"}
    )

    max_seq_length = experiment.config.max_env_steps
    # the run's env config carries its own vectorization (rust_vec wrapper);
    # the burn crate builds the same env from the embedded config JSON
    env = create_env_factory().create_env(env_config, max_seq_length)
    model = TransformerActorCritic(
        model_config,
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=max_seq_length,
        task_count=env.num_tasks,
        rngs=nnx.Rngs(experiment.params_seed),
    )

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        step = args.step
        if step is None:
            step = checkpointer.mngr.latest_step() or 0
        try:
            model = checkpointer.restore(model, step)
        except ValueError as err:
            raise SystemExit(
                f"could not restore {experiment.unique_token}: {err}\n"
                "shape mismatches here usually mean the run was trained on "
                "the JAX env (4 observation channels) rather than with "
                "`pmarl train --rust-env` (1 channel)."
            ) from err

    out = (
        Path(args.out)
        if args.out
        else Path(experiment.experiment_url) / "burn" / "policy.safetensors"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    obs_spec = env.observation_spec
    max_value = obs_spec.max_value
    assert isinstance(max_value, tuple)
    metadata = {
        "format": FORMAT,
        "model_config": model_config.model_dump_json(),
        "env_config": env_config.model_dump_json(),
        "obs_shape": json.dumps(list(obs_spec.shape)),
        "obs_max_value": json.dumps(list(max_value)),
        "action_dim": str(env.action_spec.n),
        "num_tasks": str(env.num_tasks),
        "max_seq_length": str(max_seq_length),
        "source": f"{experiment.unique_token}@{step}",
    }

    tensors = flatten_params(model, DTYPES[args.dtype])
    save_file(tensors, out, metadata=metadata)

    total = sum(t.size for t in tensors.values())
    print(f"{out}: {len(tensors)} tensors, {total:,} params, step {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="run directory name under --base-dir")
    parser.add_argument("--base-dir", default="results")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out", default=None, help="output .safetensors path")
    parser.add_argument(
        "--dtype",
        choices=list(DTYPES),
        default="float32",
        help="storage dtype; the rust loader widens float16/bfloat16 back to f32",
    )
    export_run(parser.parse_args())


if __name__ == "__main__":
    main()
