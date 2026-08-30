"""Export a trained TransformerActorCritic for the rust `mapox-burn` crate.

`uv run scripts/export_burn_policy.py <run>` loads the run's latest checkpoint
(like `demo_policy.py` does) and writes `results/<run>/burn/policy.safetensors`:
every `nnx.Param` leaf as float32 under its dotted state path, in the raw JAX
layout (the rust loader owns all layout conversion), with the model config, env
config and specs embedded as JSON strings in the safetensors metadata. A
`policy.io.safetensors` with a few reference steps (inputs and the f32 model's
log-probs/values) is written next to it so the rust side can verify the import:

    cargo run -p mapox-burn --example verify -- policy.safetensors policy.io.safetensors

`--fixtures <dir>` instead writes two tiny random-init models plus reference
steps, used as committed test fixtures by `mapox-burn`'s parity tests. The
reference outputs are generated on CPU in float32 (the checkpoint's compute
dtype is usually bfloat16; the rust side always computes in f32, so f32
references are the meaningful comparison and keep the tolerance tight).

Like `demo_policy.py`, only `--rust-env`-trained find_return runs fit: the env
config is embedded for the rust demo to rebuild, and the rust env emits one
observation channel where the JAX env emits four.
"""

import os

# before jax imports: reference outputs must be plain f32 (no GPU tf32 matmuls)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from mapox.envs.rust_env import RustEnv
from mapox.specs import ObservationSpec
from mapox.timestep import TimeStep
from safetensors.numpy import save_file

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.config import (
    AttentionConfig,
    FeedForwardConfig,
    GridCnnObsEncoderConfig,
    HlGaussConfig,
    LayerConfig,
    MseCriticConfig,
    TransformerActorCriticConfig,
)
from mapox_trainer.experiment import Experiment
from mapox_trainer.model.network import TransformerActorCritic
from mapox_trainer.util import add_seq_dim

FORMAT = "mapox-burn-v1"

jax.config.update("jax_default_matmul_precision", "highest")


def flatten_params(model: TransformerActorCritic) -> dict[str, np.ndarray]:
    state = nnx.state(model, nnx.Param)
    tensors = {}
    for path, variable in state.flat_state():
        name = ".".join(str(part) for part in path)
        array = np.asarray(jax.device_get(variable.get_value()))
        tensors[name] = array.astype(np.float32)
    return tensors


def make_metadata(
    model_config: TransformerActorCriticConfig,
    obs_spec: ObservationSpec,
    action_dim: int,
    max_seq_length: int,
    env_config_json: str | None,
    source: str,
) -> dict[str, str]:
    assert isinstance(obs_spec.max_value, tuple)
    metadata = {
        "format": FORMAT,
        "model_config": model_config.model_dump_json(),
        "obs_shape": json.dumps(list(obs_spec.shape)),
        "obs_max_value": json.dumps(list(obs_spec.max_value)),
        "action_dim": str(action_dim),
        "max_seq_length": str(max_seq_length),
        "source": source,
    }
    if env_config_json is not None:
        metadata["env_config"] = env_config_json
    return metadata


def run_reference_steps(
    model: TransformerActorCritic,
    obs_spec: ObservationSpec,
    action_dim: int,
    num_agents: int,
    steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Steps the model with its carry on synthetic inputs, recording what the
    rust implementation must reproduce: the policy's normalized log-probs and
    the scalar value, per step."""
    assert isinstance(obs_spec.max_value, tuple)
    rng = np.random.default_rng(seed)
    carry = model.initialize_carry(num_agents, nnx.Rngs(0))

    channel_max = np.asarray(obs_spec.max_value, np.uint16)
    inputs = {
        "obs": np.empty((steps, num_agents, *obs_spec.shape), np.uint16),
        "reward": np.empty((steps, num_agents), np.float32),
        "last_action": np.empty((steps, num_agents), np.uint16),
        "action_mask": np.empty((steps, num_agents, action_dim), np.uint8),
    }
    outputs = {
        "log_probs": np.empty((steps, num_agents, action_dim), np.float32),
        "value": np.empty((steps, num_agents), np.float32),
    }

    for t in range(steps):
        obs = rng.integers(0, channel_max, size=(num_agents, *obs_spec.shape))
        reward = rng.standard_normal(num_agents).astype(np.float32)
        last_action = rng.integers(0, action_dim, size=(num_agents,))
        mask = rng.random((num_agents, action_dim)) < 0.8
        mask[np.arange(num_agents), rng.integers(0, action_dim, num_agents)] = True

        timestep = TimeStep(
            obs=jnp.asarray(obs, jnp.uint16),
            time=jnp.full((num_agents,), t, jnp.int32),
            terminated=jnp.zeros((num_agents,), jnp.bool_),
            last_action=jnp.asarray(last_action, jnp.uint16),
            reward=jnp.asarray(reward),
            action_mask=jnp.asarray(mask),
            task_ids=jnp.zeros((num_agents,), jnp.int32),
        )
        value_rep, policy, carry = model(add_seq_dim(timestep), carry)

        inputs["obs"][t] = obs
        inputs["reward"][t] = reward
        inputs["last_action"][t] = last_action
        inputs["action_mask"][t] = mask
        # distrax normalizes logits to log-probs, which is also the right
        # parity target: shift-invariant and independent of the mask fill
        outputs["log_probs"][t] = np.asarray(policy.logits.squeeze(axis=1))
        value = model.get_value(value_rep)
        outputs["value"][t] = np.asarray(value.reshape(num_agents))

    return inputs | outputs


def as_float32(config: TransformerActorCriticConfig) -> TransformerActorCriticConfig:
    return config.model_copy(update={"dtype": "float32"})


def export_run(args: argparse.Namespace) -> None:
    experiment = Experiment.load(args.name, args.base_dir)
    env_config = experiment.config.environment

    env = RustEnv(env_config, num_envs=1)
    max_seq_length = experiment.config.max_env_steps
    model = TransformerActorCritic(
        as_float32(experiment.config.learner.model),
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=max_seq_length,
        task_count=1,
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

    tensors = flatten_params(model)
    metadata = make_metadata(
        as_float32(experiment.config.learner.model),
        env.observation_spec,
        env.action_spec.n,
        max_seq_length,
        env_config.model_dump_json(),
        source=f"{experiment.unique_token}@{step}",
    )
    save_file(tensors, out, metadata=metadata)

    total = sum(t.size for t in tensors.values())
    print(f"{out}: {len(tensors)} tensors, {total:,} params, step {step}")

    io = run_reference_steps(
        model,
        env.observation_spec,
        env.action_spec.n,
        num_agents=env.num_agents,
        steps=args.io_steps,
        seed=args.seed,
    )
    io_out = out.with_suffix(".io.safetensors")
    save_file(io, io_out, metadata={"format": FORMAT, "model": out.name})
    print(f"{io_out}: {args.io_steps} reference steps")


FIXTURES = {
    # exercises: GQA attention with qk-norm, post-attn/ffw norms, GLU, gelu,
    # rms_norm, hl_gauss value with value_mlp, and (via steps > max_seq)
    # the kv-cache ring wraparound
    "attn_glu_hlgauss": dict(
        config=TransformerActorCriticConfig(
            obs_encoder=GridCnnObsEncoderConfig(
                kernels=((3, 3), (3, 3), (3, 3)),
                strides=((2, 2), (1, 1), (1, 1)),
                channels=(8, 12),
            ),
            hidden_features=32,
            layer=LayerConfig(
                feed_forward=FeedForwardConfig(size=48, glu=True),
                history=AttentionConfig(
                    num_heads=2, num_kv_heads=1, head_dim=16, use_qk_norm=True
                ),
                use_post_attn_norm=True,
                use_post_ffw_norm=True,
            ),
            num_layers=2,
            value_hidden_dim=24,
            value=HlGaussConfig(min=-10.0, max=10.0, n_logits=11, sigma=0.3),
            activation="gelu",
            norm="rms_norm",
            dtype="float32",
            param_dtype="float32",
        ),
        max_seq_length=16,
        steps=24,
    ),
    # the other branches: plain FF, no qk-norm, layer_norm, silu, mse value
    # with no value_mlp
    "ff_mse": dict(
        config=TransformerActorCriticConfig(
            obs_encoder=GridCnnObsEncoderConfig(
                kernels=((3, 3), (3, 3), (3, 3)),
                strides=((2, 2), (1, 1), (1, 1)),
                channels=(8, 12),
            ),
            hidden_features=32,
            layer=LayerConfig(
                feed_forward=FeedForwardConfig(size=48, glu=False),
                history=AttentionConfig(
                    num_heads=2, num_kv_heads=2, head_dim=16, use_qk_norm=False
                ),
            ),
            num_layers=2,
            value_hidden_dim=None,
            value=MseCriticConfig(),
            activation="silu",
            norm="layer_norm",
            dtype="float32",
            param_dtype="float32",
        ),
        max_seq_length=16,
        steps=24,
    ),
}


def export_fixtures(args: argparse.Namespace) -> None:
    out_dir = Path(args.fixtures)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_spec = ObservationSpec(dtype=jnp.uint16, shape=(11, 11, 1), max_value=(9,))
    action_dim = 5
    num_agents = 3

    for name, spec in FIXTURES.items():
        model = TransformerActorCritic(
            spec["config"],
            obs_spec,
            action_dim,
            max_seq_length=spec["max_seq_length"],
            task_count=1,
            rngs=nnx.Rngs(0),
        )

        metadata = make_metadata(
            spec["config"],
            obs_spec,
            action_dim,
            spec["max_seq_length"],
            env_config_json=None,
            source=f"fixture:{name}",
        )
        model_path = out_dir / f"{name}.safetensors"
        save_file(flatten_params(model), model_path, metadata=metadata)

        io = run_reference_steps(
            model, obs_spec, action_dim, num_agents, spec["steps"], seed=42
        )
        io_path = out_dir / f"{name}.io.safetensors"
        save_file(io, io_path, metadata={"format": FORMAT, "model": model_path.name})
        print(f"{name}: {model_path} + {io_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="run directory name under --base-dir")
    parser.add_argument("--base-dir", default="results")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out", default=None, help="output .safetensors path")
    parser.add_argument("--io-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixtures",
        default=None,
        metavar="DIR",
        help="write tiny random-init parity fixtures to DIR instead",
    )
    args = parser.parse_args()

    if args.fixtures:
        export_fixtures(args)
    elif args.name:
        export_run(args)
    else:
        parser.error("either a run name or --fixtures is required")


if __name__ == "__main__":
    main()
