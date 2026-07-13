import json
import random
from typing import Any, Literal, Tuple

from mapox import EnvironmentConfig, MultiTaskConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mapox_trainer.envs import CraftaxConfig


class GridCnnObsEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    obs_type: Literal["grid_cnn"] = "grid_cnn"
    kernels: tuple[tuple[int, int], ...]
    strides: tuple[tuple[int, int], ...]
    channels: tuple[int, ...]


class FlattenedObsEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    obs_type: Literal["grid_flattened"] = "grid_flattened"


class LinearObsEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    obs_type: Literal["linear"] = "linear"


class AttentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["attention"] = "attention"

    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int | None = None

    rope_max_wavelength: float = 10_000
    use_qk_norm: bool = False


class RnnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["rnn"] = "rnn"

    # carry_dim: int


class FeedForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    size: int
    glu: bool = True


class LayerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feed_forward: FeedForwardConfig
    history: AttentionConfig | RnnConfig | None = Field(
        default=None, discriminator="type"
    )

    use_post_attn_norm: bool = False
    use_post_ffw_norm: bool = False


class MseCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["mse"] = "mse"


class HlGaussConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["hl_gauss"] = "hl_gauss"

    min: float
    max: float
    n_logits: int
    sigma: float


class TransformerActorCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obs_encoder: (
        LinearObsEncoderConfig | GridCnnObsEncoderConfig | FlattenedObsEncoderConfig
    ) = Field(discriminator="obs_type")
    hidden_features: int

    layer: LayerConfig
    num_layers: int

    value_hidden_dim: int | None = None
    value: MseCriticConfig | HlGaussConfig = Field(discriminator="type")

    activation: Literal["relu", "gelu", "silu", "mish"]
    norm: Literal["layer_norm", "rms_norm"]
    kernel_init: Literal["glorot_uniform", "he_uniform", "lecun_uniform", "normal"] = (
        "glorot_uniform"
    )
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    param_dtype: Literal["float32", "bfloat16"] = "float32"


class LoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    use_console: bool = True
    use_tb: bool = False
    use_neptune: bool = False
    use_wandb: bool = False
    use_jsonl: bool = True


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["adamw", "muon"]
    learning_rate: float
    weight_decay: float
    eps: float
    beta1: float
    beta2: float
    max_norm: float


class PPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trainer_type: Literal["ppo"]

    epoch_count: int = 1
    minibatch_count: int = 1
    vf_coef: float = 0.05
    entropy_coef: float = 0.005
    entropy_coef_end: float | None = 0.0
    vf_clip: float = 0.2
    discount: float = 0.95
    gae_lambda: float = 0.95
    normalize_advantage: bool = False


class LearnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer: OptimizerConfig
    model: TransformerActorCriticConfig
    trainer: PPOConfig


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int | Literal["random"] = "random"
    num_envs: int = 1
    max_env_steps: int

    updates_per_jit: int = 1
    update_steps: int
    num_checkpoints: int = 50

    learner: LearnerConfig
    environment: EnvironmentConfig | MultiTaskConfig | CraftaxConfig = Field(
        discriminator="env_type"
    )
    logger: LoggerConfig = LoggerConfig()

    snapshot_league: bool = False


def load_config(json_config: str) -> Config:
    config = Config.model_validate_json(json_config, strict=True)
    if config.seed == "random":
        config = config.model_copy(update={"seed": random.getrandbits(31)})

    return config
