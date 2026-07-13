from flax import nnx
from mapox.play import enjoy

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.envs import create_env_factory
from mapox_trainer.experiment import Experiment
from mapox_trainer.model.network import TransformerActorCritic


def load_policy(
    experiment: Experiment,
    env,
    max_steps,
    task_count,
    rngs: nnx.Rngs,
) -> TransformerActorCritic:
    model = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=max_steps,
        task_count=task_count,
        rngs=rngs,
    )

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        model = checkpointer.restore_latest(model)

    return model


def play_from_run(
    name: str,
    human_control: bool,
    pov: bool,
    seed: int,
    env_name: str | None = None,
    video_path: str | None = None,
    size: int = 500,
    fps: int = 15,
):
    experiment = Experiment.load(name, "results")
    config = experiment.config
    rngs = nnx.Rngs(seed)

    env_factory = create_env_factory()
    env, task_count = env_factory.create_env(
        config.environment, config.max_env_steps, 1, env_name
    )

    agent = load_policy(
        experiment,
        env,
        config.max_env_steps,
        task_count,
        rngs,
    )
    agent_state = agent.initialize_carry(env.num_agents, rngs=rngs)

    enjoy(
        env,
        agent,
        agent_state,
        rngs.env(),
        video_path,
        size,
        fps,
        human_control,
        pov,
    )
