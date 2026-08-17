import jax
import pygame
from craftax.craftax.constants import (
    BLOCK_PIXEL_SIZE_HUMAN,
    Achievement,
)
from craftax.craftax.play_craftax import CraftaxRenderer
from flax import nnx
from jax import numpy as jnp
from mapox.utils.video_writer import save_video

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.envs.third_party.craftax_wrapper import (
    CraftaxConfig,
    CraftaxEnvironment,
)
from mapox_trainer.experiment import Experiment
from mapox_trainer.model.network import TransformerActorCritic
from mapox_trainer.train import add_seq_dim


def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(
                f"{Achievement(i).name} ({new_achievements.sum()}/{len(Achievement)})"
            )


def main(name: str, base_dir: str = "results", seed: int = 121):
    experiment: Experiment = Experiment.load(name, base_dir)
    max_steps = experiment.config.max_env_steps

    env = CraftaxEnvironment(CraftaxConfig(), max_steps)
    rngs = nnx.Rngs(default=seed)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=max_steps,
        task_count=1,
        rngs=rngs,
    )

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        model = checkpointer.restore_latest(model)

    rng = rngs.env()
    rng, _rng = jax.random.split(rng)
    env_state, ts = env.reset(_rng)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN

    renderer = CraftaxRenderer(
        env._env, env._env_params, pixel_render_size=pixel_render_size
    )
    renderer.render(env_state.cstate)

    # step_fn = jax.jit(env.step)

    @nnx.jit
    def step(timestep, kv_cache, env_state, rngs):
        action_key = rngs.action()
        env_key = rngs.env()
        _, policy, kv_cache = model(add_seq_dim(timestep), kv_cache)
        actions = policy.sample(seed=action_key)
        actions = jnp.squeeze(actions, axis=-1)

        env_state, timestep = env.step(env_state, actions, env_key)

        return env_state, timestep, kv_cache, rngs

    carry = model.initialize_carry(1, rngs)

    frames = []
    reward = 0.0
    # total_reward = 0.0
    total_episodes = 0

    time = 0
    while not renderer.is_quit_requested():
        env_state, ts, carry, rngs = step(ts, carry, env_state, rngs)

        renderer.render(env_state.cstate)
        renderer.update()
        img_data = pygame.surfarray.array3d(pygame.display.get_surface())
        frames.append(img_data)

        # clock.tick(5)
        time += 1
        reward += ts.reward.squeeze().item()

        if ts.terminated or time >= experiment.config.max_env_steps:
            print(time)

            time = 0
            carry = model.initialize_carry(1, rngs)
            env_state, ts = env.reset(rngs.env())

            # total_reward += reward
            # reward = 0
            total_episodes += 1
            print(reward / total_episodes)

            if total_episodes >= 1:
                break

    save_video(frames, "videos/craftax.mp4", 8)


if __name__ == "__main__":
    main("great-spirit-8qf8n0")
