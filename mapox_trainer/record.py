"""Headless policy videos for rust-env runs: one mp4 per task.

Each task is played for one episode by the run's latest checkpoint, with the
env wrapped in mapox's native `RustVideoConfig` recorder (no window, no
display server). The raw fragmented mp4 it writes is remuxed with faststart so
browsers can seek it, and a `manifest.json` describing the videos is written
last, so its presence means the whole set is complete.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from flax import nnx
from mapox.envs.rust_env_numpy import RustEnvConfig, RustEnvNumpy, RustVideoConfig
from pydantic import TypeAdapter

from mapox_trainer.checkpointer import Checkpointer
from mapox_trainer.experiment import Experiment
from mapox_trainer.play import MapoxAgent

# the longest side of a video, before rounding the tile size down
_TARGET_SIZE = 960
# mapox's sprites; tiles at least this big are drawn at whole multiples of it
# (anything in between is letterboxed), smaller ones at any whole pixel size
_SPRITE_PIXELS = 12
# 2px tiles keep even huge maps legible
_MIN_TILE_PIXELS = 2


def _video_size(map_width: int, map_height: int) -> tuple[int, int]:
    tile = _TARGET_SIZE // max(map_width, map_height)
    if tile >= _SPRITE_PIXELS:
        tile -= tile % _SPRITE_PIXELS
    tile = max(_MIN_TILE_PIXELS, tile)

    # yuv420p needs even dimensions
    def even(n: int) -> int:
        return n + n % 2

    return even(map_width * tile), even(map_height * tile)


def _faststart(src: Path, dst: Path) -> None:
    """Moves the index to the front so browsers can seek without the whole file."""
    subprocess.run(
        [
            "ffmpeg",
            *("-hide_banner", "-loglevel", "error", "-nostdin", "-y"),
            *("-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)),
        ],
        check=True,
    )


def record_run(
    name: str,
    out_dir: str,
    base_dir: str = "results",
    seed: int = 0,
    steps: int | None = None,
    fps: int = 15,
) -> dict:
    experiment = Experiment.load(name, base_dir)
    config = experiment.config

    try:
        env_config = TypeAdapter(RustEnvConfig).validate_python(
            config.environment.model_dump()
        )
    except ValueError as err:
        raise ValueError(
            f"{name} uses env {config.environment.env_type!r}; "
            "only rust envs can be recorded"
        ) from err

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        checkpoint_step = checkpointer.mngr.latest_step()
    if checkpoint_step is None:
        raise ValueError(f"{name} has no checkpoints yet")

    max_steps = config.max_env_steps
    steps = steps or max_steps

    # A plain env reads the task list and each task's map size, which fix the
    # video dimensions the recording env must be built with.
    probe = RustEnvNumpy(env_config, max_steps)
    task_names = probe.task_names or [config.environment.env_type.removeprefix("rust_")]
    sizes = []
    for task_id in range(len(task_names)):
        probe.set_enjoy_mode(task_id)
        settings = probe.get_render_settings()
        sizes.append(_video_size(settings.tile_width, settings.tile_height))

    # Every task shares one network (the vocab and view are the multitask
    # union), so the checkpoint is loaded once.
    rngs = nnx.Rngs(default=experiment.default_seed)
    agent = MapoxAgent(experiment, probe, max_steps, probe.num_tasks, rngs)
    del probe

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    videos = []

    for task_id, (task_name, (width, height)) in enumerate(zip(task_names, sizes)):
        raw_dir = out / f".raw-{task_id}"
        video_config = RustVideoConfig(
            env=env_config,
            output_dir=str(raw_dir),
            record_steps=steps,
            interval_steps=None,
            fps=fps,
            width=width,
            height=height,
        )
        env = RustEnvNumpy(video_config, max_steps)
        env.set_enjoy_mode(task_id)

        print(
            f"Recording task {task_id + 1}/{len(task_names)}: {task_name} "
            f"({env.num_agents} agents, {steps} steps, {width}x{height})",
            flush=True,
        )
        started = time.monotonic()

        agent.reset(env.num_agents, seed)
        timestep = env.reset(seed)
        for step in range(steps):
            timestep = env.step(np.asarray(agent.act(timestep)))
            if (step + 1) % 64 == 0 or step + 1 == steps:
                print(f"  {task_name}: step {step + 1}/{steps}", flush=True)

        # The clip finalizes itself on its last frame; dropping the env
        # finalizes a partial one should the schedule and loop ever disagree.
        del env

        # Recording failures are only logged by the rust side, so check for
        # the clip rather than trusting the loop finished.
        raw = sorted(raw_dir.glob("*.mp4"))
        if not raw or raw[0].stat().st_size == 0:
            raise RuntimeError(f"no video was recorded for task {task_name!r}")

        file = f"task-{task_id}.mp4"
        _faststart(raw[0], out / file)
        for path in raw:
            path.unlink()
        raw_dir.rmdir()

        videos.append(
            {
                "task": task_name,
                "file": file,
                "width": width,
                "height": height,
                "frames": steps,
                "fps": fps,
            }
        )
        print(
            f"  {task_name}: done in {time.monotonic() - started:.1f}s", flush=True
        )

    manifest = {
        "run": name,
        "checkpoint_step": int(checkpoint_step),
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "videos": videos,
    }
    # write-then-rename: a manifest on disk always means a complete set
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out / "manifest.json")

    return manifest
