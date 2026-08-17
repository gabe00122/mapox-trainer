"""List training runs in results/ ordered by date, with environment highlighted."""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple

from rich.console import Console
from rich.table import Table

from mapox_trainer.experiment import Experiment


class RunInfo(NamedTuple):
    name: str
    env_type: str
    start_time: datetime | None
    seed: int | Literal["random"]
    layers: int
    steps: int
    checkpoints: int
    log_lines: int
    final_reward: float | None

COLOR_PALETTE = [
    "green", "red", "cyan", "magenta", "yellow",
    "blue", "bright_yellow", "bright_magenta", "bright_green", "bright_cyan",
]


def assign_env_colors(runs: list[RunInfo]) -> dict[str, str]:
    env_types = sorted({r.env_type for r in runs})
    return {env: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, env in enumerate(env_types)}


def load_run(run_dir: Path) -> RunInfo | None:
    if not (run_dir / "config.json").exists():
        return None

    try:
        exp = Experiment.load(run_dir.name, base_dir=str(run_dir.parent))
    except Exception:
        return None

    checkpoints_dir = run_dir / "checkpoints"
    n_ckpts = len(list(checkpoints_dir.iterdir())) if checkpoints_dir.is_dir() else 0

    logs_path = run_dir / "logs.jsonl"
    n_logs = 0
    final_reward = None
    if logs_path.exists():
        tail: deque[str] = deque(maxlen=50)
        with logs_path.open() as f:
            for line in f:
                n_logs += 1
                tail.append(line)
        rewards = []
        for line in tail:
            try:
                entry = json.loads(line)
                if "env/rewards" in entry:
                    rewards.append(entry["env/rewards"])
            except json.JSONDecodeError:
                pass
        if rewards:
            final_reward = sum(rewards) / len(rewards)

    return RunInfo(
        name=run_dir.name,
        env_type=exp.config.environment.env_type,
        start_time=exp.meta.start_time,
        seed=exp.config.seed,
        layers=exp.config.learner.model.num_layers,
        steps=exp.config.update_steps,
        checkpoints=n_ckpts,
        log_lines=n_logs,
        final_reward=final_reward,
    )


def build_table(runs: list[RunInfo]) -> Table:
    env_colors = assign_env_colors(runs)

    table = Table(title="Training Runs", show_lines=False)
    table.add_column("Date", style="dim")
    table.add_column("Run Name", style="bold")
    table.add_column("Environment")
    table.add_column("Layers", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("Ckpts", justify="right")
    table.add_column("Logs", justify="right")
    table.add_column("Reward", justify="right")

    for run in runs:
        date_str = run.start_time.strftime("%Y-%m-%d %H:%M") if run.start_time else "?"
        color = env_colors.get(run.env_type, "white")
        env_styled = f"[bold {color}]{run.env_type}[/bold {color}]"
        reward_str = f"{run.final_reward:.2f}" if run.final_reward is not None else "-"

        table.add_row(
            date_str,
            run.name,
            env_styled,
            str(run.layers),
            str(run.steps),
            str(run.seed),
            str(run.checkpoints),
            str(run.log_lines),
            reward_str,
        )

    return table


def list_runs_command():
    results_dir = Path("results")
    runs = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run:
            runs.append(run)

    runs.sort(key=lambda r: r.start_time or datetime.min.replace(tzinfo=None))

    console = Console()
    console.print(build_table(runs))
    console.print(f"\n[dim]{len(runs)} runs total[/dim]")
