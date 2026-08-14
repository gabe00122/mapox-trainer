import os
import shutil

import jax
import typer

from mapox_trainer.eval import main as eval_main
from mapox_trainer.experiment import Experiment
from mapox_trainer.play import play_from_run
from mapox_trainer.train import train_run
from mapox_trainer.runs import list_runs_command


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def enjoy(
    name: str,
    human: bool = False,
    pov: bool = False,
    seed: int = 0,
    env: str | None = None,
    video_path: str | None = typer.Option(
        None, help="Path to save video; if omitted, no video is recorded."
    ),
    size: int = 960,
    fps: int = 15,
):
    play_from_run(
        name,
        human,
        pov,
        seed,
        env_name=env,
        video_path=video_path,
        size=size,
        fps=fps,
    )


@app.command("train")
def train_cmd(
    config: str = "./config/return.json",
    distributed: bool = False,
    base_dir: str = "./results",
    name: str | None = None,
):
    if distributed:
        jax.distributed.initialize()
    experiment = Experiment.from_config_file(config, base_dir)

    train_run(experiment)


@app.command("profile")
def profile(
    config: str = "./config/return.json",
    distributed: bool = False,
    base_dir: str = "./results",
):
    if distributed:
        jax.distributed.initialize()
    experiment = Experiment.from_config_file(
        config, base_dir, create_directories=False
    )

    train_run(experiment, profile=True)


@app.command("clean")
def clean():
    subfolders = [f.path for f in os.scandir("./results") if f.is_dir()]

    for folder in subfolders:
        checkpoint_count = len(
            [f.path for f in os.scandir(folder + "/checkpoints") if f.is_dir()]
        )

        if checkpoint_count == 0:
            shutil.rmtree(folder)


app.command("eval")(eval_main)
app.command("runs")(list_runs_command)

if __name__ == "__main__":
    app()
