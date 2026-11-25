import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_eval_plot(df: pd.DataFrame, output_file: str) -> None:
    """One panel per task: mean first-life return against checkpoint step, a
    line per run with its 95% interval."""
    tasks = list(dict.fromkeys(df["task"].tolist()))
    runs = list(dict.fromkeys(df["run"].tolist()))  # preserve insertion order
    palette = dict(zip(runs, sns.color_palette("husl", n_colors=len(runs) or 1)))

    columns = min(len(tasks), 4)
    rows = -(-len(tasks) // columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False
    )
    for ax in axes.flat[len(tasks) :]:
        ax.set_visible(False)

    z = 1.96  # approx 95% interval assuming normality
    for ax, task in zip(axes.flat, tasks):
        task_df = df[df["task"] == task]
        for run in runs:
            run_df = task_df[task_df["run"] == run].sort_values("step")
            steps = run_df["step"].to_numpy()
            mean = run_df["mean"].to_numpy()
            sem = run_df["sem"].to_numpy()

            ax.plot(
                steps, mean, color=palette[run], linewidth=2.0, marker="o", label=run
            )
            ax.fill_between(
                steps, mean - z * sem, mean + z * sem, color=palette[run], alpha=0.2
            )

        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Mean first-life return")
        ax.set_title(task)
        ax.grid(True, alpha=0.2)

    axes[0][0].legend(title="Run", loc="best", fontsize=9)
    fig.tight_layout()

    fig.savefig(output_file)
    plt.close(fig)


def main():
    df = pd.read_csv("./eval.csv")
    save_eval_plot(df, "./output.png")


if __name__ == "__main__":
    main()
