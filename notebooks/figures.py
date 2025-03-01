import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    import sqlite3
    import numpy as np

    import polars as pl
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    import sys

    if "." not in sys.path:
        sys.path.append(".")
    import biobench.reporting

    return biobench, math, mo, mpl, np, pl, plt, sqlite3, sys


@app.cell
def _(pl, sqlite3):
    conn = sqlite3.connect("results/results.sqlite")

    preds_df = pl.read_database(
        "SELECT results.task_name, results.task_cluster, results.task_subcluster, results.model_ckpt, predictions.score, predictions.n_train FROM results JOIN predictions ON results.rowid = predictions.result_id",
        conn,
    )
    return conn, preds_df


@app.cell
def _(preds_df):
    preds_df
    return


@app.cell
def _(np, pl, preds_df):
    def bootstrap_ci(scores, n_resamples=1000):
        scores = np.array(scores)
        # Vectorized bootstrap: sample all at once
        boot_samples = np.random.choice(
            scores, size=(n_resamples, len(scores)), replace=True
        )
        boot_means = boot_samples.mean(axis=1)
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        return np.mean(scores), ci_lower, ci_upper

    def boot_func(scores):
        mean, ci_lower, ci_upper = bootstrap_ci(scores, 1000)
        return {"mean": mean, "ci_lower": ci_lower, "ci_upper": ci_upper}

    df = (
        preds_df.group_by(
            "task_name", "task_cluster", "task_subcluster", "n_train", "model_ckpt"
        )
        .all()
        .with_columns(pl.col("score").map_elements(boot_func).alias("boot"))
        .with_columns(
            mean=pl.col("boot").struct.field("mean"),
            ci_lower=pl.col("boot").struct.field("ci_lower"),
            ci_upper=pl.col("boot").struct.field("ci_upper"),
        )
    )

    df
    return boot_func, bootstrap_ci, df


@app.cell
def _(biobench, df, pl, plt):
    fig, ax = plt.subplots()
    for model, color in zip(
        sorted(df.get_column("model_ckpt").unique().to_list()),
        biobench.reporting.ALL_RGB01[1::2],
    ):
        filtered_df = df.filter((pl.col("model_ckpt") == model)).sort("n_train")

        means = filtered_df.get_column("mean").to_list()
        lowers = filtered_df.get_column("ci_lower").to_list()
        uppers = filtered_df.get_column("ci_upper").to_list()
        xs = filtered_df.get_column("n_train").to_list()

        ax.plot(xs, means, marker="o", label=model, color=color)
        ax.fill_between(xs, lowers, uppers, alpha=0.2, color=color, linewidth=0)
        ax.set_xlabel("Number of Training Samples")
        ax.set_ylabel("Mean Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("Counting")
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xlim(-0.15, 130)

        ax.legend(loc="best")

    fig.tight_layout()
    fig
    return ax, color, fig, filtered_df, lowers, means, model, uppers, xs


if __name__ == "__main__":
    app.run()
