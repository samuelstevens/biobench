import marimo

__generated_with = "0.11.25"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import sqlite3

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    return json, mo, np, os, pl, plt, sqlite3


@app.cell
<<<<<<< Updated upstream
<<<<<<< Updated upstream
def _(os, pl, sqlite3):
    def load_results():
        df = pl.read_database(
            "SELECT predictions.score, experiments.task_name FROM experiments JOIN predictions WHERE experiments.id = predictions.experiment_id",
            sqlite3.connect(os.path.join("results", "results.sqlite")),
        )
=======
=======
>>>>>>> Stashed changes
def __(os, sqlite3):
    conn = sqlite3.connect(os.path.join("reports", "reports.sqlite"))
    conn.row_factory = sqlite3.Row
    return (conn,)


@app.cell
def __(np, plt, sqlite3):
    def plot_task(conn: sqlite3.Connection, task: str) -> plt.Figure | None:
        fig, ax = plt.subplots()
        stmt = "SELECT model_ckpt, task_name, mean_score, confidence_lower, confidence_upper, MAX(posix) FROM reports WHERE task_name = (?) GROUP BY model_ckpt, task_name ORDER BY model_ckpt ASC;"
        data = conn.execute(stmt, (task,)).fetchall()

        if not data:
            return

        xs = [row["model_ckpt"] for row in data]
        ys = [row["mean_score"] for row in data]

        yerr = np.array([ys, ys])
        yerr[0] = np.max(yerr[0] - [row["confidence_lower"] for row in data], 0)
        yerr[1] = [row["confidence_upper"] for row in data] - yerr[1]

        ax.errorbar(xs, ys, yerr, fmt="o", linewidth=2, capsize=6)
        ax.set_title(f"Mean {task} Performance")
        ax.tick_params(axis="x", labelrotation=20)

        fig.tight_layout()

        return fig

    return (plot_task,)
>>>>>>> Stashed changes

        return df

<<<<<<< Updated upstream
    df = load_results()
    df
    return df, load_results
=======
@app.cell
def __(conn, plot_task):
    plot_task(conn, "KABR")
    return


@app.cell
def __(conn, plot_task):
    plot_task(conn, "NeWT")
    return


@app.cell
def __(conn, plot_task):
    plot_task(conn, "Pl@ntNet")
    return


@app.cell
def __(conn, plot_task):
    plot_task(conn, "iWildCam")
    return


@app.cell
def __():
    return
>>>>>>> Stashed changes


if __name__ == "__main__":
    app.run()
