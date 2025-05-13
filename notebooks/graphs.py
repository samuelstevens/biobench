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
def _(os, pl, sqlite3):
    def load_results():
        df = pl.read_database(
            "SELECT predictions.score, experiments.task_name FROM experiments JOIN predictions WHERE experiments.id = predictions.experiment_id",
            sqlite3.connect(os.path.join("results", "results.sqlite")),
        )

        return df

    df = load_results()
    df
    return df, load_results


if __name__ == "__main__":
    app.run()
