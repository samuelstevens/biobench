import marimo

__generated_with = "0.8.15"
app = marimo.App()


@app.cell
def __():
    import os.path
    import sqlite3

    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, os, pl, sqlite3


@app.cell
def __(mo):
    mo.md(
        """
        # Experiments with biobench

        This notebook provides several examples of different experimental questions that [biobench](https://github.com/samuelstevens/biobench) can help answer.

        biobench saves results to a `reports/reports.sqlite` file that can be queried to generate graphs, make tables, etc.
        If you don't have such a file, I recommend first running some of the benchmark tasks on your own machine to generate these reports.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Do General Benchmarks Reflect Task-Specific Performance?

        Tasks like ImageNet-1k are often used as general measures of a vision model's representation quality.
        In general, if model $A$ is better than model $B$ on ImageNet-1K under the same setting (linear probing, full fine-tuning, zero-shot, etc), then model $A$ will be better than model $B$ on an arbitrary computer vision task $T$.
        This is especially true at lower accuracies (<80%) or when comparing models with vastly different results on ImageNet.
        However, we would expect this to stop being true at some point.

        We hypothesize that tasks which are:

        * domain-specific tasks
        * not classification tasks
        * having a different image distribution (not object centric photographs)

        ...are likely to stop correlating with ImageNet earlier (at lower ImageNet accuracies).

        Can we use biobench to test this hypothesis?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We test this hypothesis by first sorting the tasks in biobench, trying to guess which tasks are likely to stop correlating with ImageNet-1K first, using the bullet points above.

        I think iWildCam is likely going to stop correlating first because camera trap images are so unlike object-centric photos, then the Beluga whale and leopard re-ID tasks are very different to classification, then FishNet and the Ages tasks which are more object-centric but still different, followed by the various classification tasks.

        For each of the models and tasks, we will plot a scatter graph comparing the correlation between the ImageNet-1K performance and the task performance.
        """
    )
    return


@app.cell
def __():
    tasks = [
        "iWildCam",
        "BelugaID",
        "LeopardID",
        "FishNet",
        "Ages",
        "Plankton",
        "KABR",
        "NeWT",
        "Birds525-1shot",
        "Pl@ntNet",
        "iNat21",
    ]

    imagenet_task = "ImageNet-1K"

    tasks
    return imagenet_task, tasks


@app.cell
def __(os, pl, sqlite3):
    def dict_factory(cursor, row):
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    conn = sqlite3.connect(os.path.join("reports", "reports.sqlite"))
    conn.row_factory = dict_factory
    stmt = """
    SELECT model_ckpt, task_name, mean_score, MAX(posix) AS posix 
    FROM reports 
    GROUP BY model_ckpt, task_name
    ORDER BY model_ckpt ASC;
    """
    df = pl.DataFrame(conn.execute(stmt, ()).fetchall())
    df
    return conn, df, dict_factory, stmt


@app.cell
def __(alt, df):
    alt.Chart(
        df.pivot("task_name", index="model_ckpt", values="mean_score")
    ).mark_point().encode(
        x="ImageNet-1K",
        y="FishNet",
        tooltip="model_ckpt",
    )
    return


@app.cell
def __(alt, df):
    alt.Chart(df).mark_point().encode(
        column="task_name", y="mean_score", color="model_ckpt", tooltip="model_ckpt"
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
