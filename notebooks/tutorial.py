import marimo

__generated_with = "0.8.10"
app = marimo.App(width="full")


@app.cell
def __(mo):
    mo.md("""
    # Tutorial for `biobench` 

    This notebook demonstrates basic analysis of results from `biobench`.
    We compare BioCLIP v1 and DINOv2 on classification tasks.
    Once we establish that DINOv2 is better, we dig into specific instances where DINOv2 outperforms BioCLIP.

    First, we need to choose reports to analyze.
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    root_selector = mo.ui.text(placeholder="./reports", label="Reports directory")
    root_selector
    return (root_selector,)


@app.cell
def __(json, mo, os, pl, root_selector):
    def explode_struct(df, col, prefix):
        return (
            df.with_columns(**{prefix: pl.col(col).name.prefix_fields(f"{prefix}_")})
            .drop(col)
            .unnest(prefix)
        )

    if os.path.isdir(root_selector.value) and os.listdir(root_selector.value):
        all_reports = pl.DataFrame([
            json.loads(line)
            for filename in os.listdir(root_selector.value)
            for line in open(os.path.join(root_selector.value, filename))
        ])

        all_reports = explode_struct(all_reports, "benchmark.py_args", "benchmark")
        all_reports = explode_struct(all_reports, "benchmark_model", "model")
        all_reports = explode_struct(all_reports, "benchmark_newt_args", "newt")
        all_reports = explode_struct(all_reports, "benchmark_kabr_args", "kabr")

        # .with_columns(
        #     benchmark=pl.col("benchmark.py_args").name.prefix_fields("benchmark_")
        # ).drop("benchmark.py_args").unnest("benchmark").with_columns(
        #     model=pl.col("benchmark_model").name.prefix_fields("model_")
        # ).drop("benchmark_model").unnest("model")

        # prefix_fields.unnest("benchmark.py_args")
    else:
        all_reports = pl.DataFrame([], schema={"name": str})

    all_reports if os.path.isdir(root_selector.value) else mo.md(
        f"Path '{root_selector.value}' does not exist."
    )
    return all_reports, explode_struct


@app.cell
def __(mo):
    classification_tasks = ("NeWT", "FungiCLEF")
    formatted_classification_tasks = "\n".join(
        f"- {task}" for task in classification_tasks
    )

    models = {
        "vit_base_patch14_dinov2.lvd142m": "DINOv2-ViT-L/14",
        "hf-hub:imageomics/bioclip": "BioCLIP",
    }

    mo.md(f"""
    Then we need to select all the classification tasks for DINOv2 and BioCLIP.
    This requires task-specific knowledge.
    These tasks are:

    {formatted_classification_tasks}
    """)
    return classification_tasks, formatted_classification_tasks, models


@app.cell
def __(all_reports, classification_tasks, mo, models, pl):
    reports = all_reports.filter(
        pl.col("name").is_in(classification_tasks)
        & pl.col("model_ckpt").is_in(models.keys())
    )

    mo.md(f"""
    ## Filtered Reports

    {mo.as_html(reports)}
    """)
    return (reports,)


@app.cell
def __(pl, reports):
    # Compute means across these tasks.
    reports.group_by(pl.col("model_ckpt")).agg(pl.col("score").mean())
    return


@app.cell
def __(mo):
    mo.md("""
    # Learnings

    1. It's still really easy to mess up constructing the BenchmarkReport object. I got a bunch of stuff wrong while debugging. The 1-hour time limit is only good if you are 99% sure that it will succeed within an hour. For development, it's way too slow.
    2. The BenchmarkReport object is kind of a pain in the butt to deal with. I want extremely flat JSON objects so I can deal with them in Polars easily.
    3. It's not clear how each object should include additional data in the BenchmarkReport object. Each task type might have similar structure. On the other hand, if you want more than a summary score, then it might require task-specific work. Maybe each task can have a single-number score between 0 and 1 (for benchmark-wide averaging), then a `splits: dict[str, float]` with a custom metric (try to stick with [0, 1] as much as possible) that indicates some kind of split. Then there can be a generic `examples: List[tuple[ExampleId, float, dict[str, object]]]` that is a list of `(ExampleId, score, info)` tuples.
    """)
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    # My imports
    import json
    import os

    import polars as pl

    return json, os, pl


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
