import marimo

__generated_with = "0.8.15"
app = marimo.App(width="full")


@app.cell
def __():
    import os.path

    import marimo as mo
    import polars as pl
    import requests

    return mo, os, pl, requests


@app.cell
def __(pl, requests):
    model_mapping = {
        "vit_base_patch14_reg4_dinov2.lvd142m": "DinoV2 (ViT-B/14)",
        "hf-hub:imageomics/bioclip": "BioCLIP (ViT-B/16)",
        "ViT-B-16-SigLIP/webli": "SigLIP (ViT-B/16)",
        "ViT-B-16/laion400m_e32": "OpenCLIP (ViT-B/16)",
        "ViT-B-16/openai": "CLIP (ViT-B/16)",
        "RN50/openai": "CLIP (ResNet50)",
    }

    r = requests.get(
        "https://raw.githubusercontent.com/samuelstevens/biobench/refs/heads/main/reports/results.csv",
        stream=True,
    )
    r.raise_for_status()
    df = (
        pl.read_csv(r.raw)
        .drop("Birds525")
        .rename({"mean": "Mean", "model": "Model", "Birds525-1shot": "Birds525"})
        .with_columns(Model=pl.col("Model").replace(model_mapping))
        .sort(by="Mean", descending=True)
    )

    tasks = [col for col in df.columns if col != "Model"]
    return df, model_mapping, r, tasks


@app.cell
def __(mo):
    mo.md(r"""# `biobench` Leaderboard""")
    return


@app.cell
def __(mo, tasks):
    _html = " ".join(f"{{{task}}}" for task in tasks)

    task_boxes = mo.md(_html).batch(**{
        task: mo.ui.checkbox(label=task, value=True) for task in tasks
    })

    group_boxes = mo.md("Group boxes not implemented yet.")

    mo.md(f"## Tasks\n\n{task_boxes}\n\n{group_boxes}")
    return group_boxes, task_boxes


@app.cell
def __(df, mo, task_boxes):
    def _percentage(f):
        return f"{f * 100:.1f}"

    always_included = ["Model"]

    mo.ui.table(
        df.select(
            always_included + [col for col, box in task_boxes.items() if box.value]
        ),
        selection=None,
        pagination=False,
        show_column_summaries=False,
        freeze_columns_left=["Model"],
        format_mapping={
            "Mean": _percentage,
            "Ages": _percentage,
            "BelugaID": _percentage,
            "Birds525": _percentage,
            "FishNet": _percentage,
            "ImageNet-1K": _percentage,
            "KABR": _percentage,
            "iWildCam": _percentage,
            "iNat21": _percentage,
            "Plankton": _percentage,
            "NeWT": _percentage,
            "LeopardID": _percentage,
            "Pl@ntNet": _percentage,
        },
    )
    return (always_included,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
