import marimo

__generated_with = "0.8.10"
app = marimo.App(width="full")


@app.cell
def __():
    import json
    import os

    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return json, mo, np, os, pl, plt


@app.cell
def __(json, os, pl):
    def load_reports(root: str):
        schema = None
        reports = []
        if os.path.isdir(root):
            for filename in os.listdir(root):
                for line in open(os.path.join(root, filename)):
                    report = json.loads(line)
                    report.pop("examples")
                    report.pop("splits")

                    report = pl.json_normalize(report, separator="_")

                    schema = report.schema
                    reports.append(report.row(0, named=True))

        reports = pl.DataFrame(reports, schema=schema)
        return reports.with_columns(model=pl.col("run_args_model_ckpt").str.replace_many({
            "vit_base_patch14_reg4_dinov2.lvd142m": "ViT-B-14/DINOv2",
            "hf-hub:imageomics/bioclip": "ViT-B-16/BioCLIP",
            "ViT-B-16/laion400m_e32": "ViT-B-16/LAION-400M",
        }))

    reports = load_reports("./reports")
    reports.select("name", "model", "mean_score").sort(by=("name", "mean_score"), descending=(False, True))
    return load_reports, reports


@app.cell
def __():
    return


@app.cell
def __(mo, np, pl, plt):
    def plot_task(reports, task: str):
        fig, ax = plt.subplots()

        reports = reports.filter(pl.col("name") == task)

        xs = reports.get_column("model").to_list()
        ys = reports.get_column("mean_score").to_list()

        yerr = np.array([ys, ys])
        yerr[0] = yerr[0] - reports.get_column("confidence_interval_lower").to_list()
        yerr[1] = reports.get_column("confidence_interval_upper").to_list() - yerr[1]

        ax.errorbar(xs, ys, yerr, fmt="o", linewidth=2, capsize=6)
        ax.set_title(f"Mean {task} Performance")
        ax.tick_params(axis='x', labelrotation=20)
        return mo.md(f"""
    Mean performance on {task}. Error bars indicate 95% confidence intervals, bootstrapped from the test set.
    {mo.as_html(fig)}""")
    return plot_task,


@app.cell
def __(plot_task, reports):
    plot_task(reports, "KABR")
    return


@app.cell
def __(plot_task, reports):
    plot_task(reports, "NeWT")
    return


@app.cell
def __(plot_task, reports):
    plot_task(reports, "")
    return


if __name__ == "__main__":
    app.run()
