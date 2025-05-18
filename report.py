import collections.abc
import dataclasses
import functools
import importlib
import json
import logging
import os.path
import pathlib
import sqlite3
import subprocess
import time

import beartype
import numpy as np
import polars as pl
import statsmodels.stats.multitest
import tyro

from biobench import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("make_json.py")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    display: str

    @functools.cached_property
    def bootstrap_scores_fn(self) -> collections.abc.Callable:
        mod = importlib.import_module(f"biobench.{self.name}")

        return mod.bootstrap_scores

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


prior_work_tasks = [
    Task("imagenet1k", "ImageNet-1K"),
    Task("newt", "NeWT"),
]

benchmark_tasks = [
    Task("beluga", "Beluga"),
    Task("fishnet", "FishNet"),
    Task("fungiclef", "FungiCLEF"),
    Task("herbarium19", "Herbarium19"),
    Task("iwildcam", "iWildCam"),
    Task("kabr", "KABR"),
    Task("mammalnet", "MammalNet"),
    Task("plankton", "Plankton"),
    Task("plantnet", "Pl@ntNet"),
]

task_lookup = {task.name: task for task in prior_work_tasks + benchmark_tasks}


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Model:
    ckpt: str
    display: str
    family: str

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


models = [
    Model("ViT-B-32/openai", "CLIP ViT-B/32", "CLIP"),
    Model("ViT-B-16/openai", "CLIP ViT-B/16", "CLIP"),
    Model("ViT-L-14/openai", "CLIP ViT-L/14", "CLIP"),
    Model("ViT-L-14-336/openai", "CLIP ViT-L/14 (336px)", "CLIP"),
    Model("ViT-B-16-SigLIP/webli", "SigLIP ViT-B/16", "SigLIP"),
    Model("ViT-B-16-SigLIP-256/webli", "SigLIP ViT-B/16 (256px)", "SigLIP"),
    Model("ViT-B-16-SigLIP-384/webli", "SigLIP ViT-B/16 (384px)", "SigLIP"),
    Model("ViT-B-16-SigLIP-512/webli", "SigLIP ViT-B/16 (512px)", "SigLIP"),
    Model("ViT-L-16-SigLIP-256/webli", "SigLIP ViT-L/16 (256px)", "SigLIP"),
    Model("ViT-L-16-SigLIP-384/webli", "SigLIP ViT-L/16 (384px)", "SigLIP"),
    Model("ViT-SO400M-14-SigLIP/webli", "SigLIP SO400M/14", "SigLIP"),
    Model("ViT-SO400M-14-SigLIP-384/webli", "SigLIP SO400M/14 (384px)", "SigLIP"),
    Model("dinov2_vits14_reg", "DINOv2 ViT-S/14", "DINOv2"),
    Model("dinov2_vitb14_reg", "DINOv2 ViT-B/14", "DINOv2"),
    Model("dinov2_vitl14_reg", "DINOv2 ViT-L/14", "DINOv2"),
    Model("dinov2_vitg14_reg", "DINOv2 ViT-g/14", "DINOv2"),
    Model("apple/aimv2-large-patch14-224", "AIMv2 ViT-L/14 (224px)", "AIMv2"),
    Model("apple/aimv2-large-patch14-336", "AIMv2 ViT-L/14 (336px)", "AIMv2"),
    Model("apple/aimv2-large-patch14-448", "AIMv2 ViT-L/14 (448px)", "AIMv2"),
    Model("apple/aimv2-huge-patch14-224", "AIMv2 ViT-H/14 (224px)", "AIMv2"),
    Model("apple/aimv2-huge-patch14-336", "AIMv2 ViT-H/14 (336px)", "AIMv2"),
    Model("apple/aimv2-huge-patch14-448", "AIMv2 ViT-H/14 (448px)", "AIMv2"),
    Model("apple/aimv2-1B-patch14-224", "AIMv2 ViT-1B/14 (224px)", "AIMv2"),
    Model("apple/aimv2-1B-patch14-336", "AIMv2 ViT-1B/14 (336px)", "AIMv2"),
    Model("apple/aimv2-1B-patch14-448", "AIMv2 ViT-1B/14 (448px)", "AIMv2"),
    Model("apple/aimv2-3B-patch14-224", "AIMv2 ViT-3B/14 (224px)", "AIMv2"),
    Model("apple/aimv2-3B-patch14-336", "AIMv2 ViT-3B/14 (336px)", "AIMv2"),
    Model("apple/aimv2-3B-patch14-448", "AIMv2 ViT-3B/14 (448px)", "AIMv2"),
    Model("efficientnet_b0.ra_in1k", "EfficientNet B0", "CNN"),
    Model("efficientnet_b3.ra2_in1k", "EfficientNet B3", "CNN"),
    Model("mobilenetv2_100.ra_in1k", "MobileNet V2", "CNN"),
    Model("mobilenetv3_large_100.ra_in1k", "MobileNet V3 L", "CNN"),
    Model("resnet18.a1_in1k", "ResNet-18", "CNN"),
    Model("resnet18d.ra2_in1k", "ResNet-18", "CNN"),
    Model("resnet50.a1_in1k", "ResNet-50", "CNN"),
    Model("resnet50d.a1_in1k", "ResNet-50d", "CNN"),
    Model("convnext_tiny.in12k", "ConvNext-T", "CNN"),
    Model("convnext_tiny.in12k_ft_in1k", "ConvNext-T (IN1K)", "CNN"),
    Model("vit_base_patch16_224.augreg2_in21k_ft_in1k", "ViT-B/16 (IN1K)", "ViT"),
    Model("hf-hub:imageomics/bioclip", "BioCLIP ViT-B/16", "cv4ecology"),
    Model("hf-hub:BGLab/BioTrove-CLIP", "BioTrove ViT-B/16", "cv4ecology"),
    Model(
        "hf-hub:BVRA/MegaDescriptor-L-384",
        "MegaDescriptor Swin-L/4 (384px)",
        "cv4ecology",
    ),
    Model("vitl16", "V-JEPA ViT-L/16", "V-JEPA"),
    Model("vith16", "V-JEPA ViT-H/16", "V-JEPA"),
    Model("sam2_hiera_tiny.fb_r896_2pt1", "SAM2 Hiera-T", "SAM2"),
    Model("sam2_hiera_small.fb_r896_2pt1", "SAM2 Hiera-S", "SAM2"),
    Model("sam2_hiera_base_plus.fb_r896_2pt1", "SAM2 Hiera-B+", "SAM2"),
    Model("sam2_hiera_large.fb_r1024_2pt1", "SAM2 Hiera-L", "SAM2"),
]


model_lookup = {model.ckpt: model for model in models}


@beartype.beartype
def get_git_hash() -> str:
    """Returns the hash of the current git commit.

    Returns:
        str: The hash of the current git commit, assuming we are in a git repo
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@beartype.beartype
def calc_scores(
    df: pl.DataFrame, *, n_bootstraps: int, alpha: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed=seed)
    scores_rows, bests_rows = [], []
    for task in helpers.progress(
        prior_work_tasks + benchmark_tasks, every=1, desc="bootstraps"
    ):
        try:
            bootstrap_scores = task.bootstrap_scores_fn
        except AttributeError:
            print(f"No `bootstrap_scores` for {task.name}")
            continue

        sub = df.filter(
            (pl.col("task_name") == task.name)
            & (pl.col("model_ckpt").is_in(model_lookup))
        )

        scores = bootstrap_scores(sub, b=n_bootstraps, rng=rng)
        # freeze model order once
        ckpts = sorted(scores)  # list[str]  length = m

        # stack into a (b, m) matrix that matches that order
        s = np.column_stack([scores[c] for c in ckpts])  # (b, m)

        # locate the empirical best (column index j*)
        best_j = s.mean(axis=0).argmax()
        best_c = ckpts[best_j]

        # one-sided p-values against the best
        p_raw = (s >= s[:, [best_j]]).mean(axis=0)  # vector length m
        rej, p_adj, *_ = statsmodels.stats.multitest.multipletests(
            p_raw, alpha=alpha, method="holm"
        )

        # list checkpoints that are *not rejected* -> bold
        ties = [c for c, r in zip(ckpts, rej) if not r]
        bests_rows.append({"task": task.name, "best": best_c, "ties": ties})

        # Calculate reference scores, then compute task-level results.
        for model_ckpt, ref_score in bootstrap_scores(sub).items():
            bootstrap_mean = scores[model_ckpt].mean()
            low = 0.5 - (1 - alpha) / 2 * 100
            high = 0.5 + (1 - alpha) / 2 * 100
            ci_low, ci_high = np.percentile(scores[model_ckpt], (low, high))

            scores_rows.append({
                "task": task.name,
                "model": model_ckpt,
                "mean": ref_score.item(),
                "bootstrap_mean": bootstrap_mean.item(),
                "ci_low": ci_low.item(),
                "ci_high": ci_high.item(),
            })

    return pl.DataFrame(scores_rows), pl.DataFrame(bests_rows)


@beartype.beartype
def main(
    db: pathlib.Path = pathlib.Path(
        os.path.expandvars("/local/scratch/$USER/experiments/biobench/reports.sqlite")
    ),
    out: pathlib.Path = pathlib.Path("docs/data/results.json"),
    seed: int = 17,
    alpha=0.05,
    n_bootstraps: int = 500,
):
    """Generate a JSON report of benchmark results with bootstrap confidence intervals.
    
    This function reads experiment results from a SQLite database, calculates bootstrap
    statistics for each task/model combination, and writes the results to a JSON file
    for visualization in the documentation.
    
    Args:
        db: Path to the SQLite database containing experiment results.
        out: Path where the JSON report will be written.
        seed: Random seed for reproducible bootstrapping.
        alpha: Significance level for confidence intervals and hypothesis tests.
        n_bootstraps: Number of bootstrap samples to generate.
    """
    stmt = "SELECT experiments.task_name, experiments.model_ckpt, predictions.score, predictions.img_id, predictions.info FROM experiments JOIN predictions ON experiments.id = predictions.experiment_id WHERE n_train = -1"
    df = (
        pl.read_database(stmt, sqlite3.connect(db), infer_schema_length=100_000)
        .lazy()
        .filter(pl.col("task_name").is_in(task_lookup))
        .with_columns(
            pl.coalesce([
                pl.col("info").str.json_path_match("$.y_true"),
                pl.col("info").str.json_path_match("$.true_y"),
            ]).alias("y_true"),
            pl.coalesce([
                pl.col("info").str.json_path_match("$.y_pred"),
                pl.col("info").str.json_path_match("$.pred_y"),
            ]).alias("y_pred"),
        )
        .select("task_name", "model_ckpt", "img_id", "score", "y_true", "y_pred")
        .collect()
    )

    scores_df, bests_df = calc_scores(
        df, n_bootstraps=n_bootstraps, alpha=alpha, seed=seed
    )

    data = {
        "meta": {
            "schema": 1,
            "generated": int(time.time() * 1000),
            "git_commit": get_git_hash(),
            "seed": seed,
            "alpha": alpha,
            "n_bootstraps": n_bootstraps,
        },
        "models": [model.to_dict() for model in models],
        "benchmark_tasks": [task.to_dict() for task in benchmark_tasks],
        "prior_work_tasks": [task.to_dict() for task in prior_work_tasks],
        "results": scores_df.to_dicts(),
        "bests": bests_df.to_dicts(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fd:
        json.dump(data, fd)


if __name__ == "__main__":
    tyro.cli(main)
