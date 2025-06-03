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
import whenever

from biobench import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("report.py")


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
def date_to_ms(year: int, month: int, day: int) -> int:
    return whenever.Instant.from_utc(year, month, day).timestamp_millis()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Model:
    ckpt: str
    display: str
    family: str
    resolution: int
    params: int
    release_ms: int | None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


models = [
    Model(
        "ViT-B-32/openai",
        "CLIP ViT-B/32",
        "CLIP",
        224,
        87_849_216,
        date_to_ms(2021, 1, 5),
    ),
    Model(
        "ViT-B-16/openai",
        "CLIP ViT-B/16",
        "CLIP",
        224,
        86_192_640,
        date_to_ms(2021, 1, 5),
    ),
    Model(
        "ViT-L-14/openai",
        "CLIP ViT-L/14",
        "CLIP",
        224,
        303_966_208,
        date_to_ms(2021, 1, 5),
    ),
    Model(
        "ViT-L-14-336/openai",
        "CLIP ViT-L/14",
        "CLIP",
        336,
        304_293_888,
        date_to_ms(2021, 1, 5),
    ),
    Model(
        "ViT-B-16-SigLIP/webli",
        "SigLIP ViT-B/16",
        "SigLIP",
        224,
        92_884_224,
        date_to_ms(2023, 9, 27),
    ),
    Model(
        "ViT-B-16-SigLIP-256/webli",
        "SigLIP ViT-B/16",
        "SigLIP",
        256,
        92_930_304,
        date_to_ms(2023, 9, 27),
    ),
    Model(
        "ViT-B-16-SigLIP-384/webli",
        "SigLIP ViT-B/16",
        "SigLIP",
        384,
        93_176_064,
        date_to_ms(2023, 9, 27),
    ),
    Model(
        "ViT-B-16-SigLIP-512/webli",
        "SigLIP ViT-B/16",
        "SigLIP",
        512,
        93_520_128,
        date_to_ms(2023, 9, 27),
    ),
    Model(
        "ViT-L-16-SigLIP-256/webli",
        "SigLIP ViT-L/16",
        "SigLIP",
        256,
        315_956_224,
        date_to_ms(2023, 9, 27),
    ),
    Model(
        "ViT-L-16-SigLIP-384/webli",
        "SigLIP ViT-L/16",
        "SigLIP",
        384,
        316_283_904,
        date_to_ms(2023, 9, 27),
    ),
    Model(
        "ViT-SO400M-14-SigLIP/webli",
        "SigLIP SO400M/14",
        "SigLIP",
        224,
        427_680_704,
        date_to_ms(2023, 9, 27),
    ),
    Model(
        "ViT-SO400M-14-SigLIP-384/webli",
        "SigLIP SO400M/14",
        "SigLIP",
        384,
        428_225_600,
        date_to_ms(2023, 9, 27),
    ),
    Model(
        "ViT-B-32-SigLIP2-256/webli",
        "SigLIP2 ViT-B/32",
        "SigLIP2",
        256,
        94_552_320,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-B-16-SigLIP2/webli",
        "SigLIP2 ViT-B/16",
        "SigLIP2",
        224,
        92_884_224,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-B-16-SigLIP2-256/webli",
        "SigLIP2 ViT-B/16",
        "SigLIP2",
        256,
        92_930_304,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-B-16-SigLIP2-384/webli",
        "SigLIP2 ViT-B/16",
        "SigLIP2",
        384,
        93_176_064,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-B-16-SigLIP2-512/webli",
        "SigLIP2 ViT-B/16",
        "SigLIP2",
        512,
        93_520_128,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-L-16-SigLIP2-256/webli",
        "SigLIP2 ViT-L/16",
        "SigLIP2",
        256,
        315_956_224,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-L-16-SigLIP2-384/webli",
        "SigLIP2 ViT-L/16",
        "SigLIP2",
        384,
        316_283_904,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-L-16-SigLIP2-512/webli",
        "SigLIP2 ViT-L/16",
        "SigLIP2",
        512,
        316_742_656,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-SO400M-16-SigLIP2-256/webli",
        "SigLIP2 SO400M/16",
        "SigLIP2",
        256,
        427_888_064,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-SO400M-16-SigLIP2-384/webli",
        "SigLIP2 SO400M/16",
        "SigLIP2",
        384,
        428_256_704,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-SO400M-16-SigLIP2-512/webli",
        "SigLIP2 SO400M/16",
        "SigLIP2",
        512,
        428_772_800,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-gopt-16-SigLIP2-256/webli",
        "SigLIP2 ViT-1B/16",
        "SigLIP2",
        256,
        1_163_168_256,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "ViT-gopt-16-SigLIP2-384/webli",
        "SigLIP2 ViT-1B/16",
        "SigLIP2",
        384,
        1_163_659_776,
        date_to_ms(2025, 2, 20),
    ),
    Model(
        "hf-hub:UCSC-VLAA/openvision-vit-tiny-patch16-224",
        "OpenVision ViT-T/16",
        "OpenVision",
        224,
        5_561_088,
        date_to_ms(2025, 5, 7),
    ),
    Model(
        "hf-hub:UCSC-VLAA/openvision-vit-small-patch16-224",
        "OpenVision ViT-S/16",
        "OpenVision",
        224,
        21_812_736,
        date_to_ms(2025, 5, 7),
    ),
    Model(
        "hf-hub:UCSC-VLAA/openvision-vit-base-patch16-224",
        "OpenVision ViT-B/16",
        "OpenVision",
        224,
        86_191_104,
        date_to_ms(2025, 5, 7),
    ),
    Model(
        "hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224",
        "OpenVision ViT-L/14",
        "OpenVision",
        224,
        303_964_160,
        date_to_ms(2025, 5, 7),
    ),
    Model(
        "hf-hub:UCSC-VLAA/openvision-vit-so400m-patch14-224",
        "OpenVision SO400M/14",
        "OpenVision",
        224,
        413_770_608,
        date_to_ms(2025, 5, 7),
    ),
    Model(
        "hf-hub:UCSC-VLAA/openvision-vit-so400m-patch14-384",
        "OpenVision SO400M/14",
        "OpenVision",
        384,
        414_315_504,
        date_to_ms(2025, 5, 7),
    ),
    Model(
        "hf-hub:UCSC-VLAA/openvision-vit-huge-patch14-224",
        "OpenVision ViT-H/14",
        "OpenVision",
        224,
        632_074_240,
        date_to_ms(2025, 5, 7),
    ),
    Model(
        "dinov2_vits14_reg",
        "DINOv2 ViT-S/14",
        "DINOv2",
        224,
        22_058_112,
        date_to_ms(2024, 4, 12),
    ),
    Model(
        "dinov2_vitb14_reg",
        "DINOv2 ViT-B/14",
        "DINOv2",
        224,
        86_583_552,
        date_to_ms(2024, 4, 12),
    ),
    Model(
        "dinov2_vitl14_reg",
        "DINOv2 ViT-L/14",
        "DINOv2",
        224,
        304_372_736,
        date_to_ms(2024, 4, 12),
    ),
    Model(
        "dinov2_vitg14_reg",
        "DINOv2 ViT-g/14",
        "DINOv2",
        224,
        1_136_486_912,
        date_to_ms(2024, 4, 12),
    ),
    Model(
        "apple/aimv2-large-patch14-224",
        "AIMv2 ViT-L/14",
        "AIMv2",
        224,
        309_197_824,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-large-patch14-336",
        "AIMv2 ViT-L/14",
        "AIMv2",
        336,
        309_525_504,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-large-patch14-448",
        "AIMv2 ViT-L/14",
        "AIMv2",
        448,
        309_984_256,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-huge-patch14-224",
        "AIMv2 ViT-H/14",
        "AIMv2",
        224,
        680_851_968,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-huge-patch14-336",
        "AIMv2 ViT-H/14",
        "AIMv2",
        336,
        681_343_488,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-huge-patch14-448",
        "AIMv2 ViT-H/14",
        "AIMv2",
        448,
        682_031_616,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-1B-patch14-224",
        "AIMv2 ViT-1B/14",
        "AIMv2",
        224,
        1_234_958_336,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-1B-patch14-336",
        "AIMv2 ViT-1B/14",
        "AIMv2",
        336,
        1_235_613_696,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-1B-patch14-448",
        "AIMv2 ViT-1B/14",
        "AIMv2",
        448,
        1_236_531_200,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-3B-patch14-224",
        "AIMv2 ViT-3B/14",
        "AIMv2",
        224,
        2_720_658_432,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-3B-patch14-336",
        "AIMv2 ViT-3B/14",
        "AIMv2",
        336,
        2_721_641_472,
        date_to_ms(2024, 11, 21),
    ),
    Model(
        "apple/aimv2-3B-patch14-448",
        "AIMv2 ViT-3B/14",
        "AIMv2",
        448,
        2_723_017_728,
        date_to_ms(2024, 11, 21),
    ),
    Model("efficientnet_b0.ra_in1k", "EfficientNet B0", "CNN", 224, 5_288_548, None),
    Model("efficientnet_b3.ra2_in1k", "EfficientNet B3", "CNN", 320, 12_233_232, None),
    Model("mobilenetv2_100.ra_in1k", "MobileNet V2", "CNN", 224, 3_504_872, None),
    Model(
        "mobilenetv3_large_100.ra_in1k", "MobileNet V3 L", "CNN", 224, 5_483_032, None
    ),
    Model("resnet18.a1_in1k", "ResNet-18", "CNN", 288, 11_689_512, None),
    Model("resnet18d.ra2_in1k", "ResNet-18", "CNN", 288, 25_576_264, None),
    Model("resnet50.a1_in1k", "ResNet-50", "CNN", 288, 25_557_032, None),
    Model("resnet50d.a1_in1k", "ResNet-50d", "CNN", 288, 25_576_264, None),
    Model("convnext_tiny.in12k", "ConvNext-T", "CNN", 224, 36_910_477, None),
    Model(
        "convnext_tiny.in12k_ft_in1k", "ConvNext-T (IN1K)", "CNN", 288, 28_589_128, None
    ),
    Model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "ViT-B/16 (IN1K)",
        "ViT",
        224,
        86_567_656,
        None,
    ),
    Model(
        "hf-hub:imageomics/bioclip",
        "BioCLIP ViT-B/16",
        "cv4ecology",
        224,
        86_192_640,
        date_to_ms(2024, 5, 14),
    ),
    Model(
        "hf-hub:imageomics/bioclip-2",
        "BioCLIP-2 ViT-L/14",
        "cv4ecology",
        224,
        303_966_208,
        date_to_ms(2025, 5, 29),
    ),
    Model(
        "hf-hub:BGLab/BioTrove-CLIP",
        "BioTrove ViT-B/16",
        "cv4ecology",
        224,
        86_192_640,
        date_to_ms(2025, 1, 27),
    ),
    Model(
        "hf-hub:BVRA/MegaDescriptor-L-384",
        "MegaDescriptor Swin-L/4",
        "cv4ecology",
        384,
        195_198_516,
        date_to_ms(2023, 12, 14),
    ),
    Model(
        "vitl16", "V-JEPA ViT-L/16", "V-JEPA", 224, 305_490_944, date_to_ms(2024, 2, 15)
    ),
    Model(
        "vith16", "V-JEPA ViT-H/16", "V-JEPA", 224, 633_655_040, date_to_ms(2024, 2, 15)
    ),
    Model(
        "sam2_hiera_tiny.fb_r896_2pt1",
        "SAM2 Hiera-T",
        "SAM2",
        896,
        26_851_008,
        date_to_ms(2024, 10, 28),
    ),
    Model(
        "sam2_hiera_small.fb_r896_2pt1",
        "SAM2 Hiera-S",
        "SAM2",
        896,
        33_948_864,
        date_to_ms(2024, 10, 28),
    ),
    Model(
        "sam2_hiera_base_plus.fb_r896_2pt1",
        "SAM2 Hiera-B+",
        "SAM2",
        896,
        68_677_504,
        date_to_ms(2024, 10, 28),
    ),
    Model(
        "sam2_hiera_large.fb_r1024_2pt1",
        "SAM2 Hiera-L",
        "SAM2",
        1024,
        212_151_600,
        date_to_ms(2024, 10, 28),
    ),
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
            logger.warning("No `bootstrap_scores` for %s", task.name)
            continue

        logger.info("Getting score for %s", task.name)

        sub = df.filter(
            (pl.col("task_name") == task.name)
            & (pl.col("model_ckpt").is_in(model_lookup))
        )

        if sub.height == 0:
            continue

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
            low = (0.5 - (1 - alpha) / 2) * 100
            high = (0.5 + (1 - alpha) / 2) * 100
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

    This function reads experiment results from a SQLite database, calculates bootstrap statistics for each task/model combination, and writes the results to a JSON file for visualization.

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
    logger.info("Loaded %d predictions.", df.height)

    # Print any unknown checkpoints
    unknown_ckpts = set(df["model_ckpt"].unique()) - set(model_lookup.keys())
    if unknown_ckpts:
        logger.warning(
            "Found %d unknown checkpoints: %s",
            len(unknown_ckpts),
            sorted(unknown_ckpts),
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
        json.dump(data, fd, indent=4)


if __name__ == "__main__":
    tyro.cli(main)
