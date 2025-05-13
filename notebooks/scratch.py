import marimo

__generated_with = "0.11.25"
app = marimo.App(width="full")


@app.cell
def _():
    import os.path
    import sys

    PATH_TO_BIOBENCH = os.path.expandvars("$HOME/projects/biobench")

    if PATH_TO_BIOBENCH not in sys.path:
        sys.path.insert(0, PATH_TO_BIOBENCH)

    import collections.abc
    import dataclasses
    import functools
    import importlib
    import json
    import math
    import sqlite3

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches
    import numpy as np
    import polars as pl
    import scipy.stats
    import sklearn.linear_model
    import sklearn.metrics
    from jaxtyping import Real, jaxtyped

    from biobench import reporting, helpers
    return (
        PATH_TO_BIOBENCH,
        Real,
        beartype,
        collections,
        dataclasses,
        functools,
        helpers,
        importlib,
        jaxtyped,
        json,
        math,
        matplotlib,
        mo,
        np,
        os,
        pl,
        plt,
        reporting,
        scipy,
        sklearn,
        sqlite3,
        sys,
    )


@app.cell
def _(beartype, collections, dataclasses, functools, importlib):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Task:
        name: str
        display: str
        _: dataclasses.KW_ONLY
        legend: str | None = None

        @functools.cached_property
        def score_fn(self) -> collections.abc.Callable:
            return importlib.import_module(f"biobench.{self.name}").score

        @functools.cached_property
        def score_fn_batch(self) -> collections.abc.Callable:
            return importlib.import_module(f"biobench.{self.name}").score_batch


    prior_work_tasks = [
        Task("imagenet1k", "ImageNet-1K", legend="lower left"),
        # Task("inat21", "iNat2021"),
        Task("newt", "NeWT"),
    ]

    benchmark_tasks = [
        Task("beluga", "Beluga"),
        Task("fishnet", "FishNet"),
        Task("fungiclef", "FungiCLEF", legend="upper right"),
        Task("herbarium19", "Herbarium19"),
        Task("iwildcam", "iWildCam"),
        Task("kabr", "KABR", legend="upper right"),
        Task("mammalnet", "MammalNet"),
        Task("plankton", "Plankton"),
        Task("plantnet", "Pl@ntNet", legend="upper right"),
    ]

    task_lookup = {task.name: task for task in prior_work_tasks + benchmark_tasks}
    return Task, benchmark_tasks, prior_work_tasks, task_lookup


@app.cell
def _(mo, os, pl, sqlite3, task_lookup):
    df = (
        pl.read_database(
            f"SELECT experiments.task_name, experiments.model_ckpt, predictions.score, predictions.img_id, predictions.info FROM experiments JOIN predictions ON experiments.id = predictions.experiment_id WHERE n_train = -1",
            sqlite3.connect(
                os.path.expandvars(
                    "/local/scratch/$USER/experiments/biobench/reports.sqlite"
                )
            ),
            infer_schema_length=100_000,
        )
        .lazy()
        .filter(pl.col("task_name").is_in(task_lookup))
        .with_columns(
            pl.coalesce(
                [
                    pl.col("info").str.json_path_match("$.y_true"),
                    pl.col("info").str.json_path_match("$.true_y"),
                ]
            ).alias("y_true"),
            pl.coalesce(
                [
                    pl.col("info").str.json_path_match("$.y_pred"),
                    pl.col("info").str.json_path_match("$.pred_y"),
                ]
            ).alias("y_pred"),
        )
        .select("task_name", "model_ckpt", "img_id", "score", "y_true", "y_pred")
        .collect()
    )
    mo.md(f"{len(df):,}")
    return (df,)


@app.cell
def _(df, json, pl, reporting, task_lookup):
    raise ValueError()


    def _struct_to_pred(st: dict) -> reporting.Prediction:
        return reporting.Prediction(
            id=st["img_id"], score=st["score"], info=json.loads(st["info"])
        )


    def _metric_udf(exprs):
        structs, task_names = exprs
        preds = [_struct_to_pred(st) for st in structs]
        return task_lookup[task_names[0]].score_fn(preds)


    scores = df.group_by("model_ckpt", "task_name").agg(
        score=pl.map_groups(
            [pl.struct("img_id", "score", "info"), "task_name"],
            _metric_udf,
            return_dtype=pl.Float64,
        )
    )
    scores
    return (scores,)


@app.cell
def _(beartype, dataclasses, df, reporting):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Model:
        ckpt: str
        display: str
        family: str


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
        # Model(
        #     "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        #     "ViT-B/16 (IN1K)",
        #     reporting.ORANGE_RGB01,
        # ),
        Model("hf-hub:imageomics/bioclip", "BioCLIP ViT-B/16", "cv4ecology"),
        Model("hf-hub:BGLab/BioTrove-CLIP", "BioTrove ViT-B/16", "cv4ecology"),
        # Model(
        #     "local:ViT-L-14/./models/vit-l-14-tol50m-laion2b-replay-ep50.pt",
        #     "BioCLIP-2 ViT-L/14 (ToL-50M)",
        #     "cv4ecology",
        # ),
        # Model(
        #     "local:ViT-L-14/./models/vit-l-14-tol200m-laion2b-replay-ep30.pt",
        #     "BioCLIP-2 ViT-L/14 (ToL-200M)",
        #     "cv4ecology",
        # ),
        Model(
            "hf-hub:BVRA/MegaDescriptor-L-384",
            "MegaDescriptor Swin-L/4 (384px)",
            "cv4ecology",
        ),
        # Model("vitl16", "V-JEPA ViT-L/16", "V-JEPA"),
        # Model("vith16", "V-JEPA ViT-H/16", "V-JEPA"),
        # Model("vith16-384", "V-JEPA ViT-H/16 (384px)", "V-JEPA"),
        Model("sam2_hiera_tiny.fb_r896_2pt1", "SAM2 Hiera-T", "SAM2"),
        Model("sam2_hiera_small.fb_r896_2pt1", "SAM2 Hiera-S", "SAM2"),
        Model("sam2_hiera_base_plus.fb_r896_2pt1", "SAM2 Hiera-B+", "SAM2"),
        Model("sam2_hiera_large.fb_r1024_2pt1", "SAM2 Hiera-L", "SAM2"),
    ]

    print(len(models))

    model_lookup = {model.ckpt: model for model in models}

    color_lookup = {
        "CLIP": reporting.BLUE_RGB01,
        "SigLIP": reporting.CYAN_RGB01,
        "DINOv2": reporting.SEA_RGB01,
        "AIMv2": reporting.GOLD_RGB01,
        "CNN": reporting.CREAM_RGB01,
        "cv4ecology": reporting.RED_RGB01,
        "V-JEPA": reporting.RUST_RGB01,
        "SAM2": reporting.BLACK_RGB01,
    }

    [
        ckpt
        for ckpt in sorted(df.get_column("model_ckpt").unique().to_list())
        if ckpt not in model_lookup
    ]
    return Model, color_lookup, model_lookup, models


@app.cell
def _(
    benchmark_tasks,
    color_lookup,
    math,
    matplotlib,
    model_lookup,
    models,
    os,
    pl,
    plt,
    scores,
):
    def make_overview_fig():
        all_tasks = benchmark_tasks
        n_rows = math.ceil(len(all_tasks) / 3)
        n_cols = 3

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            sharey=False,
            squeeze=False,
            dpi=300,
            figsize=(3 * n_cols, 3 * n_rows),
        )
        axes = axes.reshape(-1)
        # Turn off extra axes
        for ax in axes[len(all_tasks) :]:
            ax.axis("off")

        ckpts = [model.ckpt for model in models]

        for task, ax in zip(all_tasks, axes):
            xs = [model_lookup[ckpt].display for ckpt in ckpts]
            colors = [color_lookup[model_lookup[ckpt].family] for ckpt in ckpts]

            ys = []
            for ckpt in ckpts:
                try:
                    score = scores.filter(
                        (pl.col("model_ckpt") == ckpt) & (pl.col("task_name") == task.name)
                    ).item(row=0, column="score")
                except IndexError:
                    print(f"Missing score on {task.name} for {ckpt}.")
                    score = 0

                ys.append(score)

            ax.bar(xs, ys, color=colors)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_title(task.display)

            ax.spines[["right", "top"]].set_visible(False)
            ax.yaxis.grid(True, linestyle="-", linewidth=0.2, color="gray", alpha=0.3)

            if task.legend:
                handles = [
                    matplotlib.patches.Patch(color=color, label=family)
                    for family, color in color_lookup.items()
                ]
                ax.legend(
                    handles=handles,
                    ncols=2,
                    fontsize="small",
                    labelspacing=0.2,
                    handlelength=0.7,
                    handleheight=0.7,
                    handletextpad=0.4,
                    loc=task.legend,
                    framealpha=0.9,
                )

        fig.savefig(os.path.join("results", "all-tasks.pdf"))
        return fig


    make_overview_fig()
    return (make_overview_fig,)


@app.cell
def _(
    Real,
    beartype,
    jaxtyped,
    models,
    np,
    pl,
    prior_work_tasks,
    scipy,
    scores,
):
    @jaxtyped(typechecker=beartype.beartype)
    def pearson_r(x: Real[np.ndarray, " n"], y: Real[np.ndarray, " n"]) -> float:
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
        return 0.0 if denom == 0 else float((x * y).sum() / denom)


    @jaxtyped(typechecker=beartype.beartype)
    def spearman_r(x: Real[np.ndarray, " n"], y: Real[np.ndarray, " n"]) -> float:
        def rank(a):
            tmp = np.argsort(a)
            ranks = np.empty_like(tmp)
            ranks[tmp] = np.arange(len(a))
            return ranks

        return pearson_r(rank(x), rank(y))


    @jaxtyped(typechecker=beartype.beartype)
    def kendall_tau(x: Real[np.ndarray, " n"], y: Real[np.ndarray, " n"]) -> float:
        """Kendall tau (tau-b) for one-dimensional arrays x, y of equal length."""
        return scipy.stats.kendalltau(x, y).statistic.item()


    @beartype.beartype
    def make_corr_df():
        ckpts = [model.ckpt for model in models]
        ignore = {task.name for task in prior_work_tasks}
        task_names = set(scores.get_column("task_name").to_list()) - ignore

        def get_score(ckpt: str, task_name: str) -> float | None:
            try:
                return (
                    scores.filter(
                        (pl.col("model_ckpt") == ckpt) & (pl.col("task_name") == task_name)
                    )
                    .group_by("model_ckpt", "task_name")
                    .mean()
                    .item(row=0, column="score")
                )
            except IndexError:
                print(f"Missing score on {task_name} for {ckpt}.")
                return None

        rows = []

        for ckpt in ckpts:
            row = {"ckpt": ckpt}

            row["imagenet1k"] = get_score(ckpt, "imagenet1k")
            if row["imagenet1k"] is None:
                continue

            row["newt"] = get_score(ckpt, "newt")
            if row["newt"] is None:
                continue

            # row["inat21"] = get_score(ckpt, "inat21")
            # if row["inat21"] is None:
            #     row["inat21"] = 0.0

            skip_row = False
            for task_name in task_names:
                row[task_name] = get_score(ckpt, task_name)
                if row[task_name] is None:
                    skip_row = True

            if skip_row:
                continue

            rows.append(row)

        return pl.DataFrame(rows)
    return kendall_tau, make_corr_df, pearson_r, spearman_r


@app.cell
def _(benchmark_tasks, make_corr_df, os, plt, reporting):
    def make_correlation_fig(n_boot: int = 5_000):
        fig_df = make_corr_df()

        task_cols = [task.name for task in benchmark_tasks if task.name in fig_df.columns]
        xs = fig_df.get_column("imagenet1k").to_numpy() * 100
        ys = fig_df.select(task_cols).to_numpy().mean(axis=1)

        fig, ax = plt.subplots(dpi=300, sharey=True, figsize=(5, 3))

        ax.scatter(
            xs,
            ys,
            color=reporting.SEA_RGB01,
            label="Model Scores",
            marker="o",
            alpha=0.5,
        )

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"ImageNet-1K (%)")
        ax.set_ylabel(f"Mean BioBench Score")

        ax.spines[["right", "top"]].set_visible(False)
        fig.legend()
        fig.tight_layout(pad=0.0)
        fig.savefig(os.path.join("results", "imagenet1k-correlation.pdf"))

        return fig


    make_correlation_fig()
    return (make_correlation_fig,)


@app.cell
def _(benchmark_tasks, kendall_tau, make_corr_df, np, pl, spearman_r):
    def print_rank_stats(n_boot: int = 5_000):
        named_fns = [("tau", kendall_tau), ("rho", spearman_r)]

        corr_df = make_corr_df()
        task_cols = [task.name for task in benchmark_tasks if task.name in corr_df.columns]
        corr_df = corr_df.with_columns(mean=pl.mean_horizontal(task_cols))

        rng = np.random.default_rng(0)

        for min_x in (0, 0.70, 0.75):
            for name, fn in named_fns:
                print("-" * 28)
                print(f"'{name}' for all ckpts > {min_x * 100:.1f}%")
                print("-" * 28)

                xs = (
                    corr_df.filter(pl.col("imagenet1k") >= min_x)
                    .get_column("imagenet1k")
                    .to_numpy()
                    * 100
                )
                ys = (
                    corr_df.filter(pl.col("imagenet1k") >= min_x)
                    .get_column("mean")
                    .to_numpy()
                )

                point = fn(xs, ys)
                print(f"Observed: {point:.4f}")

                boot = []
                for _ in range(n_boot):
                    i = rng.choice(len(xs), len(xs), replace=True)
                    boot.append(fn(xs[i], ys[i]))

                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                print(f"95 % CI [{ci_low:.3f}, {ci_high:.3f}]")

                perm = [fn(rng.permutation(xs, axis=0), ys) for _ in range(n_boot)]
                p = (np.sum(np.array(perm) >= point) + 1) / (len(perm) + 1)
                print(f"Permutation p-value: {p}")

        for n_ckpts in (100, 10):
            task_scores = (
                corr_df.sort("mean", descending=True)
                .head(n_ckpts)
                .select(task_cols)
                .to_numpy()
            )
            xs = task_scores.mean(axis=1)
            for name, fn in named_fns:
                print("-" * 40)
                print(f"Internal '{name}' for top {n_ckpts} ckpts")
                print("-" * 40)

                point = np.mean([fn(xs, task) for task in task_scores.T])
                print(f"Observed: {point:.4f}")

                boot = []
                for _ in range(n_boot):
                    i = rng.choice(len(xs), len(xs), replace=True)
                    boot.append(np.mean([fn(xs[i], task[i]) for task in task_scores.T]))
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                print(f"95 % CI [{ci_low:.3f}, {ci_high:.3f}]")


    print_rank_stats()
    return (print_rank_stats,)


@app.cell
def _(
    beartype,
    kendall_tau,
    make_corr_df,
    np,
    os,
    pl,
    plt,
    reporting,
    task_lookup,
):
    @beartype.beartype
    def hook_fig_helper(
        fig_df,
        x_task: str,
        y_tasks: list[str],
        min_models: int = 10,
        n_bootstrap: int = 500,
    ):
        xs, ys, lo, hi = [], [], [], []
        for t in np.arange(0, 90, 4):
            sub = fig_df.filter(pl.col(x_task) >= t / 100)
            if sub.height < min_models:  # need a few points
                print(
                    f"Skipping {t} on {x_task} with only {sub.height} points (<{min_models})."
                )
                continue
            boots = []
            for _ in range(n_bootstrap):
                samp = sub.sample(sub.height, with_replacement=True)

                boots.append(
                    kendall_tau(
                        samp[x_task].to_numpy(),
                        np.mean(samp.select(y_tasks).to_numpy(), axis=1),
                    )
                )

            xs.append(t)
            ys.append(
                kendall_tau(
                    sub[x_task].to_numpy(),
                    np.mean(sub.select(y_tasks).to_numpy(), axis=1),
                )
            )
            lo.append(np.percentile(boots, 2.5))
            hi.append(np.percentile(boots, 97.5))
        return xs, ys, lo, hi


    @beartype.beartype
    def make_hook_fig():
        fig_df = make_corr_df()

        y_tasks = [
            task_lookup[task_name] for task_name in ("herbarium19", "iwildcam", "beluga")
        ]
        letters = "abcdefg"
        colors = [reporting.CYAN_RGB01, reporting.GOLD_RGB01, reporting.SCARLET_RGB01]

        fig, axes = plt.subplots(dpi=300, ncols=len(y_tasks), sharey=False, figsize=(10, 3))

        for ax, task, letter, color in zip(axes, y_tasks, letters, colors):
            xs, ys, lo, hi = hook_fig_helper(fig_df, "imagenet1k", y_tasks=[task.name])
            ax.plot(xs, ys, marker=".", color=color, linewidth=1, markersize=3)
            ax.fill_between(xs, lo, hi, alpha=0.2, color=color, linewidth=0)
            ax.set_xlabel("Min. ImageNet‑1K (%)")
            ax.set_ylabel(r"Kendall's $\tau$ (rank correlation)")
            ax.yaxis.grid(True, linestyle="-", linewidth=0.2, color="gray", alpha=0.3)
            # ax.xaxis.grid(True, linestyle="-", linewidth=0.2, color="gray", alpha=0.3)
            ax.set_ylim(-1, 1)
            gaps = [""] * 4
            ax.set_xticks(
                np.arange(0, 81, 4).tolist(),
                ["0", *gaps, "20", *gaps, "40", *gaps, "60", *gaps, "80"],
            )
            ax.set_yticks(
                [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0],
                ["-1.0", "", "-0.5", "", "0", "", "0.5", "", "1.0"],
            )
            ax.set_xlim(0, 85)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_title(f"({letter.lower()}) ImageNet-1K vs. {task.display}")
            print(f"Finished {task.name}")

        fig.tight_layout(pad=0.0)
        fig.savefig(os.path.join("results", "hook.pdf"))

        return fig


    make_hook_fig()
    return hook_fig_helper, make_hook_fig


@app.cell
def _(benchmark_tasks, models, pl, prior_work_tasks, scores):
    def make_overview_table():
        tasks = sorted(scores.get_column("task_name").unique().to_list())
        ckpts = [model.ckpt for model in models]

        for ckpt in ckpts:
            print(ckpt, end=" ")
            for task in prior_work_tasks + benchmark_tasks:
                try:
                    score = scores.filter(
                        (pl.col("model_ckpt") == ckpt) & (pl.col("task_name") == task.name)
                    ).item(row=0, column="score")
                    print(f"{score * 100:.1f}", end=" & ")
                except IndexError:
                    print("", end=" & ")
                    continue

            benchmark_scores = scores.filter(
                (pl.col("model_ckpt") == ckpt)
                & (pl.col("task_name").is_in([t.name for t in benchmark_tasks]))
            )
            if benchmark_scores.height < 9:
                print()
                continue
            mean = (
                benchmark_scores.group_by("model_ckpt").mean().item(row=0, column="score")
            )
            print(f"{mean * 100:.1f}")


    make_overview_table()
    return (make_overview_table,)


@app.cell
def _(benchmark_tasks, df, models, np, pl, prior_work_tasks):
    import statsmodels.stats.multitest


    def significance_testing():
        b, alpha = 1_000, 0.05
        rng = np.random.default_rng(seed=17)

        for task in prior_work_tasks + benchmark_tasks:
            try:
                score_batch = task.score_fn_batch
            except AttributeError:
                print(f"No `score_batch` for {task.name}")
                continue

            sub = df.filter(pl.col("task_name") == task.name)

            n, *rest = (
                sub.group_by("model_ckpt")
                .agg(pl.len().alias("n"))
                .get_column("n")
                .to_list()
            )

            if task.name == "imagenet1k":
                # For some reason one of our imagenet-1k things only has 49.3K predictions. So that's what we use. It's probably enough.
                n = 49364
            else:
                assert all(n == i for i in rest)

            idx_bs = rng.integers(0, n, size=(b, n), dtype=np.int32)

            scores = np.zeros((b, len(models)), dtype=np.float32)

            if task.name == "newt":
                scores_newt_buf = np.empty((b, n), dtype=np.int32)

                for i, model in enumerate(models):
                    scores_newt = (
                        sub.filter(pl.col("model_ckpt") == model.ckpt)
                        .select("img_id", "score")
                        .unique()
                        .sort("img_id")
                        .get_column("score")
                        .to_numpy()
                    )

                    if scores_newt.size == 0:
                        continue

                    np.take(scores_newt, idx_bs, axis=0, out=scores_newt_buf)
                    scores[:, i] = np.mean(scores_newt, axis=-1)
            else:
                y_pred_buf = np.empty((b, n), dtype=np.int32)
                y_true_buf = np.empty((b, n), dtype=np.int32)

                for i, model in enumerate(models):
                    # pull y_true and y_pred for *one* model
                    y_pred = (
                        sub.filter(pl.col("model_ckpt") == model.ckpt)
                        .select("img_id", "y_pred")
                        .unique()
                        .sort("img_id")
                        .get_column("y_pred")
                        .cast(pl.Float32)
                        .cast(pl.Int32)
                        .to_numpy()
                    )

                    y_true = (
                        sub.filter(pl.col("model_ckpt") == model.ckpt)
                        .select("img_id", "y_true")
                        .unique()
                        .sort("img_id")
                        .get_column("y_true")
                        .cast(pl.Float32)
                        .cast(pl.Int32)
                        .to_numpy()
                    )

                    if y_pred.size == 0:
                        continue

                    assert y_true.size == y_pred.size

                    # bootstrap resample into pre-allocated buffers
                    np.take(y_pred, idx_bs, axis=0, out=y_pred_buf)
                    np.take(y_true, idx_bs, axis=0, out=y_true_buf)

                    # ref = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
                    scores[:, i] = score_batch(y_true_buf, y_pred_buf)

            means = scores.mean(axis=0)

            best_m = means.argmax()

            # Null hypothesis: model m is no worse than best_m. If this is true, then the proportion of scores where scores[:, m] >= scores[:, best_m] is high. If the proportion is low, then model m is likely worse than best_m.
            pvals = (scores >= scores[:, best_m][:, None]).mean(axis=0)
            reject, pvals, _, _ = statsmodels.stats.multitest.multipletests(
                pvals, alpha=alpha, method="holm"
            )

            # Print bolding
            bold = [models[i].ckpt for i, r in enumerate(reject) if not r]
            print(f"{task.name}:", ", ".join(bold))


    significance_testing()
    return significance_testing, statsmodels


@app.cell
def _(df, pl):
    df.filter(pl.col("task_name") == "imagenet1k").group_by("model_ckpt").agg(
        pl.len().alias("n")
    ).get_column("n").unique()
    return


if __name__ == "__main__":
    app.run()
