"""
Entrypoint for running all tasks in `biobench`.

Most of this script is self documenting.
Run `python benchmark.py --help` to see all the options.

Note that you will have to download all the datasets, but each dataset includes its own download script with instructions.
For example, see `biobench.newt.download` for an example.

.. include:: ./examples.md

.. include:: ./design.md
"""

import collections
import csv
import dataclasses
import json
import logging
import os
import resource
import sqlite3
import time
import typing

import beartype
import numpy as np
import submitit
import tyro

from biobench import (
    ages,
    beluga,
    birds525,
    fishnet,
    imagenet,
    inat21,
    interfaces,
    iwildcam,
    kabr,
    leopard,
    newt,
    plankton,
    plantnet,
    rarespecies,
)

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("biobench")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Params to run one or more benchmarks in a parallel setting."""

    slurm: bool = False
    """whether to use submitit to run jobs on a slurm cluster."""
    slurm_acct: str = "PAS2136"
    """slurm account string."""

    models_cvml: typing.Annotated[
        list[interfaces.ModelArgsCvml], tyro.conf.arg(name="models")
    ] = dataclasses.field(
        default_factory=lambda: [
            interfaces.ModelArgsCvml("open-clip", "RN50/openai"),
            interfaces.ModelArgsCvml("open-clip", "ViT-B-16/openai"),
            interfaces.ModelArgsCvml("open-clip", "ViT-B-16/laion400m_e32"),
            interfaces.ModelArgsCvml("open-clip", "hf-hub:imageomics/bioclip"),
            interfaces.ModelArgsCvml("open-clip", "ViT-B-16-SigLIP/webli"),
            interfaces.ModelArgsCvml(
                "timm-vit", "vit_base_patch14_reg4_dinov2.lvd142m"
            ),
        ]
    )
    """CV models; a pair of model org (interface) and checkpoint."""
    models_vlm: typing.Annotated[
        list[interfaces.ModelArgsVlm], tyro.conf.arg(name="vlms")
    ] = dataclasses.field(
        default_factory=lambda: [
            # interfaces.ModelArgsVlm("openrouter/google/gemini-2.0-flash-001"),
            interfaces.ModelArgsVlm("openrouter/google/gemini-flash-1.5-8b"),
        ]
    )
    """VLM checkpoints."""
    device: typing.Literal["cpu", "cuda"] = "cuda"
    """which kind of accelerator to use."""
    debug: bool = False
    """whether to run in debug mode."""
    ssl: bool = True
    """Use SSL when connecting to remote servers to download checkpoints; use --no-ssl if your machine has certificate issues. See `biobench.third_party_models.get_ssl()` for a discussion of how this works."""

    # Individual benchmarks.
    ages_run_cvml: bool = False
    """Whether to run the bird age benchmark with CV+ML."""
    ages_run_vlm: bool = False
    """Whether to run the bird age benchmark with VLM."""
    ages_args: ages.Args = dataclasses.field(default_factory=ages.Args)
    """Arguments for the bird age benchmark."""

    beluga_run_cvml: bool = False
    """Whether to run the Beluga whale re-ID benchmark with CV+ML."""
    beluga_run_vlm: bool = False
    """Whether to run the Beluga whale re-ID benchmark with VLM."""
    beluga_args: beluga.Args = dataclasses.field(default_factory=beluga.Args)
    """Arguments for the Beluga whale re-ID benchmark."""

    birds525_run_cvml: bool = False
    """Whether to run the Birds 525 benchmark with CV+ML."""
    birds525_run_vlm: bool = False
    """Whether to run the Birds 525 benchmark with VLM."""
    birds525_args: birds525.Args = dataclasses.field(default_factory=birds525.Args)
    """Arguments for the Birds 525 benchmark."""

    fishnet_run_cvml: bool = False
    """Whether to run the FishNet benchmark with CV+ML."""
    fishnet_run_vlm: bool = False
    """Whether to run the FishNet benchmark with VLM."""
    fishnet_args: fishnet.Args = dataclasses.field(default_factory=fishnet.Args)
    """Arguments for the FishNet benchmark."""

    imagenet_run_cvml: bool = False
    """Whether to run the ImageNet-1K benchmark with CV+ML."""
    imagenet_run_vlm: bool = False
    """Whether to run the ImageNet-1K benchmark with VLM."""
    imagenet_args: imagenet.Args = dataclasses.field(default_factory=imagenet.Args)
    """Arguments for the ImageNet-1K benchmark."""

    inat21_run_cvml: bool = False
    """Whether to run the iNat21 benchmark with CV+ML."""
    inat21_run_vlm: bool = False
    """Whether to run the iNat21 benchmark with VLM."""
    inat21_args: inat21.Args = dataclasses.field(default_factory=inat21.Args)
    """Arguments for the iNat21 benchmark."""

    iwildcam_run_cvml: bool = False
    """Whether to run the iWildCam benchmark with CV+ML."""
    iwildcam_run_vlm: bool = False
    """Whether to run the iWildCam benchmark with VLM."""
    iwildcam_args: iwildcam.Args = dataclasses.field(default_factory=iwildcam.Args)
    """Arguments for the iWildCam benchmark."""

    kabr_run_cvml: bool = False
    """Whether to run the KABR benchmark with CV+ML."""
    kabr_run_vlm: bool = False
    """Whether to run the KABR benchmark with VLM."""
    kabr_args: kabr.Args = dataclasses.field(default_factory=kabr.Args)
    """Arguments for the KABR benchmark."""

    leopard_run_cvml: bool = False
    """Whether to run the leopard re-ID benchmark with CV+ML."""
    leopard_run_vlm: bool = False
    """Whether to run the leopard re-ID benchmark with VLM."""
    leopard_args: leopard.Args = dataclasses.field(default_factory=leopard.Args)
    """Arguments for the leopard re-ID benchmark."""

    newt_run_cvml: bool = False
    """Whether to run the NeWT benchmark with CV+ML."""
    newt_run_vlm: bool = False
    """Whether to run the NeWT benchmark with VLM."""
    newt_args: newt.Args = dataclasses.field(default_factory=newt.Args)
    """Arguments for the NeWT benchmark."""

    plankton_run_cvml: bool = False
    """Whether to run the Plankton benchmark with CV+ML."""
    plankton_run_vlm: bool = False
    """Whether to run the Plankton benchmark with VLM."""
    plankton_args: plankton.Args = dataclasses.field(default_factory=plankton.Args)
    """Arguments for the Plankton benchmark."""

    plantnet_run_cvml: bool = False
    """Whether to run the Pl@ntNet benchmark with CV+ML."""
    plantnet_run_vlm: bool = False
    """Whether to run the Pl@ntNet benchmark with VLM."""
    plantnet_args: plantnet.Args = dataclasses.field(default_factory=plantnet.Args)
    """Arguments for the Pl@ntNet benchmark."""

    rarespecies_run_cvml: bool = False
    """Whether to run the Rare Species benchmark with CV+ML."""
    rarespecies_run_vlm: bool = False
    """Whether to run the Rare Species benchmark with VLM."""
    rarespecies_args: rarespecies.Args = dataclasses.field(
        default_factory=rarespecies.Args
    )
    """Arguments for the Rare Species benchmark."""

    # Reporting and graphing.
    report_to: str = os.path.join(".", "reports")
    """where to save reports to."""
    graph: bool = True
    """whether to make graphs."""
    graph_to: str = os.path.join(".", "graphs")
    """where to save graphs to."""
    log_to: str = os.path.join(".", "logs")
    """where to save logs to."""

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def get_sqlite_connection(self) -> sqlite3.Connection:
        """Get a connection to the reports database.
        Returns:
            a connection to a sqlite3 database.
        """
        return sqlite3.connect(os.path.join(self.report_to, "reports.sqlite"))


@beartype.beartype
def save(
    args: Args,
    model_args: interfaces.ModelArgsCvml | interfaces.ModelArgsVlm,
    report: interfaces.TaskReport,
) -> None:
    """
    Saves the report to disk in a machine-readable SQLite format.

    Args:
        args: launch script arguments.
        model_args: a pair of model_org, model_ckpt strings.
        report: the task report from the model_args.
    """
    conn = args.get_sqlite_connection()
    with open("schema.sql") as fd:
        schema = fd.read()
    conn.execute(schema)

    lower, upper = report.get_confidence_interval()
    values = (
        json.dumps(dataclasses.asdict(model_args)),
        report.name,
        int(time.time()),
        report.get_mean_score(),
        lower,
        upper,
        json.dumps(dataclasses.asdict(args)),
        json.dumps(report.to_dict()),
    )
    conn.execute("INSERT INTO reports VALUES(?, ?, ?, ?, ?, ?, ?, ?)", values)
    conn.commit()

    logger.info(
        "%s on %s: %.1f%%", model_args.ckpt, report.name, report.get_mean_score() * 100
    )
    for name, score in report.splits.items():
        logger.info("%s on %s (%s): %.3f", model_args.ckpt, report.name, name, score)


@beartype.beartype
def export_to_csv(args: Args) -> set[str]:
    """
    Exports (and writes) to a wide table format for viewing (long table formats are better for additional manipulation/graphing, but wide is easy for viewing).
    """
    conn = args.get_sqlite_connection()
    stmt = """
    SELECT model_config, task_name, mean_score, MAX(posix) AS posix 
    FROM reports 
    GROUP BY model_config, task_name 
    ORDER BY model_config ASC;
    """
    data = conn.execute(stmt, ()).fetchall()

    tasks = set()
    rows = collections.defaultdict(lambda: collections.defaultdict(float))
    for model_config, task_name, mean_score, _ in data:
        ckpt = json.loads(model_config)["ckpt"]
        rows[ckpt][task_name] = mean_score
        tasks.add(task_name)

    for model, scores in rows.items():
        scores["mean"] = np.mean([scores[task] for task in tasks]).item()

    path = os.path.join(args.report_to, "results.csv")
    with open(path, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["model", "mean"] + sorted(tasks))
        for model in sorted(rows, key=lambda model: rows[model]["mean"], reverse=True):
            scores = rows[model]
            writer.writerow(
                [model, scores["mean"]] + [scores[task] for task in sorted(tasks)]
            )
    logger.info("Wrote results to '%s'.", path)
    return tasks


@beartype.beartype
def main(args: Args):
    """
    Launch all jobs, using either a local GPU or a Slurm cluster.
    Then report results and save to disk.
    """
    # 1. Setup executor.
    if args.slurm:
        executor = submitit.SlurmExecutor(folder=args.log_to)
        executor.update_parameters(
            time=30,
            gpus_per_node=1,
            cpus_per_task=8,
            stderr_to_stdout=True,
            partition="debug",
            account=args.slurm_acct,
        )
        # See biobench.third_party_models.get_ssl() for a discussion of this variable.
        if not args.ssl:
            executor.update_parameters(setup=["export BIOBENCH_DISABLE_SSL=1"])
    else:
        executor = submitit.DebugExecutor(folder=args.log_to)
        # See biobench.third_party_models.get_ssl() for a discussion of this variable.
        if not args.ssl:
            os.environ["BIOBENCH_DISABLE_SSL"] = "1"

    # 2. Run benchmarks.
    jobs = []
    for model_args in args.models_cvml:
        if args.ages_run_cvml:
            ages_args = dataclasses.replace(
                args.ages_args, device=args.device, debug=args.debug
            )
            job = executor.submit(ages.benchmark_cvml, ages_args, model_args)
            jobs.append(job)
        if args.beluga_run_cvml:
            beluga_args = dataclasses.replace(
                args.beluga_args, device=args.device, debug=args.debug
            )
            job = executor.submit(beluga.benchmark_cvml, beluga_args, model_args)
            jobs.append(job)
        if args.birds525_run_cvml:
            birds525_args = dataclasses.replace(
                args.birds525_args, device=args.device, debug=args.debug
            )
            job = executor.submit(birds525.benchmark_cvml, birds525_args, model_args)
            jobs.append(job)
        if args.fishnet_run_cvml:
            fishnet_args = dataclasses.replace(
                args.fishnet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(fishnet.benchmark_cvml, fishnet_args, model_args)
            jobs.append(job)
        if args.imagenet_run_cvml:
            imagenet_args = dataclasses.replace(
                args.imagenet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(imagenet.benchmark_cvml, imagenet_args, model_args)
            jobs.append(job)
        if args.inat21_run_cvml:
            inat21_args = dataclasses.replace(
                args.inat21_args, device=args.device, debug=args.debug
            )
            job = executor.submit(inat21.benchmark_cvml, inat21_args, model_args)
            jobs.append(job)
        if args.iwildcam_run_cvml:
            iwildcam_args = dataclasses.replace(
                args.iwildcam_args, device=args.device, debug=args.debug
            )
            job = executor.submit(iwildcam.benchmark_cvml, iwildcam_args, model_args)
            jobs.append(job)
        if args.kabr_run_cvml:
            kabr_args = dataclasses.replace(
                args.kabr_args, device=args.device, debug=args.debug
            )
            jobs.append(executor.submit(kabr.benchmark_cvml, kabr_args, model_args))
        if args.leopard_run_cvml:
            leopard_args = dataclasses.replace(
                args.leopard_args, device=args.device, debug=args.debug
            )
            job = executor.submit(leopard.benchmark_cvml, leopard_args, model_args)
            jobs.append(job)
        # Newt
        if args.newt_run_cvml:
            newt_args = dataclasses.replace(
                args.newt_args, device=args.device, debug=args.debug
            )
            jobs.append(executor.submit(newt.benchmark_cvml, newt_args, model_args))

        if args.plankton_run_cvml:
            plankton_args = dataclasses.replace(
                args.plankton_args, device=args.device, debug=args.debug
            )
            job = executor.submit(plankton.benchmark, plankton_args, model_args)
            jobs.append(job)
        if args.plantnet_run_cvml:
            plantnet_args = dataclasses.replace(
                args.plantnet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(plantnet.benchmark, plantnet_args, model_args)
            jobs.append(job)
        if args.rarespecies_run_cvml:
            rarespecies_args = dataclasses.replace(
                args.rarespecies_args, device=args.device, debug=args.debug
            )
            job = executor.submit(rarespecies.benchmark, rarespecies_args, model_args)
            jobs.append(job)

    for model_args in args.models_vlm:
        if args.ages_run_vlm:
            ages_args = dataclasses.replace(
                args.ages_args, device=args.device, debug=args.debug
            )
            job = executor.submit(ages.benchmark_vlm, ages_args, model_args)
            jobs.append(job)
        if args.beluga_run_vlm:
            beluga_args = dataclasses.replace(
                args.beluga_args, device=args.device, debug=args.debug
            )
            job = executor.submit(beluga.benchmark_vlm, beluga_args, model_args)
            jobs.append(job)
        if args.birds525_run_vlm:
            birds525_args = dataclasses.replace(
                args.birds525_args, device=args.device, debug=args.debug
            )
            job = executor.submit(birds525.benchmark_vlm, birds525_args, model_args)
            jobs.append(job)
        if args.fishnet_run_vlm:
            fishnet_args = dataclasses.replace(
                args.fishnet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(fishnet.benchmark_vlm, fishnet_args, model_args)
            jobs.append(job)
        if args.imagenet_run_vlm:
            imagenet_args = dataclasses.replace(
                args.imagenet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(imagenet.benchmark_vlm, imagenet_args, model_args)
            jobs.append(job)
        if args.inat21_run_vlm:
            inat21_args = dataclasses.replace(
                args.inat21_args, device=args.device, debug=args.debug
            )
            job = executor.submit(inat21.benchmark_vlm, inat21_args, model_args)
            jobs.append(job)
        if args.iwildcam_run_vlm:
            iwildcam_args = dataclasses.replace(
                args.iwildcam_args, device=args.device, debug=args.debug
            )
            job = executor.submit(iwildcam.benchmark_vlm, iwildcam_args, model_args)
            jobs.append(job)
        if args.kabr_run_vlm:
            kabr_args = dataclasses.replace(
                args.kabr_args, device=args.device, debug=args.debug
            )
            jobs.append(executor.submit(kabr.benchmark_vlm, kabr_args, model_args))
        if args.leopard_run_vlm:
            leopard_args = dataclasses.replace(
                args.leopard_args, device=args.device, debug=args.debug
            )
            job = executor.submit(leopard.benchmark_vlm, leopard_args, model_args)
            jobs.append(job)
        # Newt
        if args.newt_run_vlm:
            newt_args = dataclasses.replace(
                args.newt_args, device=args.device, debug=args.debug
            )
            jobs.append(executor.submit(newt.benchmark_vlm, newt_args, model_args))

        if args.plankton_run_vlm:
            plankton_args = dataclasses.replace(
                args.plankton_args, device=args.device, debug=args.debug
            )
            job = executor.submit(plankton.benchmark, plankton_args, model_args)
            jobs.append(job)
        if args.plantnet_run_vlm:
            plantnet_args = dataclasses.replace(
                args.plantnet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(plantnet.benchmark, plantnet_args, model_args)
            jobs.append(job)
        if args.rarespecies_run_vlm:
            rarespecies_args = dataclasses.replace(
                args.rarespecies_args, device=args.device, debug=args.debug
            )
            job = executor.submit(rarespecies.benchmark, rarespecies_args, model_args)
            jobs.append(job)
    logger.info("Submitted %d jobs.", len(jobs))

    # 3. Display results.
    os.makedirs(args.report_to, exist_ok=True)
    for i, future in enumerate(submitit.helpers.as_completed(jobs)):
        err = future.exception()
        if err:
            logger.warning("Error running job: %s: %s", err, err.__cause__)
            continue

        model_args, report = future.result()
        save(args, model_args, report)
        logger.info("Finished %d/%d jobs.", i + 1, len(jobs))

    # 4. Save results to CSV file for committing to git.
    tasks = export_to_csv(args)

    # 5. Make graphs.
    if args.graph:
        # For each combination of model/task, get the most recent version from the database. Then make a graph and save it to disk.
        conn = args.get_sqlite_connection()
        for task in sorted(tasks):
            fig = plot_task(conn, task)
            if fig is None:
                continue
            os.makedirs(args.graph_to, exist_ok=True)
            path = os.path.join(args.graph_to, f"{task}.png")
            fig.savefig(path)
            logger.info("Saved fig for %s to %s.", task, path)

    logger.info("Finished.")


@beartype.beartype
def plot_task(conn: sqlite3.Connection, task: str):
    """
    Plots the most recent result for each model on given task, including confidence intervals.
    Returns the figure so the caller can save or display it.

    Args:
        conn: connection to database.
        task: which task to run.

    Returns:
        matplotlib.pyplot.Figure
    """
    import matplotlib.pyplot as plt

    orig_row_factory = conn.row_factory

    conn.row_factory = sqlite3.Row
    fig, ax = plt.subplots()
    stmt = "SELECT model_config, task_name, mean_score, confidence_lower, confidence_upper, MAX(posix) FROM reports WHERE task_name = (?) GROUP BY model_config, task_name ORDER BY model_config ASC;"
    data = conn.execute(stmt, (task,)).fetchall()

    conn.row_factory = orig_row_factory

    if not data:
        return

    xs = [json.loads(row["model_config"])["ckpt"] for row in data]
    ys = [row["mean_score"] for row in data]

    yerr = np.array([ys, ys])
    yerr[0] = np.maximum(yerr[0] - [row["confidence_lower"] for row in data], 0)
    yerr[1] = np.maximum([row["confidence_upper"] for row in data] - yerr[1], 0)

    ax.errorbar(xs, ys, yerr, fmt="o", linewidth=2, capsize=6)
    ax.set_title(f"Mean {task} Performance")
    ax.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    args = tyro.cli(Args)

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    min_nofile = 1024 * 8
    if soft < min_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min_nofile, hard))

    main(args)
