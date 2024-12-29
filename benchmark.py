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

    model_args: typing.Annotated[
        list[interfaces.ModelArgs], tyro.conf.arg(name="model")
    ] = dataclasses.field(
        default_factory=lambda: [
            interfaces.ModelArgs("open-clip", "RN50/openai"),
            interfaces.ModelArgs("open-clip", "ViT-B-16/openai"),
            interfaces.ModelArgs("open-clip", "ViT-B-16/laion400m_e32"),
            interfaces.ModelArgs("open-clip", "hf-hub:imageomics/bioclip"),
            interfaces.ModelArgs("open-clip", "ViT-B-16/facebook/dinov2-base"),
            interfaces.ModelArgs("open-clip", "ViT-B-16-SigLIP/webli"),
            interfaces.ModelArgs("timm-vit", "vit_base_patch14_reg4_dinov2.lvd142m"),
        ]
    )
    """model; a pair of model org (interface) and checkpoint."""
    device: typing.Literal["cpu", "cuda"] = "cuda"
    """which kind of accelerator to use."""
    debug: bool = False
    """whether to run in debug mode."""
    ssl: bool = True
    """Use SSL when connecting to remote servers to download checkpoints; use --no-ssl if your machine has certificate issues. See `biobench.third_party_models.get_ssl()` for a discussion of how this works."""

    # Individual benchmarks.
    ages_run: bool = False
    """Whether to run the bird age benchmark."""
    ages_args: ages.Args = dataclasses.field(default_factory=ages.Args)
    """Arguments for the bird age benchmark."""
    beluga_run: bool = False
    """Whether to run the Beluga whale re-ID benchmark."""
    beluga_args: beluga.Args = dataclasses.field(default_factory=beluga.Args)
    """Arguments for the Beluga whale re-ID benchmark."""
    birds525_run: bool = False
    """whether to run the Birds 525 benchmark."""
    birds525_args: birds525.Args = dataclasses.field(default_factory=birds525.Args)
    """arguments for the Birds 525 benchmark."""
    fishnet_run: bool = False
    """Whether to run the FishNet benchmark."""
    fishnet_args: fishnet.Args = dataclasses.field(default_factory=fishnet.Args)
    """Arguments for the FishNet benchmark."""
    imagenet_run: bool = False
    """Whether to run the ImageNet-1K benchmark."""
    imagenet_args: imagenet.Args = dataclasses.field(default_factory=imagenet.Args)
    """Arguments for the ImageNet-1K benchmark."""
    inat21_run: bool = False
    """Whether to run the iNat21 benchmark."""
    inat21_args: inat21.Args = dataclasses.field(default_factory=inat21.Args)
    """Arguments for the iNat21 benchmark."""
    iwildcam_run: bool = False
    """whether to run the iWildCam benchmark."""
    iwildcam_args: iwildcam.Args = dataclasses.field(default_factory=iwildcam.Args)
    """arguments for the iWildCam benchmark."""
    kabr_run: bool = False
    """whether to run the KABR benchmark."""
    kabr_args: kabr.Args = dataclasses.field(default_factory=kabr.Args)
    """arguments for the KABR benchmark."""
    leopard_run: bool = False
    """Whether to run the leopard re-ID benchmark."""
    leopard_args: leopard.Args = dataclasses.field(default_factory=leopard.Args)
    """Arguments for the leopard re-ID benchmark."""
    newt_run: bool = False
    """whether to run the NeWT benchmark."""
    newt_args: newt.Args = dataclasses.field(default_factory=newt.Args)
    """arguments for the NeWT benchmark."""
    plankton_run: bool = False
    """Whether to run the Plankton benchmark."""
    plankton_args: plankton.Args = dataclasses.field(default_factory=plankton.Args)
    """Arguments for the Plankton benchmark."""
    plantnet_run: bool = False
    """whether to run the Pl@ntNet benchmark."""
    plantnet_args: plantnet.Args = dataclasses.field(default_factory=plantnet.Args)
    """arguments for the Pl@ntNet benchmark."""
    rarespecies_run: bool = False
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
    args: Args, model_args: interfaces.ModelArgs, report: interfaces.TaskReport
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

    model_org, model_ckpt = model_args
    lower, upper = report.get_confidence_interval()
    values = (
        model_org,
        model_ckpt,
        report.name,
        int(time.time()),
        report.get_mean_score(),
        lower,
        upper,
        json.dumps(dataclasses.asdict(args)),
        json.dumps(report.to_dict()),
    )
    conn.execute("INSERT INTO reports VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
    conn.commit()

    logger.info(
        "%s on %s: %.1f%%", model_ckpt, report.name, report.get_mean_score() * 100
    )
    for name, score in report.splits.items():
        logger.info("%s on %s (%s): %.3f", model_ckpt, report.name, name, score)


@beartype.beartype
def export_to_csv(args: Args) -> set[str]:
    """
    Exports (and writes) to a wide table format for viewing (long table formats are better for additional manipulation/graphing, but wide is easy for viewing).
    """
    conn = args.get_sqlite_connection()
    stmt = """
    SELECT model_ckpt, task_name, mean_score, MAX(posix) AS posix 
    FROM reports 
    GROUP BY model_ckpt, task_name 
    ORDER BY model_ckpt ASC;
    """
    data = conn.execute(stmt, ()).fetchall()

    tasks = set()
    rows = collections.defaultdict(lambda: collections.defaultdict(float))
    for model_ckpt, task_name, mean_score, _ in data:
        rows[model_ckpt][task_name] = mean_score
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
    for model_args in args.model_args:
        if args.ages_run:
            ages_args = dataclasses.replace(
                args.ages_args, device=args.device, debug=args.debug
            )
            job = executor.submit(ages.benchmark, ages_args, model_args)
            jobs.append(job)
        if args.beluga_run:
            beluga_args = dataclasses.replace(
                args.beluga_args, device=args.device, debug=args.debug
            )
            job = executor.submit(beluga.benchmark, beluga_args, model_args)
            jobs.append(job)
        if args.birds525_run:
            birds525_args = dataclasses.replace(
                args.birds525_args, device=args.device, debug=args.debug
            )
            job = executor.submit(birds525.benchmark, birds525_args, model_args)
            jobs.append(job)
        if args.fishnet_run:
            fishnet_args = dataclasses.replace(
                args.fishnet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(fishnet.benchmark, fishnet_args, model_args)
            jobs.append(job)
        if args.imagenet_run:
            imagenet_args = dataclasses.replace(
                args.imagenet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(imagenet.benchmark, imagenet_args, model_args)
            jobs.append(job)
        if args.inat21_run:
            inat21_args = dataclasses.replace(
                args.inat21_args, device=args.device, debug=args.debug
            )
            job = executor.submit(inat21.benchmark, inat21_args, model_args)
            jobs.append(job)
        if args.iwildcam_run:
            iwildcam_args = dataclasses.replace(
                args.iwildcam_args, device=args.device, debug=args.debug
            )
            job = executor.submit(iwildcam.benchmark, iwildcam_args, model_args)
            jobs.append(job)
        if args.kabr_run:
            kabr_args = dataclasses.replace(
                args.kabr_args, device=args.device, debug=args.debug
            )
            jobs.append(executor.submit(kabr.benchmark, kabr_args, model_args))
        if args.leopard_run:
            leopard_args = dataclasses.replace(
                args.leopard_args, device=args.device, debug=args.debug
            )
            job = executor.submit(leopard.benchmark, leopard_args, model_args)
            jobs.append(job)
        if args.newt_run:
            newt_args = dataclasses.replace(
                args.newt_args, device=args.device, debug=args.debug
            )
            jobs.append(executor.submit(newt.benchmark, newt_args, model_args))
        if args.plankton_run:
            plankton_args = dataclasses.replace(
                args.plankton_args, device=args.device, debug=args.debug
            )
            job = executor.submit(plankton.benchmark, plankton_args, model_args)
            jobs.append(job)
        if args.plantnet_run:
            plantnet_args = dataclasses.replace(
                args.plantnet_args, device=args.device, debug=args.debug
            )
            job = executor.submit(plantnet.benchmark, plantnet_args, model_args)
            jobs.append(job)
        if args.rarespecies_run:
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
    stmt = "SELECT model_ckpt, task_name, mean_score, confidence_lower, confidence_upper, MAX(posix) FROM reports WHERE task_name = (?) GROUP BY model_ckpt, task_name ORDER BY model_ckpt ASC;"
    data = conn.execute(stmt, (task,)).fetchall()

    conn.row_factory = orig_row_factory

    if not data:
        return

    xs = [row["model_ckpt"].split("/")[-1] for row in data]
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
