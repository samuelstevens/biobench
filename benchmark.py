"""
Entrypoint for running all tasks in `biobench`.

Most of this script is self documenting.
Run `python benchmark.py --help` to see all the options.

Note that you will have to download all the datasets, but each dataset includes its own download script with instructions.
For example, see `biobench.newt.download` for an example.

.. include:: ./examples.md

.. include:: ./design.md
"""

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
    models_mllm: typing.Annotated[
        list[interfaces.ModelArgsMllm], tyro.conf.arg(name="mllms")
    ] = dataclasses.field(
        default_factory=lambda: [
            interfaces.ModelArgsMllm(
                "openrouter/meta-llama/llama-3.2-3b-instruct",
                quantizations=["fp32", "bf16"],
            ),
            interfaces.ModelArgsMllm(
                "openrouter/qwen/qwen-2-vl-7b-instruct", quantizations=["fp32", "bf16"]
            ),
            interfaces.ModelArgsMllm("openrouter/google/gemini-flash-1.5-8b"),
        ]
    )
    """MLLM checkpoints."""
    device: typing.Literal["cpu", "mps", "cuda"] = "cuda"
    """which kind of accelerator to use."""
    debug: bool = False
    """whether to run in debug mode."""
    n_train: int = -1
    """Number of maximum training samples. Negative number means use all of them."""
    n_test: int = -1
    """Number of test samples. Negative number means use all of them."""
    parallel: int = 1
    """Number of parallel requests per second to MLLM service providers."""

    ssl: bool = True
    """Use SSL when connecting to remote servers to download checkpoints; use --no-ssl if your machine has certificate issues. See `biobench.third_party_models.get_ssl()` for a discussion of how this works."""

    # Individual benchmarks.
    ages_run_cvml: bool = False
    """Whether to run the bird age benchmark with CV+ML."""
    ages_run_mllm: bool = False
    """Whether to run the bird age benchmark with MLLM."""
    ages_args: ages.Args = dataclasses.field(default_factory=ages.Args)
    """Arguments for the bird age benchmark."""

    beluga_run_cvml: bool = False
    """Whether to run the Beluga whale re-ID benchmark with CV+ML."""
    beluga_run_mllm: bool = False
    """Whether to run the Beluga whale re-ID benchmark with MLLM."""
    beluga_args: beluga.Args = dataclasses.field(default_factory=beluga.Args)
    """Arguments for the Beluga whale re-ID benchmark."""

    birds525_run_cvml: bool = False
    """Whether to run the Birds 525 benchmark with CV+ML."""
    birds525_run_mllm: bool = False
    """Whether to run the Birds 525 benchmark with MLLM."""
    birds525_args: birds525.Args = dataclasses.field(default_factory=birds525.Args)
    """Arguments for the Birds 525 benchmark."""

    fishnet_run_cvml: bool = False
    """Whether to run the FishNet benchmark with CV+ML."""
    fishnet_run_mllm: bool = False
    """Whether to run the FishNet benchmark with MLLM."""
    fishnet_args: fishnet.Args = dataclasses.field(default_factory=fishnet.Args)
    """Arguments for the FishNet benchmark."""

    imagenet_run_cvml: bool = False
    """Whether to run the ImageNet-1K benchmark with CV+ML."""
    imagenet_run_mllm: bool = False
    """Whether to run the ImageNet-1K benchmark with MLLM."""
    imagenet_args: imagenet.Args = dataclasses.field(default_factory=imagenet.Args)
    """Arguments for the ImageNet-1K benchmark."""

    inat21_run_cvml: bool = False
    """Whether to run the iNat21 benchmark with CV+ML."""
    inat21_run_mllm: bool = False
    """Whether to run the iNat21 benchmark with MLLM."""
    inat21_args: inat21.Args = dataclasses.field(default_factory=inat21.Args)
    """Arguments for the iNat21 benchmark."""

    iwildcam_run_cvml: bool = False
    """Whether to run the iWildCam benchmark with CV+ML."""
    iwildcam_run_mllm: bool = False
    """Whether to run the iWildCam benchmark with MLLM."""
    iwildcam_args: iwildcam.Args = dataclasses.field(default_factory=iwildcam.Args)
    """Arguments for the iWildCam benchmark."""

    kabr_run_cvml: bool = False
    """Whether to run the KABR benchmark with CV+ML."""
    kabr_run_mllm: bool = False
    """Whether to run the KABR benchmark with MLLM."""
    kabr_args: kabr.Args = dataclasses.field(default_factory=kabr.Args)
    """Arguments for the KABR benchmark."""

    leopard_run_cvml: bool = False
    """Whether to run the leopard re-ID benchmark with CV+ML."""
    leopard_run_mllm: bool = False
    """Whether to run the leopard re-ID benchmark with MLLM."""
    leopard_args: leopard.Args = dataclasses.field(default_factory=leopard.Args)
    """Arguments for the leopard re-ID benchmark."""

    newt_run_cvml: bool = False
    """Whether to run the NeWT benchmark with CV+ML."""
    newt_run_mllm: bool = False
    """Whether to run the NeWT benchmark with MLLM."""
    newt_args: newt.Args = dataclasses.field(default_factory=newt.Args)
    """Arguments for the NeWT benchmark."""

    plankton_run_cvml: bool = False
    """Whether to run the Plankton benchmark with CV+ML."""
    plankton_run_mllm: bool = False
    """Whether to run the Plankton benchmark with MLLM."""
    plankton_args: plankton.Args = dataclasses.field(default_factory=plankton.Args)
    """Arguments for the Plankton benchmark."""

    plantnet_run_cvml: bool = False
    """Whether to run the Pl@ntNet benchmark with CV+ML."""
    plantnet_run_mllm: bool = False
    """Whether to run the Pl@ntNet benchmark with MLLM."""
    plantnet_args: plantnet.Args = dataclasses.field(default_factory=plantnet.Args)
    """Arguments for the Pl@ntNet benchmark."""

    rarespecies_run_cvml: bool = False
    """Whether to run the Rare Species benchmark with CV+ML."""
    rarespecies_run_mllm: bool = False
    """Whether to run the Rare Species benchmark with MLLM."""
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

    def update(self, other):
        return dataclasses.replace(
            other,
            device=self.device,
            debug=self.debug,
            n_train=self.n_train,
            n_test=self.n_test,
            parallel=self.parallel,
        )


@beartype.beartype
def save(
    args: Args,
    model_args: interfaces.ModelArgsCvml | interfaces.ModelArgsMllm,
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
        json.dumps(model_args.to_dict()),
        report.name,
        int(time.time()),
        report.get_mean_score(),
        lower,
        upper,
        json.dumps(args.to_dict()),
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
def make_tables(args: Args):
    """TODO: document."""

    # CSV
    columns = [
        "task",
        "model",
        "n_train",
        "n_test",
        "sampling",
        "prompting",  # MLLM only
        "classifier",  # CV+ML only
        "hparams_json",  # Both CV+ML/MLLM
        "confidence_lower",
        "mean_score",
        "confidence_upper",
        "n_successful_parses",  # MLLM only
        "usd_per_answer",  # MLLM only
    ]

    conn = args.get_sqlite_connection()
    stmt = """
    SELECT task, model_config, confidence_lower, mean_score, confidence_upper, args, MAX(posix) AS posix 
    FROM reports 
    GROUP BY model_config, task, args
    ORDER BY task, model_config, args ASC;
    """
    data = conn.execute(stmt, ()).fetchall()

    with open(os.path.join(args.report_to, "table.csv"), "w") as fd:
        writer = csv.DictWriter(fd, columns)
        writer.writeheader()
        for task, model_args_json, ci_lower, mean, ci_upper, task_args_json, _ in data:
            model_args = json.loads(model_args_json)
            task_args = json.loads(task_args_json)

            if model_args["type"] == "mllm":
                hparams = {"temp": model_args["temp"]}
            elif model_args["type"] == "cvml":
                hparams = {}
            else:
                raise ValueError(model_args)

            # breakpoint()
            writer.writerow({
                "task": task,
                "model": model_args["ckpt"],
                "n_train": task_args["n_train"],
                "n_test": task_args["n_test"],
                "sampling": "uniform",
                "prompting": model_args.get("prompts", ""),
                "classifier": model_args.get("classifier", ""),
                "hparams_json": json.dumps(hparams),
                "confidence_lower": ci_lower,
                "mean_score": mean,
                "confidence_upper": ci_upper,
            })

    # LaTeX


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

    ages_args = args.update(args.ages_args)
    beluga_args = args.update(args.beluga_args)
    birds525_args = args.update(args.birds525_args)
    fishnet_args = args.update(args.fishnet_args)
    imagenet_args = args.update(args.imagenet_args)
    inat21_args = args.update(args.inat21_args)
    iwildcam_args = args.update(args.iwildcam_args)
    kabr_args = args.update(args.kabr_args)
    leopard_args = args.update(args.leopard_args)
    newt_args = args.update(args.newt_args)
    plankton_args = args.update(args.plankton_args)
    plantnet_args = args.update(args.plantnet_args)
    rarespecies_args = args.update(args.rarespecies_args)

    # 2. Run benchmarks.
    jobs = []
    for model_args in args.models_cvml:
        if args.ages_run_cvml:
            job = executor.submit(ages.benchmark_cvml, ages_args, model_args)
            jobs.append(job)
        if args.beluga_run_cvml:
            job = executor.submit(beluga.benchmark_cvml, beluga_args, model_args)
            jobs.append(job)
        if args.birds525_run_cvml:
            job = executor.submit(birds525.benchmark_cvml, birds525_args, model_args)
            jobs.append(job)
        if args.fishnet_run_cvml:
            job = executor.submit(fishnet.benchmark_cvml, fishnet_args, model_args)
            jobs.append(job)
        if args.imagenet_run_cvml:
            job = executor.submit(imagenet.benchmark_cvml, imagenet_args, model_args)
            jobs.append(job)
        if args.inat21_run_cvml:
            job = executor.submit(inat21.benchmark_cvml, inat21_args, model_args)
            jobs.append(job)
        if args.iwildcam_run_cvml:
            job = executor.submit(iwildcam.benchmark_cvml, iwildcam_args, model_args)
            jobs.append(job)
        if args.kabr_run_cvml:
            job = executor.submit(kabr.benchmark_cvml, kabr_args, model_args)
            jobs.append()
        if args.leopard_run_cvml:
            job = executor.submit(leopard.benchmark_cvml, leopard_args, model_args)
            jobs.append(job)
        if args.newt_run_cvml:
            jobs.append(executor.submit(newt.benchmark_cvml, newt_args, model_args))
        if args.plankton_run_cvml:
            job = executor.submit(plankton.benchmark, plankton_args, model_args)
            jobs.append(job)
        if args.plantnet_run_cvml:
            job = executor.submit(plantnet.benchmark, plantnet_args, model_args)
            jobs.append(job)
        if args.rarespecies_run_cvml:
            job = executor.submit(rarespecies.benchmark, rarespecies_args, model_args)
            jobs.append(job)

    for model_args in args.models_mllm:
        if args.ages_run_mllm:
            job = executor.submit(ages.benchmark_mllm, ages_args, model_args)
            jobs.append(job)
        if args.beluga_run_mllm:
            job = executor.submit(beluga.benchmark_mllm, beluga_args, model_args)
            jobs.append(job)
        if args.birds525_run_mllm:
            job = executor.submit(birds525.benchmark_mllm, birds525_args, model_args)
            jobs.append(job)
        if args.fishnet_run_mllm:
            job = executor.submit(fishnet.benchmark_mllm, fishnet_args, model_args)
            jobs.append(job)
        if args.imagenet_run_mllm:
            job = executor.submit(imagenet.benchmark_mllm, imagenet_args, model_args)
            jobs.append(job)
        if args.inat21_run_mllm:
            job = executor.submit(inat21.benchmark_mllm, inat21_args, model_args)
            jobs.append(job)
        if args.iwildcam_run_mllm:
            job = executor.submit(iwildcam.benchmark_mllm, iwildcam_args, model_args)
            jobs.append(job)
        if args.kabr_run_mllm:
            jobs.append(executor.submit(kabr.benchmark_mllm, kabr_args, model_args))
        if args.leopard_run_mllm:
            job = executor.submit(leopard.benchmark_mllm, leopard_args, model_args)
            jobs.append(job)
        if args.newt_run_mllm:
            job = executor.submit(newt.benchmark_mllm, newt_args, model_args)
            jobs.append()
        if args.plankton_run_mllm:
            job = executor.submit(plankton.benchmark, plankton_args, model_args)
            jobs.append(job)
        if args.plantnet_run_mllm:
            job = executor.submit(plantnet.benchmark, plantnet_args, model_args)
            jobs.append(job)
        if args.rarespecies_run_mllm:
            job = executor.submit(rarespecies.benchmark, rarespecies_args, model_args)
            jobs.append(job)

    logger.info("Submitted %d jobs.", len(jobs))

    # 3. Write results to sqlite.
    os.makedirs(args.report_to, exist_ok=True)
    for i, future in enumerate(submitit.helpers.as_completed(jobs)):
        err = future.exception()
        if err:
            logger.warning("Error running job: %s: %s", err, err.__cause__)
            continue

        model_args, report = future.result()
        save(args, model_args, report)
        logger.info("Finished %d/%d jobs.", i + 1, len(jobs))

    # 4. Make the table in both CSV and LaTeX.
    make_tables(args)

    logger.info("Finished.")


if __name__ == "__main__":
    args = tyro.cli(Args)

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    min_nofile = 1024 * 8
    if soft < min_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min_nofile, hard))

    main(args)
