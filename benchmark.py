"""
Entrypoint for running all tasks in `biobench`.

Most of this script is self documenting.
Run `python benchmark.py --help` to see all the options.

Note that you will have to download all the datasets, but each dataset includes its own download script with instructions.
For example, see `biobench.newt.download` for an example.

.. include:: ./examples.md

.. include:: ./design.md
"""

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
    # ages,
    # beluga,
    # birds525,
    config,
    # fishnet,
    # imagenet,
    # inat21,
    interfaces,
    # iwildcam,
    # kabr,
    # leopard,
    newt,
    # plankton,
    # plantnet,
    # rarespecies,
)

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("benchmark.py")


@beartype.beartype
def save(report: interfaces.Report) -> None:
    """
    Saves the report to disk in a machine-readable SQLite format.

    Args:
        args: launch script arguments.
        model_args: a pair of model_org, model_ckpt strings.
        report: the task report from the model_args.
    """
    os.makedirs(report.exp_cfg.report_to, exist_ok=True)
    conn = sqlite3.connect(os.path.join(report.exp_cfg.report_to, "results.sqlite"))
    with open("schema.sql") as fd:
        schema = fd.read()
    conn.execute(schema)

    lower, upper = report.get_confidence_interval()

    values = (
        int(time.time()),
        report.task_name,
        report.n_train,
        len(report.predictions),
        report.exp_cfg.sampling,
        report.exp_cfg.model.method,
        report.exp_cfg.model.org,
        report.exp_cfg.model.ckpt,
        report.get_mean_score(),
        lower,
        upper,
        # MLLM
        report.exp_cfg.prompting,
        report.exp_cfg.cot_enabled,
        report.parse_success_rate,
        report.usd_per_answer,
        # CVML
        report.classifier,
        # JSON blobs
        json.dumps(report.exp_cfg.to_dict()),
        json.dumps(report.to_dict()),
    )
    conn.execute(
        "INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        values,
    )
    conn.commit()

    logger.info(
        "%s with %d (%d actual) examples on %s: %.3f",
        report.exp_cfg.model.ckpt,
        report.exp_cfg.n_train,
        report.n_train,
        report.task_name,
        report.get_mean_score(),
    )


@beartype.beartype
def benchmark(cfg: str):
    """
    Launch all jobs, using either a local GPU or a Slurm cluster.
    Then report results and save to disk.
    """
    cfgs = config.load(cfg)

    if not cfgs:
        # Do something
        pass

    first = cfgs[0]
    # Verify all configs have consistent execution settings
    for cfg in cfgs[1:]:
        if cfg.slurm != first.slurm:
            raise ValueError("All configs must have the same value for slurm")
        if cfg.slurm and cfg.slurm_acct != first.slurm_acct:
            raise ValueError(
                "All configs must have the same slurm_acct when slurm=True"
            )
        if cfg.log_to != first.log_to:
            raise ValueError("All configs must have the same log_to directory")
        if cfg.ssl != first.ssl:
            raise ValueError("All configs must have the same ssl setting")

    # 1. Setup executor.
    if first.slurm:
        executor = submitit.SlurmExecutor(folder=first.log_to)
        executor.update_parameters(
            time=30,
            gpus_per_node=1,
            cpus_per_task=8,
            stderr_to_stdout=True,
            partition="debug",
            account=first.slurm_acct,
        )
        # See biobench.third_party_models.get_ssl() for a discussion of this variable.
        if not first.ssl:
            executor.update_parameters(setup=["export BIOBENCH_DISABLE_SSL=1"])
    else:
        executor = submitit.DebugExecutor(folder=first.log_to)
        # See biobench.third_party_models.get_ssl() for a discussion of this variable.
        if not first.ssl:
            os.environ["BIOBENCH_DISABLE_SSL"] = "1"

    # 2. Run benchmarks.
    jobs = []
    for cfg in cfgs:
        if cfg.model.method == "cvml":
            # if cfg.fishnet_data:
            #     job = executor.submit(fishnet.benchmark_cvml, cfg)
            #     jobs.append(job)
            if cfg.newt_data:
                job = executor.submit(newt.benchmark_cvml, cfg)
                jobs.append(job)
        elif cfg.model.method == "mllm":
            # if cfg.fishnet_data:
            #     job = executor.submit(fishnet.benchmark_mllm, cfg)
            #     jobs.append(job)
            if cfg.newt_data:
                job = executor.submit(newt.benchmark_mllm, cfg)
                jobs.append(job)
        else:
            typing.assert_never(cfg.model.method)

    logger.info("Submitted %d jobs.", len(jobs))

    # 3. Write results to sqlite.
    for i, future in enumerate(submitit.helpers.as_completed(jobs)):
        err = future.exception()
        if err:
            logger.warning("Error running job: %s: %s", err, err.__cause__)
            continue

        report = future.result()
        save(report)
        logger.info("Finished %d/%d jobs.", i + 1, len(jobs))

    logger.info("Finished.")


if __name__ == "__main__":
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    min_nofile = 1024 * 8
    if soft < min_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min_nofile, hard))
    tyro.cli(benchmark)
