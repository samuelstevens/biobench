"""
Entrypoint for running benchmarking.
"""

import logging
import os
import resource
import typing

import beartype
import submitit
import tyro

from biobench import config, newt, reporting

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("benchmark.py")


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
            job = executor.submit(newt.benchmark_cvml, cfg)
            jobs.append(job)
        elif cfg.model.method == "mllm":
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

        reports: list[reporting.Report] = future.result()
        for report in reports:
            report.write()
        logger.info("Finished %d/%d jobs.", i + 1, len(jobs))

    logger.info("Finished.")


if __name__ == "__main__":
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    min_nofile = 1024 * 8
    if soft < min_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min_nofile, hard))
    tyro.cli(benchmark)
