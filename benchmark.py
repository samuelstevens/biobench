"""
Entrypoint for running all tasks in `biobench`.

Most of this script is self documenting.
Run `python benchmark.py --help` to see all the options.

Note that you will have to download all the datasets, but each dataset includes its own download script with instructions.
For example, see `biobench.newt.download` for an example.

.. include:: ./design.md
"""

import collections
import importlib
import logging
import os
import resource

import beartype
import submitit
import tyro

from biobench import config, helpers, reporting

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("biobench")


@beartype.beartype
def main(
    cfgs: list[str] = [os.path.join("configs", "neurips.toml")], dry_run: bool = True
):
    """
    Launch all jobs, using either a local GPU or a Slurm cluster. Then report results and save to disk.

    Args:
        cfgs: List of paths to TOML config files.
        dry_run: If --no-dry-run, actually run experiment.
    """
    # Load all configs from the provided paths and concatenate them.
    # Simplify this code; try not to use an intermediate variable. AI!
    all_configs = []
    for cfg_path in cfgs:
        all_configs.extend(config.load(cfg_path))
    cfgs = all_configs

    if not cfgs:
        logger.warning("No configurations loaded.")
        return

    first = cfgs[0]
    # Verify all configs have consistent execution settings
    for cfg in cfgs[1:]:
        if cfg.slurm_acct != first.slurm_acct:
            raise ValueError("All configs must have the same slurm_acct")
        if cfg.log_to != first.log_to:
            raise ValueError("All configs must have the same log_to directory")
        if cfg.ssl != first.ssl:
            raise ValueError("All configs must have the same ssl setting")

    # 1. Setup executor.
    if first.slurm_acct:
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

    db = reporting.get_db(first)

    # 2. Run benchmarks.
    jobs = []
    n_skipped = 0
    for cfg in helpers.progress(cfgs, desc="submitting jobs"):
        for task_name, data_root in cfg.data.to_dict().items():
            # Check that you can get the task_name
            try:
                module = importlib.import_module(f"biobench.{task_name}")
            except ModuleNotFoundError:
                logger.warning("Could not find task '%s'.", task_name)
                continue

            if not data_root:
                continue

            if reporting.already_ran(db, cfg, task_name):
                n_skipped += 1
                continue
            elif dry_run:
                jobs.append(cfg)
            else:
                job = executor.submit(module.benchmark, cfg)
                jobs.append(job)

    if dry_run:
        # Summarize the jobs by model and training examples
        model_counts = collections.defaultdict(int)
        for job_cfg in jobs:
            key = (job_cfg.model.ckpt, job_cfg.n_train)
            model_counts[key] += 1

        # Check if there are any jobs to run
        if not model_counts:
            logger.info("All jobs have already been completed. Nothing to run.")
            return

        # Print summary table
        logger.info("Job Summary:")
        logger.info("%-40s | %-10s | %-5s", "Model", "Train Size", "Count")
        logger.info("-" * 61)
        for (model, n_train), count in sorted(model_counts.items()):
            logger.info("%-40s | %-10d | %-5d", model, n_train, count)
        logger.info("-" * 61)
        logger.info("Total jobs to run: %d", len(jobs))
        return

    logger.info("Submitted %d jobs (skipped %d).", len(jobs), n_skipped)

    # 3. Write results to sqlite.
    for i, future in enumerate(submitit.helpers.as_completed(jobs)):
        err = future.exception()
        if err:
            logger.warning("Error running job: %s: %s", err, err.__cause__)
            continue

        report: reporting.Report = future.result()
        report.write(db)
        logger.info("Finished %d/%d jobs.", i + 1, len(jobs))

    logger.info("Finished.")


if __name__ == "__main__":
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    min_nofile = 1024 * 8
    if soft < min_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min_nofile, hard))

    tyro.cli(main)
