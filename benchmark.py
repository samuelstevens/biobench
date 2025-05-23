"""
Entrypoint for running all tasks in `biobench`.

Most of this script is self documenting.
Run `python benchmark.py --help` to see all the options.

Note that you will have to download all the datasets, but each dataset includes its own download script with instructions.
For example, see `biobench.newt.download` for an example.
"""

import collections
import importlib
import logging
import os

import beartype
import submitit
import tyro

from biobench import config, helpers, jobkit, reporting


@beartype.beartype
def main(cfgs: list[str], dry_run: bool = True, max_pending: int = 8):
    """
    Launch all jobs, using either a local GPU or a Slurm cluster. Then report results and save to disk.

    Args:
        cfgs: List of paths to TOML config files.
        dry_run: If --no-dry-run, actually run experiment.
        max_pending: Number of jobs that can be claimed by any one launcher process.
    """

    # Load all configs from the provided paths.
    cfgs = [cfg for path in cfgs for cfg in config.load(path)]

    if not cfgs:
        print("No configurations loaded.")
        return

    # ------------------------------------------------------
    # Verify all configs have consistent execution settings.
    # ------------------------------------------------------
    first = cfgs[0]
    for cfg in cfgs[1:]:
        if cfg.slurm_acct != first.slurm_acct:
            raise ValueError("All configs must have the same slurm_acct")
        if cfg.log_to != first.log_to:
            raise ValueError("All configs must have the same log_to directory")
        if cfg.ssl != first.ssl:
            raise ValueError("All configs must have the same ssl setting")

    # --------------
    # Setup logging.
    # --------------
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("benchmark.py")

    # ---------------
    # Setup executor.
    # ---------------
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
    elif first.debug:
        executor = submitit.DebugExecutor(folder=first.log_to)
        # See biobench.third_party_models.get_ssl() for a discussion of this variable.
        if not first.ssl:
            os.environ["BIOBENCH_DISABLE_SSL"] = "1"
    else:
        executor = jobkit.SerialExecutor(folder=first.log_to)
        # See biobench.third_party_models.get_ssl() for a discussion of this variable.
        if not first.ssl:
            os.environ["BIOBENCH_DISABLE_SSL"] = "1"

    db = reporting.get_db(first)

    # Clear old (5 days+) runs.
    cleared = reporting.clear_stale_claims(db, max_age_hours=24 * 5)
    logger.info("Cleared %d stale jobs from 'runs' table.", cleared)

    job_stats = collections.defaultdict(int)
    model_stats = collections.defaultdict(int)
    fq = jobkit.FutureQueue(max_size=max_pending)
    exit_hook = jobkit.ExitHook(
        lambda args: reporting.release_run(db, *args)
    ).register()

    def flush_one():
        """
        Get the next finished job from queue, blocking if necessary, write the report and relinquish the claim.
        """
        job, cfg, task = fq.pop()
        try:
            report: reporting.Report = job.result()
            report.write()
            logger.info("%s+%s/%s done", task, cfg.model.org, cfg.model.ckpt)
        except Exception as err:
            logger.info("%s+%s/%s failed: %s", task, cfg.model.org, cfg.model.ckpt, err)
        finally:
            exit_hook.discard((cfg, task))

    for cfg in cfgs:
        for task, data_root in cfg.data.to_dict().items():
            reason = get_skip_reason(db, cfg, task, data_root, dry_run)
            if reason:
                job_stats[reason] += 1
                continue

            if dry_run:
                job_stats["todo"] += 1
                model_stats[cfg.model.ckpt] += 1
                continue  # no side-effect

            if not reporting.claim_run(db, cfg, task):
                job_stats["queued"] += 1  # someone else just grabbed it
                continue

            exit_hook.add((cfg, task))  # for signal/atexit handler
            job = executor.submit(worker, task, cfg)
            fq.submit((job, cfg, task))
            job_stats["submitted"] += 1

            while fq.full():
                flush_one()

    if dry_run:
        logger.info("Job Summary:")
        logger.info("%-20s | %-5s", "Reason", "Count")
        logger.info("-" * 31)
        for reason, count in sorted(job_stats.items()):
            logger.info("%-20s | %5d", reason, count)
        logger.info("-" * 31)

        logger.info("Model Summary:")
        logger.info("%-50s | %-5s", "Model", "Count")
        logger.info("-" * 61)
        for model, count in sorted(model_stats.items()):
            logger.info("%-50s | %5d", model, count)
        logger.info("-" * 61)
        return

    while fq:
        flush_one()

    logger.info("Finished.")


@beartype.beartype
def worker(task_name: str, cfg: config.Experiment) -> reporting.Report:
    helpers.bump_nofile(512)

    module = importlib.import_module(f"biobench.{task_name}")
    return module.benchmark(cfg)


@beartype.beartype
def get_skip_reason(
    db, cfg: config.Experiment, task: str, data_root: str, dry_run: bool
) -> str | None:
    """Return a short reason string if we should skip (None -> keep)."""
    try:
        importlib.import_module(f"biobench.{task}")
    except ModuleNotFoundError:
        return "no code"

    if not data_root:
        return "no data"

    if reporting.already_ran(db, cfg, task):
        return "done"

    if reporting.is_claimed(db, cfg, task):
        return "queued"

    return None


if __name__ == "__main__":
    tyro.cli(main)
