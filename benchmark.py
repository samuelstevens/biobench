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

import beartype
import submitit
import tyro

from biobench import config, helpers, reporting


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
    # Load all configs from the provided paths and concatenate them
    cfgs = [cfg for path in cfgs for cfg in config.load(path)]

    if not cfgs:
        print("No configurations loaded.")
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

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.verbose else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("benchmark.py")

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
    job_queue = reporting.JobQueue(10)  # Set a reasonable max_size
    counts = collections.defaultdict(int)
    for cfg in helpers.progress(cfgs, desc="submitting jobs"):
        for task_name, data_root in cfg.data.to_dict().items():
            # Check that you can get the task_name
            try:
                importlib.import_module(f"biobench.{task_name}")
            except ModuleNotFoundError:
                counts["no_code"] += 1
                logger.warning("Could not find task '%s'.", task_name)
                continue

            if not data_root:
                counts["no_data"] += 1
                continue

            if reporting.already_ran(db, cfg, task_name):
                counts["done"] += 1
                continue

            if dry_run:
                if reporting.is_claimed(db, cfg, task_name):
                    counts["queued"] += 1
                    continue  # would be skipped
            else:
                if not reporting.claim_run(db, cfg, task_name):
                    counts["queued"] += 1
                    continue

            if dry_run:
                counts["pending"] += 1
                job_queue.submit((None, cfg, task_name))
                continue

            job = executor.submit(worker, task_name, cfg)
            job_queue.submit((job, cfg, task_name))
            counts["pending"] += 1

            # throttle
            while job_queue.full():
                job, cfg, task_name = job_queue.pop()
                try:
                    report: reporting.Report = job.result()
                    report.write()
                    logger.info(
                        "%s+%s/%s done",
                        report.task_name,
                        report.cfg.model.org,
                        report.cfg.model.ckpt,
                    )
                except Exception as err:
                    logger.warning("Failed: %s", err)
                finally:
                    reporting.release_run(db, cfg, task_name)

    if dry_run:
        # Summarize the jobs by model and training examples
        model_counts = collections.defaultdict(int)
        # Convert job_queue to a list to iterate over it
        jobs_list = []
        while job_queue:
            jobs_list.append(job_queue.pop())
        
        for _, job_cfg, _ in jobs_list:
            key = (job_cfg.model.ckpt, job_cfg.n_train)
            model_counts[key] += 1

        # Check if there are any jobs to run
        if not model_counts:
            logger.info("All jobs have already been completed. Nothing to run.")
            return

        # Print summary table
        logger.info("Job Summary:")
        logger.info("%-50s | %-10s | %-5s", "Model", "Train Size", "Count")
        logger.info("-" * 71)
        for (model, n_train), count in sorted(model_counts.items()):
            logger.info("%-50s | %10d | %5d", model, n_train, count)
        logger.info("-" * 71)
        logger.info(
            "Total jobs to run: %d (skipped %d already completed)",
            len(jobs_list),
            counts["done"] + counts["queued"],
        )
        return

    logger.info("Submitted %d jobs (skipped %d).", counts["pending"], counts["done"] + counts["queued"])

    # 3. Write results to sqlite.
    while job_queue:
        job, cfg, task_name = job_queue.pop()
        try:
            report: reporting.Report = job.result()
            report.write()
            logger.info(
                "%s+%s/%s done",
                report.task_name,
                report.cfg.model.org,
                report.cfg.model.ckpt,
            )
        except Exception as err:
            logger.warning("Failed: %s", err)
        finally:
            reporting.release_run(db, cfg, task_name)

    logger.info("Finished.")


@beartype.beartype
def worker(task_name: str, cfg: config.Experiment) -> reporting.Report:
    helpers.bump_nofile(512)

    module = importlib.import_module(f"biobench.{task_name}")
    return module.benchmark(cfg)


if __name__ == "__main__":
    tyro.cli(main)
