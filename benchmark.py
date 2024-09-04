import concurrent.futures
import dataclasses
import logging
import multiprocessing
import typing

import beartype
import torch
import tyro

from biobench import interfaces, kabr, models, newt

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("biobench")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    jobs: typing.Literal["slurm", "process", "none"] = "none"
    """what kind of jobs we should use for parallel processing: slurm cluster, multiple processes on the same machine, or just a single process."""

    # How to set up the model.
    model: models.Params = dataclasses.field(default_factory=models.Params)
    """arguments for the vision backbone."""
    device: typing.Literal["cpu", "cuda"] = "cuda"
    """which kind of accelerator to use."""

    # Individual benchmarks.
    newt_run: bool = False
    """whether to run the NeWT benchmark."""
    newt_args: newt.Args = dataclasses.field(default_factory=newt.Args)
    """arguments for the NeWT benchmark."""
    kabr_run: bool = False
    """whether to run the KABR benchmark."""
    kabr_args: kabr.Args = dataclasses.field(default_factory=kabr.Args)
    """arguments for the KABR benchmark."""


@beartype.beartype
def display(report: interfaces.BenchmarkReport) -> None:
    # TODO: probably needs to write results to a machine readable format.
    print(f"{report.name}: {report.score * 100:.1f}%")


class DummyExecutor(concurrent.futures.Executor):
    """Dummy class to satisfy the Executor interface. Directly runs the function in the main process for easy debugging."""

    def submit(self, fn, /, *args, **kwargs):
        """runs `fn` directly in the main process and returns a Future with the result."""
        future = concurrent.futures.Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)

        return future


@beartype.beartype
def main(args: Args):
    if args.jobs == "process":
        executor = concurrent.futures.ProcessPoolExecutor()
    elif args.jobs == "slurm":
        raise NotImplementedError("submitit not tested yet!")
        # TODO
        # executor = submitit.AutoExecutor()
    elif args.jobs == "none":
        executor = DummyExecutor()
    else:
        typing.assert_never(args.jobs)

    # 1. Load model.
    backbone = models.load_model(args.model)

    # 2. Run benchmarks.
    jobs = []
    if args.newt_run:
        newt_args = dataclasses.replace(args.newt_args, device=args.device)
        jobs.append(executor.submit(newt.benchmark, backbone, newt_args))
    if args.kabr_run:
        kabr_args = dataclasses.replace(args.kabr_args, device=args.device)
        jobs.append(executor.submit(kabr.benchmark, backbone, kabr_args))

    # 3. Display results.
    for future in concurrent.futures.as_completed(jobs):
        if future.exception():
            raise RuntimeError("Error running job.") from future.exception()
        report = future.result()
        display(report)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # 0. Check on hardware accelerator.
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA GPU found. Using CPU instead.")
        # Can't use CUDA, so might be on macOS, which cannot use spawn with pickle.
        multiprocessing.set_start_method("fork")
        args.device = "cpu"
    elif args.device == "cuda" and torch.cuda.is_available():
        multiprocessing.set_start_method("spawn")

    main(args)
