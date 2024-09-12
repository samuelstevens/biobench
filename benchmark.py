"""
Entrypoint for running all benchmarks.

.. include:: ./tutorial.md
"""

import concurrent.futures
import dataclasses
import json
import logging
import os
import resource
import time
import typing

import beartype
import torch
import tyro

from biobench import ModelOrg, interfaces, iwildcam, kabr, newt, plantnet

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger("biobench")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Params to run one or more benchmarks in a parallel setting."""

    slurm: bool = False
    """whether to use submitit to run jobs on a slurm cluster."""

    model_args: typing.Annotated[
        list[tuple[ModelOrg, str]], tyro.conf.arg(name="model")
    ] = dataclasses.field(
        default_factory=lambda: [
            ("open-clip", "RN50/openai"),
            ("open-clip", "ViT-B-16/openai"),
            ("open-clip", "hf-hub:imageomics/bioclip"),
            ("timm-vit", "vit_base_patch14_reg4_dinov2.lvd142m"),
        ]
    )
    """model; a pair of model org (interface) and checkpoint."""
    device: typing.Literal["cpu", "cuda"] = "cuda"
    """which kind of accelerator to use."""
    debug: bool = False
    """whether to run in debug mode."""

    # Individual benchmarks.
    newt_run: bool = True
    """whether to run the NeWT benchmark."""
    newt_args: newt.Args = dataclasses.field(default_factory=newt.Args)
    """arguments for the NeWT benchmark."""
    kabr_run: bool = True
    """whether to run the KABR benchmark."""
    kabr_args: kabr.Args = dataclasses.field(default_factory=kabr.Args)
    """arguments for the KABR benchmark."""
    plantnet_run: bool = True
    """whether to run the Pl@ntNet benchmark."""
    plantnet_args: plantnet.Args = dataclasses.field(default_factory=plantnet.Args)
    """arguments for the Pl@ntNet benchmark."""
    iwildcam_run: bool = True
    """whether to run the iWildCam benchmark."""
    iwildcam_args: iwildcam.Args = dataclasses.field(default_factory=iwildcam.Args)
    """arguments for the iWildCam benchmark."""
    # Saving
    report_to: str = os.path.join(".", "reports")
    """where to save reports to."""

    def report_path(self, report: interfaces.TaskReport) -> str:
        posix = int(time.time())
        return os.path.join(args.report_to, f"{posix}.jsonl")

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class DummyExecutor(concurrent.futures.Executor):
    """Dummy class to satisfy the Executor interface. Directly runs the function in the main process for easy debugging."""

    def submit(self, fn, /, *args, **kwargs):
        """runs `fn` directly in the main process and returns a `concurrent.futures.Future` with the result.

        Returns:
        """
        future = concurrent.futures.Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)

        return future


@beartype.beartype
def save(
    args: Args, model_args: interfaces.ModelArgs, report: interfaces.TaskReport
) -> None:
    """
    Saves the report to disk in a machine-readable JSON format.
    """
    dct = {}
    dct.update(**{f"report_{key}": value for key, value in report.to_dict().items()})
    model_org, model_ckpt = model_args
    dct["report_model_org"] = model_org
    dct["report_model_ckpt"] = model_ckpt
    dct["report_mean_score"] = report.get_mean_score()
    lower, upper = report.get_confidence_interval()
    dct["report_confidence_interval_lower"] = lower
    dct["report_confidence_interval_upper"] = upper
    # Add benchmark.py args
    dct.update(**{
        f"benchmark_{key}": value for key, value in dataclasses.asdict(args).items()
    })

    with open(args.report_path(report), "a") as fd:
        fd.write(json.dumps(dct) + "\n")

    logger.info(
        "%s on %s: %.1f%%", model_ckpt, report.name, report.get_mean_score() * 100
    )
    for key, value in report.splits.items():
        logger.info("%s on %s; split '%s': %.3f", model_ckpt, report.name, key, value)


@beartype.beartype
def main(args: Args):
    # 1. Setup executor.
    pool_cls, pool_args, pool_kwargs = DummyExecutor, (), {}
    if args.slurm:
        raise NotImplementedError("submitit not implemented.")
        # TODO: implement submitit
        # executor = submitit.AutoExecutor()

    # 2. Run benchmarks.
    try:
        executor = pool_cls(*pool_args, **pool_kwargs)
        jobs = []
        for model_args in args.model_args:
            if args.newt_run:
                newt_args = dataclasses.replace(
                    args.newt_args, device=args.device, debug=args.debug
                )
                jobs.append(executor.submit(newt.benchmark, newt_args, model_args))
            if args.kabr_run:
                kabr_args = dataclasses.replace(
                    args.kabr_args, device=args.device, debug=args.debug
                )
                jobs.append(executor.submit(kabr.benchmark, kabr_args, model_args))
            if args.plantnet_run:
                plantnet_args = dataclasses.replace(
                    args.plantnet_args, device=args.device, debug=args.debug
                )
                job = executor.submit(plantnet.benchmark, plantnet_args, model_args)
                jobs.append(job)
            if args.iwildcam_run:
                iwildcam_args = dataclasses.replace(
                    args.iwildcam_args, device=args.device, debug=args.debug
                )
                job = executor.submit(iwildcam.benchmark, iwildcam_args, model_args)
                jobs.append(job)

        logger.info("Submitted %d jobs.", len(jobs))

        # 3. Display results.
        os.makedirs(args.report_to, exist_ok=True)
        for i, future in enumerate(concurrent.futures.as_completed(jobs)):
            err = future.exception()
            if err:
                logger.warning("Error running job: %s: %s", err, err.__cause__)
                continue

            model_args, report = future.result()
            save(args, model_args, report)
            logger.info("Finished %d/%d jobs.", i + 1, len(jobs))
    finally:
        logger.info("Shutting down job executor.")
        executor.shutdown(cancel_futures=True, wait=True)
    logger.info("Finished.")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # 0. Check on hardware accelerator.
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA GPU found. Using CPU instead.")
        # Can't use CUDA, so might be on macOS, which cannot use spawn with pickle.
        torch.multiprocessing.set_start_method("fork")
        args = dataclasses.replace(args, device="cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        torch.multiprocessing.set_start_method("spawn")

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    min_nofile = 1024 * 8
    if soft < min_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min_nofile, hard))

    main(args)
