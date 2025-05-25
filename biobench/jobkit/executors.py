import logging
import os
import pathlib
import typing

import beartype
import submitit


@beartype.beartype
class SerialJob(submitit.DebugJob):
    """DebugJob without the pdb step."""

    def results(self) -> list[object]:
        self._check_not_cancelled()
        if self._submission.done():
            return [self._submission._result]

        environ_backup = dict(os.environ)
        # Restore os.environ from job creation time.
        os.environ.clear()
        os.environ.update(self.environ)

        root_logger = logging.getLogger("")
        self.paths.stdout.parent.mkdir(exist_ok=True, parents=True)
        stdout_handler = logging.FileHandler(self.paths.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stderr_handler = logging.FileHandler(self.paths.stderr)
        stderr_handler.setLevel(logging.WARNING)
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)
        root_logger.warning(
            f"Logging is written both to stderr/stdout and to {self.paths.stdout}/err. "
            "But call to print will only appear in the console."
        )
        try:
            return [self._submission.result()]
        except Exception as e:
            print(e)
            raise
        finally:
            os.environ.clear()
            os.environ.update(environ_backup)
            root_logger.removeHandler(stdout_handler)
            root_logger.removeHandler(stderr_handler)


@beartype.beartype
class SerialExecutor(submitit.Executor):
    """
    Execute submitit jobs **sequentially in-process** with no interactive debugger.

    * One Python process, one GPU/CPU context: the function is called directly in the parent interpreter--exactly like DebugExecutor--but any un-caught exception is *immediately* re-raised instead of dropping into ``pdb.post_mortem``.

    * This lets a launcher loop handle failures programmatically (log, skip, retry, etc.) while preserving DebugExecutor's simple bookkeeping, stdout/stderr redirection, and environment capture.

    * Contrast:
        DebugExecutor  - in-process, blocks in pdb on error.
        LocalExecutor  - spawns a child process per job, may run jobs concurrently depending on submitit version.
        SerialExecutor - in-process, **no pdb**, always exactly one job running.

    Usage
    -----
    >>> ex = SerialExecutor("logs")
    >>> job = ex.submit(train_one, cfg)
    >>> try:
    ...     result = job.result()   # raises on failure, no (pdb) prompt
    ... except Exception as err:
    ...     handle(err, job)

    Notes
    -----
    * Inherits all parameter handling from DebugExecutor; only ``job_class`` is swapped to remove the post-mortem call.
    * Suitable for deterministic, single-GPU jobs where you want clean failure handling without interactive prompts.
    """

    job_class = SerialJob

    def __init__(self, folder: str | pathlib.Path):
        super().__init__(folder)

    def _internal_process_submissions(
        self, delayed_submissions: list[submitit.core.utils.DelayedSubmission]
    ) -> list[submitit.core.core.Job[typing.Any]]:
        return [self.job_class(self.folder, ds) for ds in delayed_submissions]
