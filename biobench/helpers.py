"""
Useful helpers for more than two tasks that don't fit anywhere else.
"""

import collections.abc
import io
import logging
import os.path
import time
import typing

import beartype
import pybase64
import submitit
from PIL import Image


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress"):
        """
        Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
        """
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)

    def __iter__(self):
        start = time.time()
        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if isinstance(self.it, collections.abc.Sized):
                    pred_min = (len(self) - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        len(self),
                        (i + 1) / len(self) * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        return len(self.it)


@beartype.beartype
def fs_safe(string: str) -> str:
    """Makes a string safe for filesystems by removing typical special characters."""
    return string.replace(":", "_").replace("/", "_")


@beartype.beartype
def write_hparam_sweep_plot(
    task: str,
    model: str,
    clf,
    x: str = "param_ridgeclassifier__alpha",
    y: str = "mean_test_score",
) -> str:
    import matplotlib.pyplot as plt
    import polars as pl

    df = pl.DataFrame(clf.cv_results_)

    fig, ax = plt.subplots()

    if "n_resources" in df.columns:
        for n_resources in df.get_column("n_resources").unique().sort():
            ax.scatter(
                x=df.filter(pl.col("n_resources") == n_resources)[x],
                y=df.filter(pl.col("n_resources") == n_resources)[y],
                label=f"{n_resources} ex.",
            )
        fig.legend()
    else:
        ax.scatter(x=df[x], y=df[y])

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xscale("log")
    ax.set_title(model)

    fig.tight_layout()
    filepath = os.path.join("logs", f"{task}_{fs_safe(model)}_hparam.png")
    fig.savefig(filepath)
    return filepath


@beartype.beartype
def load_img_b64(path: str) -> str:
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format="webp")
    b64 = pybase64.b64encode(buf.getvalue())
    s64 = b64.decode("utf8")
    return "data:image/webp;base64," + s64


R = typing.TypeVar("R", covariant=True)


def as_completed(
    jobs: typing.Sequence[submitit.Job[R]],
    timeout: int | float | None = None,
    poll_frequency: float = 10,
) -> typing.Iterator[submitit.Job[R]]:
    """
    Yields jobs as they complete (finished, failed or were cancelled).
    Raises a TimeoutError if the result isn’t available after timeout seconds.
    timeout can be an int or float. If timeout is not specified or None, there is no
    limit to the wait time.

    Parameters
    ----------
    jobs: list
        Jobs instances

    timeout: int/float
        Maximum time (in sec) to wait for jobs completion

    poll_frequency: float
        Frequency in second at which we check job status.

    Yields
    ------
    Job
        The next completed job
    """
    start = time.time()
    jobs_done: set[int] = set()
    while True:
        if timeout is not None and time.time() - start > timeout:
            raise TimeoutError
        for i, job in enumerate(jobs):
            if i in jobs_done:
                continue
            if job.done():
                jobs_done.add(i)
                yield job
        if len(jobs_done) == len(jobs):
            break
        time.sleep(poll_frequency)
