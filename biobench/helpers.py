"""
Useful helpers for more than two tasks that don't fit anywhere else.
"""

import collections
import collections.abc
import contextlib
import dataclasses
import gc
import itertools
import logging
import os
import os.path
import pathlib
import resource
import subprocess
import sys
import time
import warnings

import beartype
import numpy as np
import torch
from jaxtyping import Int, jaxtyped


@beartype.beartype
def get_cache_dir() -> str:
    cache_dir = ""
    for var in ("BIOBENCH_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


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

    if not hasattr(clf, "cv_results_"):
        return ""

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


@jaxtyped(typechecker=beartype.beartype)
def balanced_random_sample(
    labels: Int[np.ndarray, " n_labels"], n: int
) -> Int[np.ndarray, " n"]:
    """
    Select n random examples while balancing the number of examples per class.
    """
    # Count the occurrences of each class
    class_counts = collections.Counter(labels)
    unique_classes = list(class_counts.keys())
    n_classes = len(unique_classes)

    if not n_classes:
        return np.array([], dtype=int)

    # Calculate ideal number of samples per class
    samples_per_class = n // n_classes

    # Handle remainder by allocating extra samples to random classes
    remainder = n % n_classes
    extra_samples = np.zeros(n_classes, dtype=int)
    if remainder > 0:
        extra_indices = np.random.choice(n_classes, remainder, replace=False)
        extra_samples[extra_indices] = 1

    # Calculate final samples per class
    final_samples = np.array([samples_per_class] * n_classes) + extra_samples

    # Initialize result array
    selected_indices = []

    # For each class, select random samples
    for i, class_label in enumerate(unique_classes):
        # Get all indices for this class
        class_indices = np.where(labels == class_label)[0]

        # Calculate how many to take (minimum of available samples and desired samples)
        n_to_take = min(len(class_indices), final_samples[i])

        # Randomly sample without replacement
        if n_to_take > 0:
            sampled_indices = np.random.choice(class_indices, n_to_take, replace=False)
            selected_indices.extend(sampled_indices)

    # If we still don't have enough samples (due to some classes having too few examples),
    # sample from the remaining examples across all classes
    if len(selected_indices) < n:
        # Create a mask of already selected indices
        mask = np.ones(len(labels), dtype=bool)
        mask[selected_indices] = False
        remaining_indices = np.where(mask)[0]

        # How many more do we need?
        needed = n - len(selected_indices)

        # Sample without replacement from remaining indices
        if needed > 0 and len(remaining_indices) > 0:
            additional_indices = np.random.choice(
                remaining_indices, min(needed, len(remaining_indices)), replace=False
            )
            selected_indices.extend(additional_indices)

    return np.array(selected_indices, dtype=int)


@beartype.beartype
class batched_idx:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """

    def __init__(self, total_size: int, batch_size: int):
        """
        Args:
            total_size: total number of examples
            batch_size: maximum distance between the generated indices
        """
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        """Yield (start, end) index pairs for batching."""
        for start in range(0, self.total_size, self.batch_size):
            stop = min(start + self.batch_size, self.total_size)
            yield start, stop

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.total_size + self.batch_size - 1) // self.batch_size


@beartype.beartype
def bump_nofile(margin: int = 512) -> None:
    """
    Make RLIMIT_NOFILE.soft = RLIMIT_NOFILE.hard - margin (if that is higher than the current soft limit).  No change if margin would push soft < 1. Raises RuntimeError if hard <= margin.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    if hard <= margin:
        raise RuntimeError(
            f"hard limit ({hard}) is <= margin ({margin}); ask an admin to raise the hard limit."
        )

    target_soft = hard - margin
    if soft < target_soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))


@beartype.beartype
def _default_batchsize_schedule(start: int = 2) -> collections.abc.Iterable[int]:
    """
    2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 196, 256, 384, 512, 768, 1024, ...
    """

    while start < 2:
        yield start
        start += 1

    x = start
    for m in itertools.cycle((3 / 2, 4 / 3)):  # 3/2, 4/3, 3/2, 4/3, ...
        yield int(x)
        x *= m


@beartype.beartype
def infer_batch_size(batch: object) -> int | None:
    """
    Return the leading dimension of the *first* tensor found inside `batch`. Works for arbitrary nested structures.
    """
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]

    if isinstance(batch, (list, tuple)):
        for item in batch:
            bs = infer_batch_size(item)
            if bs is not None:
                return bs

    if isinstance(batch, dict):
        for item in batch.values():
            bs = infer_batch_size(item)
            if bs is not None:
                return bs

    if dataclasses.is_dataclass(batch):
        return infer_batch_size(dataclasses.asdict(batch))

    # Fallback: inspect attributes (namedtuple, SimpleNamespace, custom)
    if hasattr(batch, "__dict__"):
        return infer_batch_size(vars(batch))

    return None


@contextlib.contextmanager
@beartype.beartype
def auto_batch_size(
    dataloader: torch.utils.data.DataLoader,
    *,
    probe: collections.abc.Callable[[torch.Tensor], torch.Tensor],
    schedule: collections.abc.Iterable[int] | None = None,
    upper: int = 4096,
    backoff: int = 0,
):
    """Context manager that mutates `dataloader.batch_size` in-place to use the largest batch that fits GPU RAM.

    This function tests progressively larger batch sizes until it finds the maximum that can be processed without running out of memory.

    Args:
        dataloader: The already constructed loader you use in your loop. Its `batch_sampler.batch_size` attribute is patched on the fly.
        probe: A 1-argument callable used to test memory. Typical usage: `lambda x: backbone.img_encode(x).img_features`.
        schedule: An iterator of candidate batch sizes. If None, use the canonical schedule.
        schedule: An iterator of strictly increasing candidate batch sizes (2, 4, 8, ...). A *ValueError* is raised when a non-increasing value is encountered. If None, use the canonical schedule.
        upper: Maximum batch size to try, regardless of available memory.
        backoff: int, default = 0. How far to step **back** in the candidate schedule from the largest batch-size that completes without OOM  (clamped to the smallest candidate if ``n`` is too big).
        * `backoff = 0`  -> use the **largest** successful size
        * `backoff = 1`  -> use the **second-largest** successful size
        * `backoff = n`  -> use the *n*-th size below the largest success

    Yields:
        int: The selected batch size.
    """

    @beartype.beartype
    class OrderedSet[T]:
        def __init__(self):
            self._lst = []

        def __bool__(self) -> bool:
            return bool(self._lst)

        @property
        def last(self) -> T:
            return self._lst[-1]

        def append(self, t: T) -> None:
            if t in self._lst:
                return

            self._lst.append(t)

        def pop(self) -> T:
            return self._lst.pop()

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self._items!r})"

        def __str__(self) -> str:
            return str(self._items)

    if dataloader.batch_sampler is None:
        raise ValueError("dataloader must have a batch_sampler")

    if backoff < 0:
        raise ValueError(f"backoff '{backoff}' < 0; must be >= 0")

    logger = logging.getLogger("auto-bsz")

    oom_signatures = (
        "out of memory",
        "cuda error: invalid configuration argument",  # SPM-efficient-attn OOM
        "expected canuse32bitindexmath(input) && canuse32bitindexmath(output) to be true, but got false.",  # Conv layers with big batch sizes.
    )

    dataloader.batch_sampler.batch_size = min(
        dataloader.batch_sampler.batch_size, upper
    )

    orig_bsz = int(dataloader.batch_sampler.batch_size)
    good_bszs = OrderedSet()
    schedule_iter = schedule or _default_batchsize_schedule(orig_bsz)

    torch.cuda.empty_cache()  # be nice

    t_start = time.perf_counter()

    for tried_bsz in schedule_iter:
        if good_bszs and tried_bsz <= good_bszs.last:
            raise ValueError(
                f"Schedule not monotonically increasing: {tried_bsz} <= {good_bszs.last}"
            )

        if tried_bsz > upper:
            tried_bsz = upper

        # patch sampler attr
        dataloader.batch_sampler.batch_size = tried_bsz
        logger.info("Trying batch_size=%d", tried_bsz)

        # pull ONE mini-batch, send through probe
        try:
            batch = next(iter(dataloader))
            probe(batch)  # forward only; discard output

            # If the loader produced fewer items than we asked for, we've reached the dataset size -- any larger batch will give the same tensor, so stop growing.
            effective_bsz = infer_batch_size(batch)
            if effective_bsz is None:
                raise RuntimeError(
                    "Unable to deduce batch size from probe batch; ensure it contains at least one torch.Tensor."
                )

            if effective_bsz < tried_bsz:
                logger.info(
                    "Dataset exhausted at %d examples (asked for %d); capping batch size.",
                    effective_bsz,
                    tried_bsz,
                )
                ok_bsz = effective_bsz
                good_bszs.append(ok_bsz)
                break

            ok_bsz = tried_bsz
            good_bszs.append(ok_bsz)
            logger.info("batch_size=%d succeeded", ok_bsz)

            # honor explicit ceiling
            if upper is not None and ok_bsz >= upper:
                logger.info("Reached ok_bsz (%d) >= upper (%d)", ok_bsz, upper)
                ok_bsz = upper
                break

        except RuntimeError as err:
            msg = str(err).lower()
            if any(sig in msg for sig in oom_signatures):
                logger.info("OOM at batch_size=%d; reverting to %d", tried_bsz, ok_bsz)
                torch.cuda.empty_cache()
                break
            else:
                raise

    # (re-)verify ok_bs once more in a clean context
    while good_bszs:
        ok_bsz = good_bszs.pop()
        dataloader.batch_sampler.batch_size = ok_bsz
        try:
            batch = next(iter(dataloader))
            probe(batch)
            break  # we know ok_bsz is actually good.

        except RuntimeError as err:
            if any(sig in str(err).lower() for sig in oom_signatures):
                logger.info("Still OOM at %d; trying previous candidate", ok_bsz)
            else:
                raise

    while good_bszs and backoff:
        backoff -= 1
        ok_bsz = good_bszs.pop()
        dataloader.batch_sampler.batch_size = ok_bsz

    elapsed = time.perf_counter() - t_start
    logger.info("Selected batch_size %d after %.2f s", ok_bsz, elapsed)

    # Final tidy-up to avoid residual OOMs
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # frees cached blocks from other procs
    gc.collect()  # clears Python refs / fragments

    try:
        yield ok_bsz  # user code runs here with patched batch_size
    finally:
        # always restore original value
        dataloader.batch_sampler.batch_size = orig_bsz


NFS_TYPES = {"nfs", "nfs4", "nfsd", "autofs"}  # extend if you wish


@beartype.beartype
def warn_if_nfs(path: str | os.PathLike):
    """
    If *path* is on an NFS mount, emit a RuntimeWarning.

    Works on Linux (/proc/mounts) and macOS/BSD (`mount` CLI); silently returns on other OSes or if detection fails.
    """
    p = pathlib.Path(path).resolve()

    # Linux: /proc/self/mountinfo
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/mountinfo") as fd:
                entries = [line.split() for line in fd]
            # fields: 4= mount point, - separator -, fstype
            mounts = {fields[4]: fields[-3] for fields in entries}
        except Exception:
            return

    # macOS / BSD: `mount`
    elif sys.platform in {"darwin", "freebsd"}:
        try:
            out = subprocess.check_output(["mount", "-p"], text=True)
            mounts = {}
            for line in out.splitlines():
                mp, _dev, fstype, *_ = line.split()  # mount-point, ...
                mounts[mp] = fstype
        except Exception:
            return
    else:
        return  # unsupported OS

    # find longest mount-point prefix of *p*
    mount_point = max(
        (mp for mp in mounts if p.is_relative_to(mp) or mp == "/"),
        key=len,
        default="/",
    )
    if mounts.get(mount_point) in NFS_TYPES:
        warnings.warn(
            f"SQLite database '{path}' appears to be on an NFS mount (fs type: {mounts[mount_point]}). Concurrent writers over NFS can corrupt the journal; consider using a local SSD or tmpfs instead.",
            RuntimeWarning,
            stacklevel=2,
        )
