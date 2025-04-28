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
import os.path
import resource
import time

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
def batched_idx(
    total_size: int, batch_size: int
) -> collections.abc.Iterator[tuple[int, int]]:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """
    for start in range(0, total_size, batch_size):
        stop = min(start + batch_size, total_size)
        yield start, stop


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

    x = start
    for m in itertools.cycle((3 / 2, 4 / 3)):  # 3/2, 4/3, 3/2, 4/3, ...
        yield int(x)
        x *= m


@beartype.beartype
def _infer_batch_size(batch: object) -> int | None:
    """
    Return the leading dimension of the *first* tensor found inside `batch`. Works for arbitrary nested structures.
    """
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]

    if isinstance(batch, (list, tuple)):
        for item in batch:
            bs = _infer_batch_size(item)
            if bs is not None:
                return bs

    if isinstance(batch, dict):
        for item in batch.values():
            bs = _infer_batch_size(item)
            if bs is not None:
                return bs

    if dataclasses.is_dataclass(batch):
        return _infer_batch_size(dataclasses.asdict(batch))

    # Fallback: inspect attributes (namedtuple, SimpleNamespace, custom)
    if hasattr(batch, "__dict__"):
        return _infer_batch_size(vars(batch))

    return None


@contextlib.contextmanager
@beartype.beartype
def auto_batch_size(
    dataloader: torch.utils.data.DataLoader,
    *,
    probe: collections.abc.Callable[[torch.Tensor], torch.Tensor],
    schedule: collections.abc.Iterable[int] | None = None,
    upper: int = 4096,
):
    """
    Context manager that **mutates `dataloader.batch_size` in-place** so you always run with the largest batch that fits GPU RAM.

    Parameters
    ----------
    dataloader:
        The *already constructed* loader you use in your loop. Its `batch_sampler.batch_size` attribute is patched on the fly.
    probe:
        A 1-argument callable used to test memory. Typical usage: `lambda x: backbone.img_encode(x).img_features`.
    schedule:
        An iterator of candidate batch sizes.  If `None`, use the canonical schedule.
    """
    logger = logging.getLogger("auto-bsz")

    if dataloader.batch_sampler is None:
        raise ValueError("dataloader must have a batch_sampler")

    orig_bs = int(dataloader.batch_sampler.batch_size)
    tried_bs = orig_bs
    ok_bs = orig_bs
    schedule_iter = schedule or _default_batchsize_schedule(orig_bs)

    torch.cuda.empty_cache()  # be nice

    t_start = time.perf_counter()

    for tried_bs in schedule_iter:
        # honor explicit ceiling
        if upper is not None and tried_bs > upper:
            dataloader.batch_sampler.batch_size = upper
            ok_bs = upper
            logger.debug("Reached user upper limit=%d", upper)
            break

        # quick sanity: do not create impossible 0-batch situations
        if tried_bs <= ok_bs:
            continue

        # patch both public and sampler attrs
        dataloader.batch_sampler.batch_size = tried_bs

        logger.info("Trying batch_size=%d", tried_bs)

        # pull ONE mini-batch, send through probe
        try:
            batch = next(iter(dataloader))
            probe(batch)  # forward only; discard output

            # If the loader produced fewer items than we asked for, we've reached the dataset size — any larger batch will give the same tensor, so stop growing.
            effective_bs = _infer_batch_size(batch)
            if effective_bs is None:
                raise RuntimeError(
                    "Unable to deduce batch size from probe batch; ensure it contains at least one torch.Tensor."
                )
            if effective_bs < tried_bs:
                logger.info(
                    "Dataset exhausted at %d examples (asked for %d); capping batch size.",
                    effective_bs,
                    tried_bs,
                )
                ok_bs = effective_bs
                dataloader.batch_sampler.batch_size = ok_bs
                break

        except RuntimeError as err:
            _msg = str(err).lower()
            oom_signatures = (
                "out of memory",
                "cuda error: invalid configuration argument",  # SPM-efficient-attn OOM
            )
            if any(sig in _msg for sig in oom_signatures):
                logger.info("OOM at batch_size=%d; reverting to %d", tried_bs, ok_bs)
                torch.cuda.empty_cache()
                # restore smaller and stop searching
                dataloader.batch_sampler.batch_size = ok_bs
                break
            raise
        else:
            ok_bs = tried_bs
            logger.info("batch_size=%d succeeded", ok_bs)

    # final guard: ensure we never exceed user-provided upper
    if upper is not None and ok_bs > upper:
        ok_bs = upper
        dataloader.batch_sampler.batch_size = ok_bs

    elapsed = time.perf_counter() - t_start
    logger.info("Selected batch_size %d after %.2f s", ok_bs, elapsed)

    # Final tidy-up to avoid residual OOMs
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # frees cached blocks from other procs
    gc.collect()  # clears Python refs / fragments

    try:
        yield ok_bs  # user code runs here with patched batch_size
    finally:
        # always restore original value
        dataloader.batch_sampler.batch_size = orig_bs
