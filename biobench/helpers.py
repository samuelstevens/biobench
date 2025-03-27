"""
Useful helpers for more than two tasks that don't fit anywhere else.
"""

import collections
import collections.abc
import logging
import os.path
import time

import beartype
import numpy as np
from jaxtyping import Int, jaxtyped


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
