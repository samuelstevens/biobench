import dataclasses
import numpy as np
import socket
import subprocess
import sys
import time

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class EncodedImgBatch:
    """The output of a `VisionBackbone`'s `img_encode()` method."""

    img_features: Float[Tensor, "batch img_dim"]
    """Image-level features. Each image is represented by a single vector."""
    patch_features: Float[Tensor, "batch n_patches patch_dim"] | None
    """Patch-level features. Only ViTs have patch-level features. These features might be a different dimension that the image features because of projection heads or such."""


@jaxtyped(typechecker=beartype.beartype)
class VisionBackbone(torch.nn.Module):
    """ """

    @jaxtyped(typechecker=beartype.beartype)
    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> EncodedImgBatch:
        err_msg = f"{self.__class__.__name__} must implemented img_encode()."
        raise NotImplementedError(err_msg)

    def make_img_transform(self):
        err_msg = f"{self.__class__.__name__} must implemented make_img_transform()."
        raise NotImplementedError(err_msg)


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class BenchmarkReport:
    """
    The result of running a benchmark.

    TODO: this needs to store more than just a summary statistic (`score`). It should include many raw results that can be used for analysis later on. It can even reference invidividual examples in a dataset so that they can be viewed.

    This should probably be in the form of

    summary: float
    splits: dict[str, float]
    examples: list[tuple[object, float, dict[str, object]]]

    See notebooks/tutorial.py for details.
    """

    # Actual details of the report
    name: str
    """the benchmark name."""
    examples: list[tuple[str, float, dict[str, object]]]
    """a list of (example_id, score, info) tuples"""
    splits: dict[str, float]
    """individual splits and scores; can be anything you want."""

    # Stuff for trying to reproduce this result. These are filled in by default.
    argv: list[str] = dataclasses.field(default_factory=lambda: sys.argv)
    """command used to get this report."""
    commit: str = get_git_hash()
    """Git commit for this current report."""
    posix_time: float = dataclasses.field(default_factory=time.time)
    """time when this report was constructed."""
    gpu_name: str = dataclasses.field(
        default_factory=lambda: torch.cuda.get_device_properties(0).name
    )
    hostname: str = dataclasses.field(default_factory=socket.gethostname)

    def __repr__(self):
        return f"Report({self.name} with {len(self.examples)} examples)"

    def __str__(self):
        return repr(self)

    @property
    def mean_score(self) -> float:
        """mean score across the entire task, as a number between 0 and 1."""
        return float(np.mean([score for _, score, _ in self.examples]))

    def get_confidence_interval(
        self,
        statistic="mean",
        confidence: float = 95,
        n_resamples: int = 500,
        seed: int = 42,
    ) -> tuple[float, float]:
        """confidence interval for the statistics (mean) by bootstrapping individual scores of the examples."""
        scores = np.array([score for _, score, _ in self.examples])

        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(len(scores), size=(n_resamples, len(scores)), replace=True)
        means = np.mean(scores[idx], axis=0)

        percentiles = (100 - confidence) / 2, (100 - confidence) / 2 + confidence
        lower, upper = np.percentile(means, percentiles).tolist()

        return lower, upper
