"""
Common interfaces for models and tasks so that it's easy to add new models (which will work right away with all tasks) and easy to add new tasks (which will work right away with all models).

The model interface is `VisionBackbone`.
See `biobench.third_party_models` for examples of how to subclass it, and note that you have to call `biobench.register_vision_backbone` for it to show up.

The benchmark interface is informal, but is a function that matches the following signature:

```py
def benchmark(args: Args, model_args: tuple[str, str]) -> tuple[tuple[str, str], interfaces.TaskReport]:
    ...
```

In a Haskell-like signature, this is more like `Args -> (str, str) -> ((str, str), TaskReport)`.

"""

import dataclasses
import socket
import subprocess
import sys
import time
import typing

import beartype
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class TaskArgs:
    """Common args for all tasks."""

    seed: int = 42
    """random seed."""
    datadir: str = ""
    """dataset directory; where you downloaded this task's data to."""
    # Computed at runtime.
    device: str = "cuda"
    """(computed at runtime) which kind of accelerator to use."""
    debug: bool = False
    """(computed at runtime) whether to run in debug mode."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class EncodedImgBatch:
    """The output of a `VisionBackbone`'s `VisionBackbone.img_encode()` method."""

    img_features: Float[Tensor, "batch img_dim"]
    """Image-level features. Each image is represented by a single vector."""
    patch_features: Float[Tensor, "batch n_patches patch_dim"] | None
    """Patch-level features. Only ViTs have patch-level features. These features might be a different dimension that the image features because of projection heads or such."""


@jaxtyped(typechecker=beartype.beartype)
class VisionBackbone(torch.nn.Module):
    """
    A frozen vision model that embeds batches of images into batches of vectors.

    To add new models to the benchmark, you can simply create a new class that satisfies this interface and register it.
    See the tutorial on adding SAM to biobench.
    """

    @jaxtyped(typechecker=beartype.beartype)
    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> EncodedImgBatch:
        """Encode a batch of images."""
        err_msg = f"{self.__class__.__name__} must implemented img_encode()."
        raise NotImplementedError(err_msg)

    def make_img_transform(self):
        """
        Return whatever function the backbone wants for image preprocessing.
        This should be an evaluation transform, not a training transform, because we are using the output features of this backbone as data and not updating this backbone.
        """
        err_msg = f"{self.__class__.__name__} must implemented make_img_transform()."
        raise NotImplementedError(err_msg)


def get_git_hash() -> str:
    """
    Returns the hash of the current git commit, assuming we are in a git repo.
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Example:
    """An individual test example."""

    id: str
    """Whatever kind of ID; used to find the original image/example."""
    score: float
    """Test score; typically 0 or 1 for classification tasks."""
    info: dict[str, object]
    """Any additional information included. This might be the original class, the true label, etc."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class TaskReport:
    """
    The result of running a benchmark task.

    See notebooks/tutorial.py for details.
    """

    # Actual details of the report
    name: str
    """The benchmark name."""
    examples: list[Example]
    """A list of (example_id, score, info) objects"""
    calc_mean_score: typing.Callable[[list[Example]], float]
    """A way to calculate mean score from a list of examples."""

    # Stuff for trying to reproduce this result. These are filled in by default.
    argv: list[str] = dataclasses.field(default_factory=lambda: sys.argv)
    """Command used to get this report."""
    commit: str = get_git_hash()
    """Git commit for this current report."""
    posix_time: float = dataclasses.field(default_factory=time.time)
    """Time when this report was constructed."""
    gpu_name: str = dataclasses.field(
        default_factory=lambda: torch.cuda.get_device_properties(0).name
    )
    """Name of the GPU that ran this experiment."""
    hostname: str = dataclasses.field(default_factory=socket.gethostname)
    """Machine hostname that ran this experiment."""

    def __repr__(self):
        return f"Report({self.name} with {len(self.examples)} examples)"

    def __str__(self):
        return repr(self)

    def get_mean_score(self) -> float:
        """
        Get the mean score of all examples.
        """
        return self.calc_mean_score(self.examples)

    def get_confidence_interval(
        self,
        statistic="mean",
        confidence: float = 95,
        n_resamples: int = 500,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Get the confidence interval for the statistics (mean) by bootstrapping individual scores of the examples.

        NOTE: it's crazy how much easier this would be in Jax. PyTrees of Examples would simply contains batch dimensions, and then I would `jax.vmap(get_mean_score)(batched_examples)`.
        """

        rng = np.random.default_rng(seed=seed)
        choices = rng.choice(
            len(self.examples), size=(n_resamples, len(self.examples)), replace=True
        )

        scores = []
        for choice in choices:
            scores.append(self.calc_mean_score([self.examples[i] for i in choice]))

        percentiles = (100 - confidence) / 2, (100 - confidence) / 2 + confidence
        lower, upper = np.percentile(scores, percentiles).tolist()

        return lower, upper

    def to_dict(self) -> dict[str, object]:
        """
        Returns a json-encodable dictionary representation of self.
        """
        return {
            "name": self.name,
            "examples": [dataclasses.asdict(example) for example in self.examples],
            "argv": self.argv,
            "commit": self.commit,
            "posix_time": self.posix_time,
            "gpu_name": self.gpu_name,
            "hostname": self.hostname,
        }


ModelArgs = tuple[str, str]
