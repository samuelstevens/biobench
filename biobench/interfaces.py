"""
Common interfaces for models and tasks so that it's easy to add new models (which will work right away with all tasks) and easy to add new tasks (which will work right away with all models).

The model interface is `VisionBackbone`.
See `biobench.third_party_models` for examples of how to subclass it, and note that you have to call `biobench.register_vision_backbone` for it to show up.

The benchmark interface is informal, but is a function that matches the following signature:

```py
def benchmark(args: Args, model_args: ModelArgs) -> tuple[ModelArgs], interfaces.TaskReport]:
    ...
```

In a Haskell-like signature, this is more like `Args -> ModelArgs -> (ModelArgs, TaskReport)`.
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

from . import helpers


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
    See `biobench.registry` for a tutorial on adding new vision backbones.
    """

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
class Prediction:
    """An individual test prediction."""

    id: str
    """Whatever kind of ID; used to find the original image/example."""
    score: float
    """Test score; typically 0 or 1 for classification tasks."""
    info: dict[str, object]
    """Any additional information included. This might be the original class, the true label, etc."""


def default_calc_mean_score(predictions: list[Prediction]) -> float:
    return np.mean([prediction.score for prediction in predictions]).item()


def get_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).name
    else:
        return ""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class TaskReport:
    """
    The result of running a benchmark task.
    """

    # Actual details of the report
    name: str
    """The benchmark name."""
    predictions: list[Prediction]
    """A list of (example_id, score, info) objects"""
    _: dataclasses.KW_ONLY
    calc_mean_score: typing.Callable[[list[Prediction]], float] = (
        default_calc_mean_score
    )
    """A way to calculate mean score from a list of predictions."""
    splits: dict[str, float] = dataclasses.field(default_factory=dict)
    """Other scores that you would like to report. These do not have confidence intervals."""

    # Stuff for trying to reproduce this result. These are filled in by default.
    argv: list[str] = dataclasses.field(default_factory=lambda: sys.argv)
    """Command used to get this report."""
    commit: str = get_git_hash()
    """Git commit for this current report."""
    posix_time: float = dataclasses.field(default_factory=time.time)
    """Time when this report was constructed."""
    gpu_name: str = dataclasses.field(default_factory=get_gpu_name)
    """Name of the GPU that ran this experiment."""
    hostname: str = dataclasses.field(default_factory=socket.gethostname)
    """Machine hostname that ran this experiment."""

    def __repr__(self):
        return f"Report({self.name} with {len(self.predictions)} predictions)"

    def __str__(self):
        return repr(self)

    def get_mean_score(self) -> float:
        """
        Get the mean score of all predictions.
        """
        return self.calc_mean_score(self.predictions)

    def get_confidence_interval(
        self,
        statistic="mean",
        confidence: float = 95,
        n_resamples: int = 500,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Get the confidence interval for the statistics (mean) by bootstrapping individual scores of the predictions.

        NOTE: it's crazy how much easier this would be in Jax. PyTrees of Predictions would simply contains batch dimensions, and then I would `jax.vmap(get_mean_score)(batched_predictions)`.
        """

        rng = np.random.default_rng(seed=seed)
        choices = rng.choice(
            len(self.predictions),
            size=(n_resamples, len(self.predictions)),
            replace=True,
        )

        scores = []
        for choice in helpers.progress(choices, desc=f"CI for {self.name}", every=100):
            scores.append(self.calc_mean_score([self.predictions[i] for i in choice]))

        percentiles = (100 - confidence) / 2, (100 - confidence) / 2 + confidence
        lower, upper = np.percentile(scores, percentiles).tolist()

        return lower, upper

    def to_dict(self) -> dict[str, object]:
        """
        Returns a json-encodable dictionary representation of self.
        """
        return {
            "name": self.name,
            "predictions": [
                dataclasses.asdict(prediction) for prediction in self.predictions
            ],
            "argv": self.argv,
            "commit": self.commit,
            "posix_time": self.posix_time,
            "gpu_name": self.gpu_name,
            "hostname": self.hostname,
        }


@dataclasses.dataclass(frozen=True)
class ModelArgsCvml:
    org: str
    ckpt: str

    def to_dict(self) -> dict[str, object]:
        return {"type": "cvml", **dataclasses.asdict(self)}


@dataclasses.dataclass(frozen=True)
class ModelArgsMllm:
    ckpt: str
    temp: float = 0.0
    prompts: typing.Literal["single-turn", "multi-turn"] = "single-turn"
    quantizations: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"type": "mllm", **dataclasses.asdict(self)}
