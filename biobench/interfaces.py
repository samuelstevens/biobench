"""
Common interfaces for models and tasks so that it's easy to add new models (which will work right away with all tasks) and easy to add new tasks (which will work right away with all models).

The model interface is `VisionBackbone`.
See `biobench.third_party_models` for examples of how to subclass it, and note that you have to call `biobench.register_vision_backbone` for it to show up.

The benchmark interface is informal.
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
from jaxtyping import jaxtyped

from . import config, helpers


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Prediction:
    """An individual test prediction."""

    id: str
    """Whatever kind of ID; used to find the original image/example."""
    score: float
    """Test score; typically 0 or 1 for classification tasks."""
    n_train: int
    """Number of training examples used in this prection."""
    info: dict[str, object]
    """Any additional information included. This might be the original class, the true label, etc."""

    def to_dict(self) -> dict[str, object]:
        """Convert prediction to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)


def get_git_hash() -> str:
    """
    Returns the hash of the current git commit, assuming we are in a git repo.
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def default_calc_mean_score(predictions: list[Prediction]) -> float:
    return np.mean([prediction.score for prediction in predictions]).item()


def get_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).name
    else:
        return ""


@dataclasses.dataclass
class Report:
    """
    The result of running a benchmark task.

    This class is designed to store results and metadata, with experiment configuration
    stored in the exp_cfg field to avoid duplication.
    """

    task_name: str
    predictions: list[Prediction]
    n_train: int
    """Number of training samples *actually* used."""

    exp_cfg: config.Experiment

    _: dataclasses.KW_ONLY

    calc_mean_score: typing.Callable[[list[Prediction]], float] = (
        default_calc_mean_score
    )

    # MLLM-specific
    parse_success_rate: float | None = None
    usd_per_answer: float | None = None

    # CVML-specific
    classifier: typing.Literal["knn", "svm", "ridge"] | None = None

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
        model_name = self.exp_cfg.model.ckpt
        return f"Report({self.task_name}, {model_name}, {len(self.predictions)} predictions)"

    def to_dict(self) -> dict[str, object]:
        """
        Convert the report to a JSON-compatible dictionary.
        Uses dataclasses.asdict() with custom handling for special types.
        """

        dct = dataclasses.asdict(self)

        # Handle special cases
        dct["exp_cfg"] = self.exp_cfg.to_dict()
        dct["calc_mean_score"] = self.calc_mean_score.__name__
        dct["predictions"] = [p.to_dict() for p in self.predictions]

        return dct

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
        for choice in helpers.progress(
            choices, desc=f"CI for {self.task_name}", every=100
        ):
            scores.append(self.calc_mean_score([self.predictions[i] for i in choice]))

        percentiles = (100 - confidence) / 2, (100 - confidence) / 2 + confidence
        lower, upper = np.percentile(scores, percentiles).tolist()

        return lower, upper
