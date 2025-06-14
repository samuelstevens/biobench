"""
Uniform-random baseline for FishNet functional-trait prediction.

Each bootstrap trial:

1. Loads the FishNet test split specified in the experiment config.
2. Generates a 9-bit prediction vector for every image by i.i.d. Bernoulli(0.5).
3. Scores macro-F1 via the existing `fishnet.score` helper.

After *n* trials it prints the mean and standard deviation of the macro-F1 (expressed as a percentage). No training statistics are used; this is a pure chance-level reference.

Run from the command line:

uv run python -m biobench.fishnet.random_baseline --cfg configs/neurips.toml
"""

import logging

import beartype
import numpy as np
import sklearn.metrics
import tyro

from .. import config, helpers
from . import ImageDataset


@beartype.beartype
def main(cfg: str, n: int = 1_000, seed: int = 42):
    """Estimate the macro-F1 of uniform random guessing on FishNet.

    Args:
        cfg: Path or key understood by `config.load`; must point to an experiment that defines `data.fishnet` and the `verbose` flag.
        n: Number of bootstrap trials. Default is 1_000.
        seed: Seed for NumPy's `default_rng`. Default is 42.

    Prints:
        "Mean score: mean (std dev)" where scores are macro-F1 x 100.
    """
    cfg = next(cfg for cfg in config.load(cfg))

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger(__name__)

    rng = np.random.default_rng(seed)

    test_ds = ImageDataset(cfg.data.fishnet, "test.csv", transform=None)
    logger.info("Loaded test dataset.")

    scores = []
    for _ in helpers.progress(range(n), every=n // 100, desc="bootstrapping"):
        y_pred = (rng.random((len(test_ds), 9)) < 0.5).astype(int)
        y_true = test_ds.labels
        score = sklearn.metrics.f1_score(
            test_ds.labels, y_pred, average="macro", labels=np.unique(y_true)
        )
        scores.append(score * 100)

    print(f"Mean score: {np.mean(scores):.3f} ({np.std(scores):.3f})")


if __name__ == "__main__":
    tyro.cli(main)
