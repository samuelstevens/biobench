"""
Uniform-random baseline for FungiCLEF.

After *n* trials it prints the mean and standard deviation of the user score (expressed as a percentage). No training statistics are used; this is a pure chance-level reference.

Run from the command line:

uv run python -m biobench.fungiclef.random_baseline --cfg configs/neurips.toml
"""

import logging

import beartype
import numpy as np
import tyro

from .. import config, helpers, reporting
from . import FungiDataset, score


@beartype.beartype
def main(cfg: str, n: int = 1_000, seed: int = 42):
    """
    Evaluate uniform random guessing on FungiCLEF.

    Parameters
    ----------
    cfg : str
        Path or key understood by `config.load`; must point to an experiment that defines `data.fungiclef` and the `verbose` flag.
    n : int, default 1_000
        Number of bootstrap trials.
    seed : int, default 42
        Seed for NumPy's `default_rng`.

    Prints
    ------
    "Mean score: mean (std dev)" where scores are fungiclef.score x 100.
    """
    cfg = next(cfg for cfg in config.load(cfg))

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.verbose else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger(__name__)

    train_ds = FungiDataset(cfg.data.fungiclef, "train", None)
    logger.info("Loaded train dataset.")
    val_ds = FungiDataset(cfg.data.fungiclef, "val", None)
    logger.info("Loaded val dataset.")

    rng = np.random.default_rng(seed=seed)
    label_space = np.concatenate(([-1], sorted(train_ds.labels))).astype(int)

    scores = []
    for _ in helpers.progress(range(n), every=n // 100, desc="bootstrapping"):
        preds = [
            reporting.Prediction(
                "deadbeef",
                float(p == t),
                {"y_pred": int(p), "y_true": int(t), "ood": t == -1},
            )
            for p, t in zip(rng.choice(label_space, size=len(val_ds)), val_ds.labels)
        ]
        scores.append(score(preds) * 100)

    print(f"Mean score: {np.mean(scores):.3f} ({np.std(scores):.3f})")


if __name__ == "__main__":
    tyro.cli(main)
