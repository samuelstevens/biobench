import beartype
import numpy as np
import polars as pl
from jaxtyping import Float, jaxtyped

from .. import config, reporting


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report: ...


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]: ...
