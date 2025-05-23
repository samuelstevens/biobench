import dataclasses
import json
import logging
import os.path
import pathlib
import socket
import sqlite3
import subprocess
import sys
import time

import beartype
import numpy as np
import polars as pl
import sklearn.metrics
from jaxtyping import Float, Int, jaxtyped

from . import config, helpers

logger = logging.getLogger(__name__)

schema_fpath = pathlib.Path(__file__).parent / "schema.sql"


@beartype.beartype
def get_db(cfg: config.Experiment) -> sqlite3.Connection:
    """Get a connection to the reports database.

    Args:
        cfg: Experiment configuration

    Returns:
        sqlite3.Connection: A connection to the SQLite database
    """
    os.makedirs(os.path.expandvars(cfg.report_to), exist_ok=True)
    helpers.warn_if_nfs(cfg.report_to)
    db_fpath = os.path.join(os.path.expandvars(cfg.report_to), "reports.sqlite")
    db = sqlite3.connect(db_fpath, autocommit=True)

    with open(schema_fpath) as fd:
        schema = fd.read()
    db.executescript(schema)
    db.autocommit = False

    return db


@beartype.beartype
def already_ran(db: sqlite3.Connection, cfg: config.Experiment, task_name: str) -> bool:
    """Check if an experiment has already been run.

    Args:
        db: SQLite database connection
        cfg: Experiment configuration
        task_name: Name of the task to check

    Returns:
        bool: True if the experiment has already been run, False otherwise
    """
    query = """
    SELECT COUNT(*)
    FROM experiments
    WHERE task_name = ?
    AND model_org = ?
    AND model_ckpt = ?
    AND n_train = ?
    """
    values = (task_name, cfg.model.org, cfg.model.ckpt, cfg.n_train)

    (count,) = db.execute(query, values).fetchone()
    return count > 0


@beartype.beartype
def is_claimed(db: sqlite3.Connection, cfg: config.Experiment, task_name: str) -> bool:
    """Check if a run is already claimed by another process.

    Args:
        db: SQLite database connection
        cfg: Experiment configuration
        task_name: Name of the task to check

    Returns:
        bool: True if the run is already claimed, False otherwise
    """
    query = """
    SELECT COUNT(*)
    FROM runs
    WHERE task_name = ?
    AND model_org = ?
    AND model_ckpt = ?
    AND n_train = ?
    """
    values = (task_name, cfg.model.org, cfg.model.ckpt, cfg.n_train)

    (count,) = db.execute(query, values).fetchone()
    return count > 0


@beartype.beartype
def claim_run(db: sqlite3.Connection, cfg: config.Experiment, task_name: str) -> bool:
    """Try to claim (task_name, model, n_train).

    Args:
        db: SQLite database connection
        cfg: Experiment configuration
        task_name: Name of the task to claim

    Returns:
        bool: True if this process inserted the row and now "owns" the run,
              False if row already existed and another worker has it
    """

    stmt = """
    INSERT OR IGNORE INTO runs
    (task_name, model_org, model_ckpt, n_train, pid, posix)
    VALUES (?,?,?,?,?,?)
    """
    values = (
        task_name,
        cfg.model.org,
        cfg.model.ckpt,
        cfg.n_train,
        os.getpid(),
        time.time(),
    )

    try:
        cur = db.execute(stmt, values)
        db.commit()
        return cur.rowcount == 1  # 1 row inserted -> we won
    except Exception:
        db.rollback()
        raise


@beartype.beartype
def release_run(db: sqlite3.Connection, cfg: config.Experiment, task_name: str) -> None:
    """Delete the coordination row so others may claim again.

    Args:
        db: SQLite database connection
        cfg: Experiment configuration
        task_name: Name of the task to release
    """
    stmt = """
    DELETE FROM runs
    WHERE task_name=? AND model_org=? AND model_ckpt=? AND n_train=?
    """
    values = (task_name, cfg.model.org, cfg.model.ckpt, cfg.n_train)
    logger.info("Releasing claim on (%s, %s, %s, %d)", *values)

    try:
        db.execute(stmt, values)
        db.commit()
        logger.info("Released claim on (%s, %s, %s, %d)", *values)
    except Exception:
        db.rollback()
        raise


@beartype.beartype
def clear_stale_claims(db: sqlite3.Connection, *, max_age_hours: int = 72) -> int:
    """
    Delete rows in `runs` whose POSIX timestamp is older than `max_age_hours`.

    Returns
    -------
    int
        Number of rows deleted.
    """
    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive")

    cutoff = time.time() - max_age_hours * 3600
    try:
        cur = db.execute("DELETE FROM runs WHERE posix < ?", (cutoff,))
        db.commit()
        return cur.rowcount
    except Exception:
        db.rollback()
        raise


@beartype.beartype
def get_git_hash() -> str:
    """Returns the hash of the current git commit.

    Returns:
        str: The hash of the current git commit, assuming we are in a git repo
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Prediction:
    """An individual test prediction."""

    id: str
    """Whatever kind of ID; used to find the original image/example."""
    score: float
    """Test score; typically 0 or 1 for classification tasks."""
    info: dict[str, object]
    """Any additional information included. This might be the original class, the true label, etc."""


def get_gpu_name() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).name
    else:
        return ""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Report:
    """
    The result of running a benchmark task.
    """

    # Actual details of the report
    task_name: str
    """The benchmark name."""
    predictions: list[Prediction]
    """A list of (example_id, score, info) objects"""
    cfg: config.Experiment
    """Experimental config."""
    _: dataclasses.KW_ONLY
    splits: dict[str, float] = dataclasses.field(default_factory=dict)
    """Other scores that you would like to report. These do not have confidence intervals."""

    # Stuff for trying to reproduce this result. These are filled in by default.
    argv: list[str] = dataclasses.field(default_factory=lambda: sys.argv)
    """Command used to get this report."""
    git_commit: str = get_git_hash()
    """Git commit for this current report."""
    posix: float = dataclasses.field(default_factory=time.time)
    """Time when this report was constructed."""
    gpu_name: str = dataclasses.field(default_factory=get_gpu_name)
    """Name of the GPU that ran this experiment."""
    hostname: str = dataclasses.field(default_factory=socket.gethostname)
    """Machine hostname that ran this experiment."""

    def __repr__(self):
        return f"Report({self.task_name} with {len(self.predictions)} predictions)"

    def __str__(self):
        return repr(self)

    @beartype.beartype
    def write(self) -> None:
        """Saves the report to disk in a machine-readable SQLite format."""
        db = get_db(self.cfg)

        preds_stmt = "INSERT INTO predictions(img_id, score, info, experiment_id) VALUES(?, ?, ?, ?)"
        exp_stmt = "INSERT INTO experiments(task_name, model_org, model_ckpt, n_train, exp_cfg, argv, git_commit, posix, gpu_name, hostname) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        try:
            cursor = db.cursor()

            exp_values = (
                self.task_name.lower(),
                self.cfg.model.org,
                self.cfg.model.ckpt,
                self.cfg.n_train,
                json.dumps(self.cfg.to_dict()),
                json.dumps(self.argv),
                self.git_commit,
                self.posix,
                self.gpu_name,
                self.hostname,
            )
            cursor.execute(exp_stmt, exp_values)
            exp_id = cursor.lastrowid
            preds_values = [
                (pred.id, pred.score, json.dumps(pred.info), exp_id)
                for pred in self.predictions
            ]
            cursor.executemany(preds_stmt, preds_values)

            # Commit the transaction if all statements succeed
            db.commit()
        except sqlite3.Error as err:
            # Roll back the transaction in case of error
            db.rollback()
            logger.critical("Error writing report for '%s': %s", self.task_name, err)
            raise


@beartype.beartype
def micro_acc(preds: list[Prediction]) -> float:
    y_pred = np.array([
        next(p.info[key] for key in ("y_pred", "pred_y") if key in p.info)
        for p in preds
    ])
    y_true = np.array([p.info.get("y_true", p.info.get("true_y")) for p in preds])
    return sklearn.metrics.accuracy_score(y_true, y_pred)


@beartype.beartype
def macro_acc(preds: list[Prediction]) -> float:
    y_pred = np.array([
        next(p.info[key] for key in ("y_pred", "pred_y") if key in p.info)
        for p in preds
    ])
    y_true = np.array([p.info.get("y_true", p.info.get("true_y")) for p in preds])
    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred)


@beartype.beartype
def micro_f1(preds: list[Prediction]) -> float:
    y_pred = np.array([
        next(p.info[key] for key in ("y_pred", "pred_y") if key in p.info)
        for p in preds
    ])
    y_true = np.array([p.info.get("y_true", p.info.get("true_y")) for p in preds])
    return sklearn.metrics.f1_score(y_true, y_pred, average="micro")


@beartype.beartype
def macro_f1(preds: list[Prediction], *, labels: list[int] | None = None) -> float:
    y_pred = np.array([
        next(p.info[key] for key in ("y_pred", "pred_y") if key in p.info)
        for p in preds
    ])
    y_true = np.array([p.info.get("y_true", p.info.get("true_y")) for p in preds])

    if labels is None:
        labels = np.unique(np.stack([y_true, y_pred]))
        labels = np.arange(labels.max() + 1)
    else:
        labels = np.array(labels)

    assert (np.arange(labels.size) == labels).all()

    return sklearn.metrics.f1_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=0.0
    )


@beartype.beartype
def micro_acc_batch(
    y_true: Int[np.ndarray, "*batch n"], y_pred: Int[np.ndarray, "*batch n"]
) -> Float[np.ndarray, "*batch"]:
    """
    Vectorised **micro-accuracy** (overall proportion of correct predictions).

    * Works on any leading `*batch` prefix; the final axis `n` is the number of examples.
    * Complexity O(B·n) time, O(1) extra memory.

    Parameters
    ----------
    y_true, y_pred
        Integer class labels / predictions >= 0.  All leading dimensions
        (`*batch`) must match; `n` is the sample count.

    Returns
    -------
    acc : np.ndarray
        Shape `*batch`; accuracy for every element of the batch prefix.
    """
    acc = (y_true == y_pred).mean(axis=-1, dtype=float)
    return acc


@jaxtyped(typechecker=beartype.beartype)
def macro_f1_batch(
    y_true: Int[np.ndarray, "*batch n"],
    y_pred: Int[np.ndarray, "*batch n"],
    *,
    labels: Int[np.ndarray, " c"] | None = None,
) -> Float[np.ndarray, "*batch"]:
    """
    Vectorised macro-F1 for large class counts.

    Accepts any leading `*batch` prefix; the last axis `n` is the number
    of examples.  Runs in O(B·n) time and O(B·C) memory, where
    B = prod(*batch) and C = #classes.

    All elements in y_true, y_pred, and labels must be integers >= 0. (Negative IDs would break the offset arithmetic; floats break np.bincount.)
    """

    # flatten batch prefix
    *batch_shape, n = y_true.shape
    b = int(np.prod(batch_shape))  # total batches

    y_true = y_true.reshape(b, n)
    y_pred = y_pred.reshape(b, n)

    # label remapping to dense 0...C-1
    if labels is None:
        labels = np.unique(np.stack([y_true, y_pred]))

    labels = np.arange(labels.max() + 1)
    c = labels.size
    assert (np.arange(c) == labels).all()

    # offsets for per-batch bincounts
    offset = np.arange(b, dtype=np.int64)[:, None] * c  # (b, 1)
    yz_true = y_true + offset  # (b, n)
    yz_pred = y_pred + offset  # (b, n)

    # counts for all examples
    true_cnt = np.bincount(yz_true.ravel(), minlength=(b * c)).reshape(b, c)
    pred_cnt = np.bincount(yz_pred.ravel(), minlength=(b * c)).reshape(b, c)

    # true-positives
    tp_mask = y_true == y_pred  # (B, n)
    tp_off = yz_true[tp_mask]  # 1-D
    tp_cnt = np.bincount(tp_off, minlength=(b * c)).reshape(b, c)

    # F1 per class, then macro average
    fp = pred_cnt - tp_cnt
    fn = true_cnt - tp_cnt
    denom = 2 * tp_cnt + fp + fn  # (B, C)

    f1_c = np.zeros_like(denom, dtype=float)
    np.divide(2 * tp_cnt, denom, out=f1_c, where=denom != 0)

    macro_f1 = f1_c.mean(axis=-1)  # (B,)
    return macro_f1.reshape(batch_shape)


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores_macro_f1(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
    """
    Polars dataframe with schema

    Schema({'task_name': String, 'model_ckpt': String, 'img_id': String, 'score': Float64, 'y_true': String, 'y_pred': String})
    """

    n, *rest = df.group_by("model_ckpt").agg(n=pl.len()).get_column("n").to_list()
    assert all(n == i for i in rest)

    if b > 0:
        assert rng is not None, "must provide rng argument"
        i_bs = rng.integers(0, n, size=(b, n), dtype=np.int32)

    scores = {}

    y_pred_buf = np.empty((b, n), dtype=np.int32)
    y_true_buf = np.empty((b, n), dtype=np.int32)

    for model_ckpt in df.get_column("model_ckpt").unique().sort().to_list():
        # pull y_true and y_pred for *one* model
        y_pred = (
            df.filter(pl.col("model_ckpt") == model_ckpt)
            .select("img_id", "y_pred")
            .unique()
            .sort("img_id")
            .get_column("y_pred")
            .cast(pl.Float32)
            .cast(pl.Int32)
            .to_numpy()
        )

        if len(y_pred) == 0:
            continue

        y_true = (
            df.filter(pl.col("model_ckpt") == model_ckpt)
            .select("img_id", "y_true")
            .unique()
            .sort("img_id")
            .get_column("y_true")
            .cast(pl.Float32)
            .cast(pl.Int32)
            .to_numpy()
        )
        assert y_true.size == y_pred.size

        if b > 0:
            # bootstrap resample into pre-allocated buffers
            np.take(y_pred, i_bs, axis=0, out=y_pred_buf)
            np.take(y_true, i_bs, axis=0, out=y_true_buf)
            scores[model_ckpt] = macro_f1_batch(y_true_buf, y_pred_buf)
        else:
            scores[model_ckpt] = np.array([macro_f1_batch(y_true, y_pred)])

    return scores


##########
# COLORS #
##########


# https://coolors.co/palette/001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-9b2226

BLACK_HEX = "001219"
BLACK_RGB = (0, 18, 25)
BLACK_RGB01 = tuple(c / 256 for c in BLACK_RGB)

BLUE_HEX = "005f73"
BLUE_RGB = (0, 95, 115)
BLUE_RGB01 = tuple(c / 256 for c in BLUE_RGB)

CYAN_HEX = "0a9396"
CYAN_RGB = (10, 147, 150)
CYAN_RGB01 = tuple(c / 256 for c in CYAN_RGB)

SEA_HEX = "94d2bd"
SEA_RGB = (148, 210, 189)
SEA_RGB01 = tuple(c / 256 for c in SEA_RGB)

CREAM_HEX = "e9d8a6"
CREAM_RGB = (233, 216, 166)
CREAM_RGB01 = tuple(c / 256 for c in CREAM_RGB)

GOLD_HEX = "ee9b00"
GOLD_RGB = (238, 155, 0)
GOLD_RGB01 = tuple(c / 256 for c in GOLD_RGB)

ORANGE_HEX = "ca6702"
ORANGE_RGB = (202, 103, 2)
ORANGE_RGB01 = tuple(c / 256 for c in ORANGE_RGB)

RUST_HEX = "bb3e03"
RUST_RGB = (187, 62, 3)
RUST_RGB01 = tuple(c / 256 for c in RUST_RGB)

SCARLET_HEX = "ae2012"
SCARLET_RGB = (174, 32, 18)
SCARLET_RGB01 = tuple(c / 256 for c in SCARLET_RGB)

RED_HEX = "9b2226"
RED_RGB = (155, 34, 38)
RED_RGB01 = tuple(c / 256 for c in RED_RGB)
