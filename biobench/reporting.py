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
import sklearn.metrics
from jaxtyping import jaxtyped

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
        db.execute(stmt, values)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return db.total_changes == 1  # 1 row inserted -> we won


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

    try:
        db.execute(stmt, values)
        db.commit()
    except Exception:
        db.rollback()
        raise


@beartype.beartype
class JobQueue[T]:
    def __init__(self, max_size: int):
        """Create queue. max_size >= 0; 0 => always full."""
        self._max_size = max_size
        self._items = []  # FIFO queue of items

    def submit(self, item: T) -> None:
        """RuntimeError if full()."""
        if self.full():
            raise RuntimeError(f"Queue is full (max_size={self._max_size})")
        self._items.append(item)

    def pop(self) -> T:
        """Block until *some* contained Job is done, remove and return its payload."""
        if not self._items:
            return None

        # First, check if any job is already done (non-blocking)
        for i, item in enumerate(self._items):
            if self._is_done(item):
                return self._items.pop(i)

        # If no job is done, wait for the first one to complete
        while self._items:
            for i, item in enumerate(self._items):
                if self._is_done(item):
                    return self._items.pop(i)
            # No job is done yet, sleep briefly before checking again
            time.sleep(0.1)
        
        return None

    def _is_done(self, obj) -> bool:
        """Check if an object or any of its nested items is done."""
        # Direct check for objects with done() method
        if hasattr(obj, "done") and callable(obj.done) and obj.done():
            return True
            
        # Check first level of nesting (tuples, lists, dicts)
        if isinstance(obj, (tuple, list)):
            for item in obj:
                if hasattr(item, "done") and callable(item.done) and item.done():
                    return True
                # Check second level of nesting
                if isinstance(item, (tuple, list)):
                    for subitem in item:
                        if hasattr(subitem, "done") and callable(subitem.done) and subitem.done():
                            return True
                elif isinstance(item, dict):
                    for subitem in item.values():
                        if hasattr(subitem, "done") and callable(subitem.done) and subitem.done():
                            return True
        elif isinstance(obj, dict):
            for item in obj.values():
                if hasattr(item, "done") and callable(item.done) and item.done():
                    return True
                # Check second level of nesting
                if isinstance(item, (tuple, list)):
                    for subitem in item:
                        if hasattr(subitem, "done") and callable(subitem.done) and subitem.done():
                            return True
                elif isinstance(item, dict):
                    for subitem in item.values():
                        if hasattr(subitem, "done") and callable(subitem.done) and subitem.done():
                            return True
                    
        return False

    def full(self) -> bool:
        """Return True if the queue is at capacity."""
        if self._max_size == 0:
            return True
        return len(self._items) >= self._max_size

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self._items)

    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return bool(self._items)

    def __iter__(self):
        """Iterate over items in FIFO order."""
        return iter(self._items)


@beartype.beartype
class ClaimReaper:
    def __init__(self, db): ...

    def __len__(self) -> int:
        return 0

    def add_claim(self, cfg: config.Experiment, task: str): ...

    def register(self) -> "ClaimReaper":
        return self


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
def macro_f1(preds: list[Prediction]) -> float:
    y_pred = np.array([
        next(p.info[key] for key in ("y_pred", "pred_y") if key in p.info)
        for p in preds
    ])
    y_true = np.array([p.info.get("y_true", p.info.get("true_y")) for p in preds])
    return sklearn.metrics.f1_score(
        y_true, y_pred, average="macro", labels=np.unique(y_true)
    )


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
