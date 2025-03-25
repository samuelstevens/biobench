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
from jaxtyping import jaxtyped

from . import config

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("biobench")
schema_fpath = pathlib.Path(__file__).parent / "schema.sql"


@beartype.beartype
def get_db(cfg: config.Experiment) -> sqlite3.Connection:
    """Get a connection to the reports database.
    Returns:
        a connection to a sqlite3 database.
    """
    os.makedirs(cfg.report_to, exist_ok=True)
    db = sqlite3.connect(os.path.join(cfg.report_to, "reports.sqlite"), autocommit=True)

    with open(schema_fpath) as fd:
        schema = fd.read()
    db.executescript(schema)
    db.autocommit = False

    return db


@beartype.beartype
def already_ran(db: sqlite3.Connection, cfg: config.Experiment, task_name: str) -> bool:
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


def get_git_hash() -> str:
    """
    Returns the hash of the current git commit, assuming we are in a git repo.
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
    def write(self, db: sqlite3.Connection) -> None:
        """
        Saves the report to disk in a machine-readable SQLite format.

        Args:
        """
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
