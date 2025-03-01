""" """

import dataclasses
import json
import os
import pathlib
import socket
import sqlite3
import subprocess
import sys
import time
import typing

import beartype
import torch
from jaxtyping import jaxtyped

from . import config

schema_fpath = pathlib.Path(__file__).parent / "schema.sql"


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Prediction:
    """An individual test prediction."""

    img_id: str
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

    task_cluster: str | None = None
    """The cluster this task belongs to (e.g., 'birds', 'plants')."""
    task_subcluster: str | None = None
    """The subcluster this task belongs to (e.g., 'songbirds', 'trees')."""

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

    def get_conn(self) -> sqlite3.Connection:
        os.makedirs(self.exp_cfg.report_to, exist_ok=True)
        conn = sqlite3.connect(
            os.path.join(self.exp_cfg.report_to, "results.sqlite"), autocommit=False
        )
        with open(schema_fpath) as fd:
            schema = fd.read()
        conn.executescript(schema)
        return conn

    def to_dict(self) -> dict[str, object]:
        """
        Convert the report to a JSON-compatible dictionary.
        Uses dataclasses.asdict() with custom handling for special types.
        """

        dct = dataclasses.asdict(self)

        # Handle special cases
        dct["exp_cfg"] = self.exp_cfg.to_dict()
        dct["predictions"] = [p.to_dict() for p in self.predictions]

        return dct

    def write(self, conn: sqlite3.Connection | None = None):
        """
        Write this report to a SQLite database.

        Args:
            conn: SQLite connection to write to
        """
        if not conn:
            conn = self.get_conn()

        # Insert into results table
        cursor = conn.cursor()

        # Determine method-specific fields
        model_method = (
            "mllm" if self.exp_cfg.model.org in ["anthropic", "openai"] else "cvml"
        )

        # Prepare values for results table
        results_values = {
            "task_name": self.task_name,
            "task_cluster": self.task_cluster,
            "task_subcluster": self.task_subcluster,
            "n_train": self.n_train,
            "n_test": len(self.predictions),
            "sampling": self.exp_cfg.sampling,
            "model_method": model_method,
            "model_org": self.exp_cfg.model.org,
            "model_ckpt": self.exp_cfg.model.ckpt,
            # MLLM-specific fields
            "prompting": self.exp_cfg.prompting
            if hasattr(self.exp_cfg, "prompting")
            else None,
            "cot_enabled": 1
            if hasattr(self.exp_cfg, "cot") and self.exp_cfg.cot
            else 0,
            "parse_success_rate": self.parse_success_rate,
            "usd_per_answer": self.usd_per_answer,
            # CVML-specific fields
            "classifier_type": self.classifier,
            # Configuration and metadata
            "exp_cfg": json.dumps(self.exp_cfg.to_dict()),
            "argv": json.dumps(self.argv),
            "git_commit": self.commit,
            "posix": int(self.posix_time),
            "gpu_name": self.gpu_name,
            "hostname": self.hostname,
        }

        # Build the SQL query
        columns = ", ".join(results_values.keys())
        placeholders = ", ".join(["?"] * len(results_values))

        # Insert into results table
        cursor.execute(
            f"INSERT INTO results ({columns}) VALUES ({placeholders})",
            list(results_values.values()),
        )

        # Get the rowid of the inserted result
        result_id = cursor.lastrowid

        # Insert predictions
        for pred in self.predictions:
            cursor.execute(
                "INSERT INTO predictions (img_id, score, n_train, info, result_id) VALUES (?, ?, ?, ?, ?)",
                (
                    pred.img_id,
                    pred.score,
                    pred.n_train,
                    json.dumps(pred.info),
                    result_id,
                ),
            )

        # Commit the transaction
        conn.commit()


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

ALL_HEX = [
    BLACK_HEX,
    BLUE_HEX,
    CYAN_HEX,
    SEA_HEX,
    CREAM_HEX,
    GOLD_HEX,
    ORANGE_HEX,
    RUST_HEX,
    SCARLET_HEX,
    RED_HEX,
]

ALL_RGB01 = [
    BLACK_RGB01,
    BLUE_RGB01,
    CYAN_RGB01,
    SEA_RGB01,
    CREAM_RGB01,
    GOLD_RGB01,
    ORANGE_RGB01,
    RUST_RGB01,
    SCARLET_RGB01,
    RED_RGB01,
]
