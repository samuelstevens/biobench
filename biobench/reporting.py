"""
Common interfaces for models and tasks so that it's easy to add new models (which will work right away with all tasks) and easy to add new tasks (which will work right away with all models).

The model interface is `VisionBackbone`.
See `biobench.third_party_models` for examples of how to subclass it, and note that you have to call `biobench.register_vision_backbone` for it to show up.

The benchmark interface is informal.
"""

import dataclasses
import json
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
        dct["predictions"] = [p.to_dict() for p in self.predictions]

        return dct

    def write(self, conn: sqlite3.Connection):
        """
        Write this report to a SQLite database.
        
        Args:
            conn: SQLite connection to write to
        """
        # Insert into results table
        cursor = conn.cursor()
        
        # Determine method-specific fields
        model_method = "mllm" if self.exp_cfg.model.org in ["anthropic", "openai"] else "cvml"
        
        # Prepare values for results table
        results_values = {
            "task_name": self.task_name,
            "n_train": self.n_train,
            "n_test": len(self.predictions),
            "sampling": self.exp_cfg.sampling,
            
            "model_method": model_method,
            "model_org": self.exp_cfg.model.org,
            "model_ckpt": self.exp_cfg.model.ckpt,
            
            # MLLM-specific fields
            "prompting": self.exp_cfg.prompting if hasattr(self.exp_cfg, "prompting") else None,
            "cot_enabled": 1 if hasattr(self.exp_cfg, "cot") and self.exp_cfg.cot else 0,
            "parse_success_rate": self.parse_success_rate,
            "usd_per_answer": self.usd_per_answer,
            
            # CVML-specific fields
            "classifier_type": self.classifier,
            
            # Configuration and metadata
            "exp_cfg": json.dumps(self.exp_cfg.to_dict()),
            "argv": json.dumps(self.argv),
            "commit": self.commit,
            "posix": int(self.posix_time),
            "gpu_name": self.gpu_name,
            "hostname": self.hostname
        }
        
        # Build the SQL query
        columns = ", ".join(results_values.keys())
        placeholders = ", ".join(["?"] * len(results_values))
        
        # Insert into results table
        cursor.execute(
            f"INSERT INTO results ({columns}) VALUES ({placeholders})",
            list(results_values.values())
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
                    result_id
                )
            )
        
        # Commit the transaction
        conn.commit()
