import json
import math
import multiprocessing
import pathlib
import sqlite3
import time

import beartype
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import config, reporting

multiprocessing.set_start_method("spawn", force=True)


@st.composite
def _prediction_list(draw):
    """Generate an arbitrary non-empty list[Prediction] for single-label multiclass."""

    n = draw(st.integers(min_value=1, max_value=256))
    # Allow up to 50 distinct class IDs (0-50) - plenty for the property test.
    y_true = draw(
        st.lists(st.integers(min_value=0, max_value=50), min_size=n, max_size=n)
    )
    y_pred = draw(
        st.lists(st.integers(min_value=0, max_value=50), min_size=n, max_size=n)
    )

    preds = [
        reporting.Prediction(
            id=str(i),
            score=float(y_pred[i] == y_true[i]),
            info={"y_true": y_true[i], "y_pred": y_pred[i]},
        )
        for i in range(n)
    ]
    return preds


@st.composite
def _prediction_batch(draw):
    """Generate a *batch* (B ≥ 1) of equal-length prediction lists."""
    B = draw(st.integers(1, 2))  # batch size
    n = draw(st.integers(1, 4))  # common sample count
    batch = []
    for _ in range(B):
        y_true = draw(st.lists(st.integers(0, 50), min_size=n, max_size=n))
        y_pred = draw(st.lists(st.integers(0, 50), min_size=n, max_size=n))
        preds = [
            reporting.Prediction(
                id=str(i),
                score=float(y_pred[i] == y_true[i]),
                info={"y_true": y_true[i], "y_pred": y_pred[i]},
            )
            for i in range(n)
        ]
        batch.append(preds)
    return batch


@given(preds=_prediction_list())
def test_micro_f1_equals_micro_accuracy(preds):
    """Micro-averaged F1 must equal micro accuracy for single-label data."""

    acc = reporting.micro_acc(preds)
    f1 = reporting.micro_f1(preds)

    # Floating math can introduce tiny error, so compare with tolerance
    assert math.isclose(acc, f1, rel_tol=1e-12, abs_tol=1e-12)


@given(preds=_prediction_list())
def test_macro_f1_batch_matches_macro_f1_bsz0(preds):
    """Vectorised implementation must equal our non-batched macro-F1."""
    y_true = np.fromiter((p.info["y_true"] for p in preds), dtype=int)
    y_pred = np.fromiter((p.info["y_pred"] for p in preds), dtype=int)

    ours = reporting.macro_f1_batch(y_true, y_pred)
    ref = reporting.macro_f1(preds)

    assert math.isclose(ours, ref, rel_tol=1e-12, abs_tol=1e-12)


@given(preds=_prediction_list())
def test_macro_f1_batch_matches_macro_f1_bsz1(preds):
    """Vectorised implementation must equal our non-batched macro-F1."""
    y_true = np.fromiter((p.info["y_true"] for p in preds), dtype=int)
    y_pred = np.fromiter((p.info["y_pred"] for p in preds), dtype=int)

    ours = reporting.macro_f1_batch(y_true[None, :], y_pred[None, :])[0]
    ref = reporting.macro_f1(preds)

    assert math.isclose(ours, ref, rel_tol=1e-12, abs_tol=1e-12)


@given(batch=_prediction_batch())
def test_macro_f1_batch_matches_macro_f1_bsz_n(batch):
    """For a true batch (B > 1 possible), vectorised `macro_f1_batch` must equal looping over the legacy `macro_f1`."""
    # stack into (B, n)
    y_true = np.stack([
        np.fromiter((p.info["y_true"] for p in preds), dtype=int) for preds in batch
    ])
    y_pred = np.stack([
        np.fromiter((p.info["y_pred"] for p in preds), dtype=int) for preds in batch
    ])

    labels = np.unique(np.stack([y_true, y_pred]))
    labels = np.arange(labels.max() + 1)

    ours = reporting.macro_f1_batch(y_true, y_pred)
    ref = np.array([
        reporting.macro_f1(preds, labels=labels.tolist()) for preds in batch
    ])

    assert np.allclose(ours, ref, rtol=1e-12, atol=1e-12)


f1_macro_edgecases = [
    [[reporting.Prediction(id="0", score=1.0, info={"y_true": 2, "y_pred": 2})]],
    # [[reporting.Prediction(id="0", score=1.0, info={"y_true": 0, "y_pred": 0})]],
    # [
    #     [
    #         reporting.Prediction(id="0", score=1.0, info={"y_true": 0, "y_pred": 0}),
    #         reporting.Prediction(id="1", score=0.0, info={"y_true": 1, "y_pred": 0}),
    #     ],
    # ],
    # [
    #     [
    #         reporting.Prediction(id="0", score=1.0, info={"y_true": 0, "y_pred": 0}),
    #         reporting.Prediction(id="1", score=0.0, info={"y_true": 0, "y_pred": 1}),
    #     ]
    # ],
    # [
    #     [
    #         reporting.Prediction(id="0", score=1.0, info={"y_true": 0, "y_pred": 0}),
    #         reporting.Prediction(id="1", score=0.0, info={"y_true": 1, "y_pred": 0}),
    #     ],
    #     [
    #         reporting.Prediction(id="1", score=0.0, info={"y_true": 1, "y_pred": 0}),
    #         reporting.Prediction(id="1", score=0.0, info={"y_true": 1, "y_pred": 0}),
    #     ],
    # ],
]


@pytest.mark.parametrize("batch", f1_macro_edgecases)
def test_macro_f1_batch_matches_macro_f1_edgecases(batch):
    """For a true batch (B > 1 possible), vectorised `macro_f1_batch` must equal looping over the legacy `macro_f1`."""
    # stack into (B, n)
    y_true = np.stack([
        np.fromiter((p.info["y_true"] for p in preds), dtype=int) for preds in batch
    ])
    y_pred = np.stack([
        np.fromiter((p.info["y_pred"] for p in preds), dtype=int) for preds in batch
    ])

    ours = reporting.macro_f1_batch(y_true, y_pred)  # shape (B,)
    ref = np.array([reporting.macro_f1(preds) for preds in batch])

    assert np.allclose(ours, ref, rtol=1e-12, atol=1e-12)


##############################
# Tests for SQLite Reporting #
##############################

NOW = 1_700_000_000.0  # fixed "current" time so tests are deterministic
HOUR = 3600


@beartype.beartype
def make_cfg(tmp_path: pathlib.Path) -> config.Experiment:
    """Return a minimal Experiment that points its report DB inside tmp_path."""
    return config.Experiment(
        model=config.Model(org="openai", ckpt="ViT-B/16"),
        n_train=-1,
        report_to=str(tmp_path),
        log_to=str(tmp_path / "logs"),
    )


@beartype.beartype
def _insert_experiment(db: sqlite3.Connection, cfg: config.Experiment, task: str):
    """Directly insert a row into experiments (used to pre-populate scenario 1)."""
    values = (
        task,
        cfg.model.org,
        cfg.model.ckpt,
        cfg.n_train,
        json.dumps(cfg.to_dict()),
        "[]",
        "deadbeef",
        time.time(),
        "",
        "pytest",
    )
    stmt = """
    INSERT INTO experiments
    (task_name, model_org, model_ckpt, n_train,
     exp_cfg, argv, git_commit, posix, gpu_name, hostname)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    """
    db.execute(stmt, values)
    db.commit()


@beartype.beartype
def _insert_claim(
    db: sqlite3.Connection, cfg: config.Experiment, task: str, *, age_h: float | int
):
    """Insert a claim with `age_h` hours of staleness via claim_run."""
    # fake the clock *during* the INSERT
    t0 = time.time
    time.time = lambda: NOW - age_h * HOUR
    try:
        assert reporting.claim_run(db, cfg, task)  # must succeed
    finally:
        time.time = t0


def test_skip_when_experiment_exists(tmp_path):
    cfg = make_cfg(tmp_path)
    task = "plantnet"

    db = reporting.get_db(cfg)
    _insert_experiment(db, cfg, task)

    assert reporting.already_ran(db, cfg, task) is True


BUSY_TIMEOUT = 30
WAIT = BUSY_TIMEOUT + 5


def _worker(cfg: config.Experiment, task: str, q, succeed: bool):
    db = reporting.get_db(cfg)
    if reporting.claim_run(db, cfg, task):
        time.sleep(20)
        if succeed:
            _insert_experiment(db, cfg, task)
            reporting.release_run(db, cfg, task)
            q.put("winner")
        else:
            reporting.release_run(db, cfg, task)
            q.put("failed")
    else:
        q.put("skip")


def test_one_winner_many_launchers(tmp_path):
    cfg = make_cfg(tmp_path)
    task = "kabr"
    q = multiprocessing.Queue()

    n_workers = 4

    procs = [
        multiprocessing.Process(target=_worker, args=(cfg, task, q, True))
        for _ in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=WAIT)

    results = [q.get(timeout=WAIT) for _ in range(n_workers)]
    assert results.count("winner") == 1
    assert results.count("skip") == n_workers - 1

    db = reporting.get_db(cfg)
    # experiments row present
    assert reporting.already_ran(db, cfg, task)
    # runs table cleaned up
    assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_reclaim_after_failure(tmp_path):
    cfg = make_cfg(tmp_path)
    task = "kabr"
    q = multiprocessing.Queue()

    n_workers = 4

    # wave 1: one process fails
    procs1 = [multiprocessing.Process(target=_worker, args=(cfg, task, q, False))]
    procs1.extend([
        multiprocessing.Process(target=_worker, args=(cfg, task, q, True))
        for i in range(n_workers - 1)
    ])
    for p in procs1:
        p.start()
        time.sleep(0.5)
    for p in procs1:
        p.join(timeout=40)

    results1 = [q.get(timeout=40) for _ in range(n_workers)]
    assert results1.count("failed") == 1

    # wave 2: slot should be claimable again
    procs2 = [
        multiprocessing.Process(target=_worker, args=(cfg, task, q, True))
        for _ in range(n_workers)
    ]
    for p in procs2:
        p.start()
    for p in procs2:
        p.join(timeout=40)

    results2 = [q.get(timeout=40) for _ in range(n_workers)]
    assert results2.count("winner") == 1

    db = reporting.get_db(cfg)
    assert reporting.already_ran(db, cfg, task)
    assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_empty_db(tmp_path):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    assert reporting.clear_stale_claims(db, max_age_hours=72) == 0


def test_fresh_kept(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    _insert_claim(db, cfg, "task", age_h=5)

    monkeypatch.setattr(time, "time", lambda: NOW)
    assert reporting.clear_stale_claims(db) == 0
    assert reporting.is_claimed(db, cfg, "task")


def test_stale_removed_and_reclaimable(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    _insert_claim(db, cfg, "task", age_h=100)

    # can't claim while it's stale & present
    assert reporting.is_claimed(db, cfg, "task")

    monkeypatch.setattr(time, "time", lambda: NOW)
    assert reporting.clear_stale_claims(db) == 1
    # now slot is free again
    assert not reporting.is_claimed(db, cfg, "task")
    assert reporting.claim_run(db, cfg, "task")


def test_mixed_rows(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    _insert_claim(db, cfg, "task1", age_h=10)
    _insert_claim(db, cfg, "task2", age_h=90)

    monkeypatch.setattr(time, "time", lambda: NOW)
    assert reporting.clear_stale_claims(db, max_age_hours=72) == 1
    assert reporting.is_claimed(db, cfg, "task1")
    assert not reporting.is_claimed(db, cfg, "task2")


@pytest.mark.parametrize("hrs", [71.99, 72.0])
def test_exact_cutoff_not_deleted(tmp_path, monkeypatch, hrs):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    _insert_claim(db, cfg, "task", age_h=hrs)

    monkeypatch.setattr(time, "time", lambda: NOW)
    assert reporting.clear_stale_claims(db) == 0
    assert reporting.is_claimed(db, cfg, "task")


def test_custom_threshold(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    _insert_claim(db, cfg, "task", age_h=50)

    monkeypatch.setattr(time, "time", lambda: NOW)
    assert reporting.clear_stale_claims(db, max_age_hours=45) == 1


@pytest.mark.parametrize("bad", [0, -5])
def test_bad_threshold_raises(tmp_path, bad):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    with pytest.raises(ValueError):
        reporting.clear_stale_claims(db, max_age_hours=bad)


def test_idempotent(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    _insert_claim(db, cfg, "task", age_h=150)

    monkeypatch.setattr(time, "time", lambda: NOW)
    assert reporting.clear_stale_claims(db) == 1
    # second pass should remove nothing
    assert reporting.clear_stale_claims(db) == 0


def test_rowcount_exact(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    _insert_claim(db, cfg, "task1", age_h=120)
    _insert_claim(db, cfg, "task2", age_h=130)

    monkeypatch.setattr(time, "time", lambda: NOW)
    assert reporting.clear_stale_claims(db) == 2


def test_future_schema_extra_column(tmp_path, monkeypatch):
    # add extra column to ensure helper ignores unknown columns
    cfg = make_cfg(tmp_path)
    db = reporting.get_db(cfg)
    db.execute("ALTER TABLE runs ADD COLUMN note TEXT")
    _insert_claim(db, cfg, "task", age_h=100)

    monkeypatch.setattr(time, "time", lambda: NOW)
    assert reporting.clear_stale_claims(db) == 1
