import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import metrics


# Override poisonous species for deterministic testing
@pytest.fixture(autouse=True)
def stub_poison(monkeypatch):
    monkeypatch.setattr(metrics, "POISONOUS_SPECIES", np.array([1, 3]), raising=True)


def test_classification_error_all_correct():
    y = np.array([0, 1, 2, 3])
    y_pred = y.copy()
    assert metrics.classification_error(y, y_pred) == 0.0
    # with unknown costs both = 1 => same
    assert metrics.classification_error_with_unknown(y, y_pred) == 0.0


def test_classification_error_all_wrong():
    y = np.array([0, 1, 2, 3])
    y_pred = np.array([4, 5, 6, 7])
    # all mismatches, cost_unknown_mis=1, cost_mis_as_unknown=1
    assert metrics.classification_error(y, y_pred) == 1.0
    # no unknowns => same
    assert metrics.classification_error_with_unknown(y, y_pred) == 1.0


def test_classification_error_with_unknown_edge():
    y_true = np.array([-1, 10])
    y_pred = np.array([5, -1])
    # one true unknown mispredicted (cost 10), one true known predicted unknown (cost 0.1)
    ce_unk = metrics.classification_error_with_unknown(
        y_true, y_pred, cost_unknown_mis=10.0, cost_mis_as_unknown=0.1
    )
    # (10 + 0.1) / 2 = 5.05
    assert pytest.approx(ce_unk, abs=1e-6) == 5.05
    # classification_error uses cost=1,1 => (1 + 1)/2 = 1
    assert metrics.classification_error(y_true, y_pred) == 1.0


def test_num_psc_decisions_and_num_esc_decisions():
    # with stubbed POISONOUS_SPECIES = [1,3]
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 3, 4, 1])
    # true poison at positions 0 and 2; pred poison when pred in [1,3]
    # pos 0: true poison, pred=2 not poison => PSC
    # pos 2: true poison, pred=4 not poison => PSC
    assert metrics.num_psc_decisions(y_true, y_pred) == 2
    # ESC: non-poison predicted as poison
    # pos1: true=2 not poison, pred=3 poison => ESC
    # pos3: true=4 not poison, pred=1 poison => ESC
    assert metrics.num_esc_decisions(y_true, y_pred) == 2


def test_psc_esc_cost_score_default_costs():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 3, 4, 1])
    # PSC=2, ESC=2, cost_psc=100, cost_esc=1 => (200 + 2)/4 = 50.5
    cost = metrics.psc_esc_cost_score(y_true, y_pred)
    assert pytest.approx(cost, abs=1e-6) == 50.5


def test_evaluate_metrics_basic():
    # simple balanced case
    y_true = np.array([0, 1, 2, 3])
    y_pred = y_true.copy()
    res = metrics.evaluate_metrics(y_true, y_pred)
    assert res["f1_macro"] == 100.0
    assert res["ce"] == 0.0
    assert res["psc_esc_cost"] == 0.0
    assert res["user_loss"] == 0.0
    assert res["ce_unk"] == 0.0


def test_all_correct():
    y = np.arange(5)
    y_pred = y.copy()
    assert metrics.classification_error(y, y_pred) == 0.0


def test_all_wrong():
    y = np.arange(5)
    y_pred = np.arange(5) + 10
    assert metrics.classification_error(y, y_pred) == 1.0


@given(
    y_true=st.lists(st.integers(min_value=-1, max_value=10), min_size=1, max_size=20),
    y_pred=st.lists(st.integers(min_value=-1, max_value=10), min_size=1, max_size=20),
)
def test_fuzz_raises_or_returns(y_true, y_pred):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    try:
        out = metrics.evaluate_metrics(y_t, y_p)
        # Expect dict keys and numeric values
        assert isinstance(out, dict)
        for k, v in out.items():
            assert isinstance(k, str)
            assert isinstance(v, float)
    except ValueError:
        # mismatch lengths should raise
        assert y_t.shape != y_p.shape
