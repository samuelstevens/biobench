"""Unit tests for `openset.MahalanobisOpenSetClassifier`."""

import numpy as np
import pytest
import sklearn.linear_model
from hypothesis import given, settings
from hypothesis import strategies as st

from . import openset

# --------#
# Helpers #
# --------#


def _simple_classifier():
    return openset.MahalanobisOpenSetClassifier(
        base_estimator=sklearn.linear_model.RidgeClassifier(),
        alpha=0.95,
        unknown_label=-1,
    )


def _toy_data():
    """Return a trivially separable 1-D two-class dataset."""
    x = np.array([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0]).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    return x, y


# ------------------ #
# Estimator API test #
# ------------------ #


def test_estimator_api():
    clf = _simple_classifier()
    x, y = _toy_data()

    # scikit-learn compliance: fit / predict / get_params / set_params
    assert hasattr(clf, "get_params") and hasattr(clf, "set_params")

    clf.fit(x, y)
    preds = clf.predict(x)

    # Attributes created by fit
    for attr in ("clf_", "means_", "inv_covariance_", "tpr_"):
        assert hasattr(clf, attr), f"missing attribute {attr} after fit()"

    # Perfect prediction on training data
    assert np.array_equal(preds, y)


# ----------------------------------------------- #
# Hypothesis: fuzz for exceptions (fit & predict) #
# ----------------------------------------------- #


@given(
    n_samples=st.integers(min_value=30, max_value=200),
    n_features=st.integers(min_value=2, max_value=60),
    n_classes=st.integers(min_value=2, max_value=40),
)
@settings(deadline=400)
def test_fuzz_no_exceptions(n_samples, n_features, n_classes):
    rng = np.random.default_rng()
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples)

    clf = _simple_classifier()
    clf.fit(x, y)  # should not raise
    preds = clf.predict(x)

    assert preds.shape[0] == n_samples


# -------------------------------- #
# Hand-written deterministic cases #
# -------------------------------- #


@pytest.mark.parametrize(
    "test_pt, expected",
    [
        (np.array([[-1.0]]), 0),  # near class-0 centroid
        (np.array([[1.2]]), 1),  # near class-1 centroid
        (np.array([[100.0]]), -1),  # far away -> unknown
        (np.array([[-50.0]]), -1),  # far other side -> unknown
    ],
)
def test_known_vs_unknown(test_pt, expected):
    x, y = _toy_data()
    clf = _simple_classifier().fit(x, y)
    pred = clf.predict(test_pt)[0]
    assert pred == expected


# --------------------------------------------------------------#
# Property-based tests that performance is *better than random* #
# --------------------------------------------------------------#


def _make_cluster_data(n_classes=3, pts_per_class=30, d=4, noise=0.05):
    """Generate tight Gaussian clusters plus a distant OOD cluster."""
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=3.0, size=(n_classes, d))

    x_known = []
    y_known = []
    for k, c in enumerate(centers):
        x_known.append(rng.normal(loc=c, scale=noise, size=(pts_per_class, d)))
        y_known.append(np.full(pts_per_class, k))
    x_known = np.vstack(x_known)
    y_known = np.concatenate(y_known)

    # OOD data around a far corner
    ood_center = np.full(d, 10.0)
    x_ood = rng.normal(loc=ood_center, scale=noise, size=(pts_per_class, d))
    y_ood = np.full(pts_per_class, -1)  # unknown label for evaluation

    return (x_known, y_known), (x_ood, y_ood)


@pytest.mark.parametrize("seed", [0, 42])
def test_high_known_accuracy(seed):
    (x_k, y_k), (x_ood, _) = _make_cluster_data()
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(x_k), size=60, replace=False)
    x_train, y_train = x_k[idx], y_k[idx]
    x_test = np.vstack([x_k, x_ood])
    y_test = np.concatenate([y_k, np.full(len(x_ood), -1, dtype=y_k.dtype)])

    clf = _simple_classifier().fit(x_train, y_train)
    preds = clf.predict(x_test)

    # Basic expectations:
    acc = (preds == y_test).mean()
    assert acc > 0.8, f"Expected >80 % overall accuracy, got {acc:.2f}"


def test_roc_like_property():
    (x_k, y_k), (x_ood, _) = _make_cluster_data(n_classes=2)
    x_train, y_train = x_k[:40], y_k[:40]
    clf = _simple_classifier().fit(x_train, y_train)

    scores = clf.decision_function(np.vstack([x_k, x_ood]))

    # Simple AUC approximation: score separation
    separation = scores[: len(x_k)].mean() - scores[len(x_k) :].mean()
    assert separation > 0.5, "Mahalanobis scores should separate ID from OOD"


@st.composite
def random_spd_inv(draw, d: int):
    """Random symmetric-positive-definite inverse covariance."""
    A = draw(
        st.lists(
            st.floats(-5, 5, allow_nan=False, allow_infinity=False),
            min_size=d * d,
            max_size=d * d,
        )
    )
    A = np.asarray(A, dtype=np.float64).reshape(d, d)
    cov = A @ A.T + 1e-3 * np.eye(d)
    return np.linalg.inv(cov)


@st.composite
def maha_inputs(draw):
    # dimensions
    n = draw(st.integers(min_value=1, max_value=64))
    d = draw(st.integers(min_value=1, max_value=32))
    C = draw(st.integers(min_value=1, max_value=16))

    # feature matrix
    X = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=n * d,
            max_size=n * d,
        )
    )
    X = np.asarray(X, dtype=np.float64).reshape(n, d)

    # class means
    mu = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=C * d,
            max_size=C * d,
        )
    )
    mu = np.asarray(mu, dtype=np.float64).reshape(C, d)

    # shared precision (inverse covariance)
    cov_inv = draw(random_spd_inv(d))

    return X, mu, cov_inv


@given(maha_inputs())
def test_min_mahalanobis_sq_no_error(data):
    X, mu, cov_inv = data
    out = openset.min_mahalanobis_sq(X, mu, cov_inv)

    # basic sanity: shape and finiteness
    assert out.shape == (X.shape[0],)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


@given(maha_inputs())
def test_min_mahalanobis_sq_batched_no_error(data):
    X, mu, cov_inv = data
    out = openset.min_mahalanobis_sq_batched(X, mu, cov_inv)

    # basic sanity: shape and finiteness
    assert out.shape == (X.shape[0],)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


@given(maha_inputs())
def test_min_mahalanobis_sq_batched_equal(data):
    X, mu, cov_inv = data
    expected = openset.min_mahalanobis_sq(X, mu, cov_inv)
    actual = openset.min_mahalanobis_sq_batched(X, mu, cov_inv)

    assert expected.shape == actual.shape
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
