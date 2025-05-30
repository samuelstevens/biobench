import collections

import beartype
import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from jaxtyping import Int, jaxtyped

from . import helpers


@st.composite
def _labels_and_n(draw):
    """
    Helper strategy: (labels array, n) with n <= len(labels) and n >= #classes
    """
    labels_list = draw(
        st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=300)
    )
    labels = np.array(labels_list, dtype=int)
    n_classes = len(np.unique(labels))
    # choose n in [n_classes, len(labels)]
    n = draw(st.integers(min_value=n_classes, max_value=len(labels)))
    return labels, n


@jaxtyped(typechecker=beartype.beartype)
def _measure_balance(
    labels: Int[np.ndarray, " n_labels"], indices: Int[np.ndarray, " n"]
) -> float:
    """
    Calculate a balance metric (coefficient of variation, lower is better) for the selected samples (labels[indices]).

    Returns 0 for perfect balance, higher for more imbalance.
    """
    if len(indices) == 0:
        return 0.0

    # Get the distribution of classes in the selected samples
    selected_labels = labels[indices]
    class_counts = collections.Counter(selected_labels)

    # Get all unique classes in the original dataset
    all_classes = set(labels)

    # Check if it was possible to include at least one of each class but didn't
    if len(indices) >= len(all_classes) and len(class_counts) < len(all_classes):
        return float("inf")

    # Calculate coefficient of variation (standard deviation / mean)
    counts = np.array(list(class_counts.values()))

    # If only one class is present, return a high value to indicate imbalance
    if len(counts) == 1:
        return float("inf")

    mean = np.mean(counts)
    std = np.std(counts, ddof=1)  # Using sample standard deviation

    # Return coefficient of variation (0 for perfect balance)
    return std / mean if mean > 0 else 0.0


@given(
    st.text(
        # printable ASCII (includes '/' ':' we want to scrub)
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        max_size=100,
    )
)
def test_fs_safe_idempotent_and_clean(random_text):
    cleaned = helpers.fs_safe(random_text)

    # 1) Forbidden characters removed
    assert ":" not in cleaned and "/" not in cleaned

    # 2) Idempotent
    assert helpers.fs_safe(cleaned) == cleaned


@given(
    total=st.integers(min_value=0, max_value=1_000),
    batch=st.integers(min_value=1, max_value=400),
)
def test_batched_idx_covers_range_without_overlap(total, batch):
    """batched_idx must partition [0,total) into consecutive, non-overlapping spans, each of length <= batch."""
    spans = list(helpers.batched_idx(total, batch))

    # edge-case: nothing to iterate
    if total == 0:
        assert spans == []
        return

    # verify each span and overall coverage
    covered = []
    expected_start = 0
    for start, stop in spans:
        # bounds & width checks
        assert 0 <= start < stop <= total
        assert (stop - start) <= batch
        # consecutiveness (no gaps/overlap)
        assert start == expected_start
        expected_start = stop
        covered.extend(range(start, stop))

    # spans collectively cover exactly [0, total)
    assert covered == list(range(total))


@given(
    labels=npst.arrays(
        dtype=np.int32,
        shape=st.integers(min_value=1000, max_value=10000),
        elements=st.integers(min_value=0, max_value=100),
    ),
    n=st.integers(min_value=10, max_value=1000),
)
def test_correct_sample_size(labels, n):
    """Test that the function returns exactly n samples (or all if n > len(labels))"""
    assume(len(np.unique(labels)) > 1)  # Ensure we have at least 2 classes

    indices = helpers.balanced_random_sample(labels, n)

    # Check that the number of samples is correct
    n_expected = min(n, len(labels))
    assert len(indices) == n_expected

    # Check that all indices are valid
    assert np.all(indices < len(labels)), "Some indices are out of bounds"

    # Check that there are no duplicate indices
    assert len(indices) == len(np.unique(indices)), "Duplicate indices found"


# Test case 2: Class balance property
@given(
    labels=npst.arrays(
        dtype=np.int32,
        shape=st.integers(min_value=1000, max_value=10000),
        elements=st.integers(min_value=0, max_value=20),
    ),
    n=st.integers(min_value=100, max_value=1000),
)
def test_class_balance(labels, n):
    """
    Test that the class distribution in the sample is more balanced than random sampling would be.
    """
    unique_classes = np.unique(labels)
    assume(len(unique_classes) > 1)  # Ensure we have at least 2 classes
    assume(n >= len(unique_classes))  # Ensure we request at least one sample per class

    # Get balanced samples
    balanced_indices = helpers.balanced_random_sample(labels, n)
    balanced_balance = _measure_balance(labels, balanced_indices)

    # Get a normal random sample for comparison
    random_indices = np.random.choice(len(labels), min(n, len(labels)), replace=False)
    random_balance = _measure_balance(labels, random_indices)

    # Check if our balanced sampling is generally better than random
    # Note: This might occasionally fail due to randomness, but should pass most of the time
    assert balanced_balance <= random_balance * 1.5, (
        f"Balance metric: balanced={balanced_balance}, random={random_balance}"
    )


def test_single_class_sampling():
    """Test sampling when all samples are from the same class"""
    labels = np.array([1, 1, 1, 1, 1], dtype=int)
    indices = helpers.balanced_random_sample(labels, 3)
    assert len(indices) == 3
    assert len(np.unique(indices)) == 3


def test_small_sample_size():
    """Test sampling with a very small n"""
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
    indices = helpers.balanced_random_sample(labels, 2)
    assert len(indices) == 2


def test_sample_size_larger_than_dataset():
    """Test when requested sample size exceeds dataset size"""
    labels = np.array([0, 1, 2], dtype=int)
    indices = helpers.balanced_random_sample(labels, 10)
    assert len(indices) == 3  # Should return all samples
    assert set(indices) == {0, 1, 2}


def test_empty_dataset():
    """Test sampling from an empty dataset"""
    labels = np.array([], dtype=int)
    indices = helpers.balanced_random_sample(labels, 5)
    assert len(indices) == 0


def test_zero_samples_requested():
    """Test when zero samples are requested"""
    labels = np.array([0, 1, 2, 3], dtype=int)
    indices = helpers.balanced_random_sample(labels, 0)
    assert len(indices) == 0


@given(_labels_and_n())
def test_balanced_random_sample_includes_each_class_when_possible(data):
    labels, n = data
    idx = helpers.balanced_random_sample(labels, n)
    assert set(labels[idx]) == set(np.unique(labels))


@given(
    labels=st.lists(st.integers(min_value=0, max_value=100), min_size=10, max_size=500),
    n=st.integers(min_value=1, max_value=500),
)
def test_balanced_random_sample_never_duplicates_indices(labels, n):
    """Returned index list must have no duplicates."""
    labels = np.array(labels, dtype=int)
    n = min(n, len(labels))  # cap n to dataset size

    idx = helpers.balanced_random_sample(labels, n)

    # uniqueness & length
    assert len(idx) == len(set(idx))
    assert len(idx) == n or len(idx) == len(labels)  # handles n>len(labels)
