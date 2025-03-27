import collections

import beartype
import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from . import helpers


@beartype.beartype
def measure_balance(labels, indices) -> float:
    """
    Calculate a balance metric (coefficient of variation, lower is better) for the selected samples (labels[indices]).

    Returns 0 for perfect balance, higher for more imbalance
    """
    # Fill this function in. AI!


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
    balanced_dist = get_class_distribution(labels, balanced_indices)

    # Get a normal random sample for comparison
    random_indices = np.random.choice(len(labels), min(n, len(labels)), replace=False)
    random_balance = measure_balance(labels, random_indices)

    # Calculate balance metrics (lower is better)
    balanced_balance = get_balance_metric(balanced_dist)
    random_balance = get_balance_metric(random_dist)

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


def test_highly_imbalanced_dataset():
    """Test sampling from a highly imbalanced dataset"""
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int)
    indices = helpers.balanced_random_sample(labels, 4)
    distribution = get_class_distribution(labels, indices)
    # Should have at least one of each class if possible
    assert 0 in distribution and 1 in distribution


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
