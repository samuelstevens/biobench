from collections import Counter

import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from . import helpers


# Helper functions for testing
def get_class_distribution(labels, indices):
    """Get distribution of classes in the selected samples"""
    if len(indices) == 0:
        return {}
    selected_labels = labels[indices]
    return Counter(selected_labels)


def get_balance_metric(distribution):
    """
    Calculate a balance metric: coefficient of variation (lower is better)
    Returns 0 for perfect balance, higher for more imbalance
    """
    if not distribution:
        return 0
    values = list(distribution.values())
    if len(values) <= 1:
        return 0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance**0.5
    return std_dev / mean


# Test case 1: Correct number of samples
@given(
    labels=npst.arrays(
        dtype=np.int32,
        shape=st.integers(min_value=1000, max_value=10000),
        elements=st.integers(min_value=0, max_value=100),
    ),
    n=st.integers(min_value=10, max_value=1000),
)
@settings(max_examples=20)
def test_correct_sample_size(labels, n):
    """Test that the function returns exactly n samples (or all if n > len(labels))"""
    assume(len(np.unique(labels)) > 1)  # Ensure we have at least 2 classes

    indices = helpers.balanced_random_sample(labels, n)

    # Check that the number of samples is correct
    expected_samples = min(n, len(labels))
    assert len(indices) == expected_samples, (
        f"Expected {expected_samples} samples, got {len(indices)}"
    )

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
@settings(max_examples=20)
def test_class_balance(labels, n):
    """
    Test that the class distribution in the sample is more balanced
    than random sampling would be
    """
    unique_classes = np.unique(labels)
    assume(len(unique_classes) > 1)  # Ensure we have at least 2 classes
    assume(n >= len(unique_classes))  # Ensure we request at least one sample per class

    # Get balanced samples
    balanced_indices = helpers.balanced_random_sample(labels, n)
    balanced_dist = get_class_distribution(labels, balanced_indices)

    # Get a normal random sample for comparison
    random_indices = np.random.choice(len(labels), min(n, len(labels)), replace=False)
    random_dist = get_class_distribution(labels, random_indices)

    # Calculate balance metrics (lower is better)
    balanced_balance = get_balance_metric(balanced_dist)
    random_balance = get_balance_metric(random_dist)

    # Check if our balanced sampling is generally better than random
    # Note: This might occasionally fail due to randomness, but should pass most of the time
    assert balanced_balance <= random_balance * 1.5, (
        f"Balance metric: balanced={balanced_balance}, random={random_balance}"
    )


# Test case 3: Edge cases
def test_edge_cases():
    """Test the function with various edge cases"""

    # Case 1: All samples from the same class
    labels = np.array([1, 1, 1, 1, 1])
    indices = helpers.balanced_random_sample(labels, 3)
    assert len(indices) == 3
    assert len(np.unique(indices)) == 3

    # Case 2: Very imbalanced dataset
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    indices = helpers.balanced_random_sample(labels, 4)
    distribution = get_class_distribution(labels, indices)
    # Should have at least one of each class if possible
    assert 0 in distribution and 1 in distribution

    # Case 3: Very small n
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    indices = helpers.balanced_random_sample(labels, 2)
    assert len(indices) == 2

    # Case 4: n larger than dataset
    labels = np.array([0, 1, 2])
    indices = helpers.balanced_random_sample(labels, 10)
    assert len(indices) == 3  # Should return all samples
    assert set(indices) == {0, 1, 2}

    # Case 5: Empty array
    labels = np.array([])
    indices = helpers.balanced_random_sample(labels, 5)
    assert len(indices) == 0

    # Case 6: Request 0 samples
    labels = np.array([0, 1, 2, 3])
    indices = helpers.balanced_random_sample(labels, 0)
    assert len(indices) == 0

    # Separate this into separate, well-named test functions. AI!
