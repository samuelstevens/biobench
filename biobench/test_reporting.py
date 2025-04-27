import math

from hypothesis import given
from hypothesis import strategies as st

from . import reporting


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


@given(preds=_prediction_list())
def test_micro_f1_equals_micro_accuracy(preds):
    """Micro-averaged F1 must equal micro accuracy for single-label data."""

    acc = reporting.micro_acc(preds)
    f1 = reporting.micro_f1(preds)

    # Floating math can introduce tiny error, so compare with tolerance
    assert math.isclose(acc, f1, rel_tol=1e-12, abs_tol=1e-12)
