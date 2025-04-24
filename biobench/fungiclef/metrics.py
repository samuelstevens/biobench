"""
FungiCLEF2023 custom metrics implementation.
Adapted from BohemianVRA's evaluate.py:
https://github.com/BohemianVRA/FGVC-Competitions/blob/main/FungiCLEF2023/evaluate.py
"""

import beartype
import numpy as np
import polars as pl
import sklearn.metrics

# Load poison status via Polars
_poison_df = pl.read_csv(
    "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/poison_status_list.csv"
)
POISONOUS_SPECIES = (
    _poison_df.filter(pl.col("poisonous") == 1)
    .select("class_id")
    .unique()
    .to_series()
    .to_numpy()
)


@beartype.beartype
def classification_error_with_unknown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_unknown_mis: float = 10.0,
    cost_mis_as_unknown: float = 0.1,
) -> float:
    """
    Classification error allowing for an "unknown" class (encoded as -1).

    Args:
        y_true: ground-truth labels (with -1 for unknown)
        y_pred: predicted labels (with -1 for unknown)
        cost_unknown_mis: cost of misclassifying a true unknown as known
        cost_mis_as_unknown: cost of misclassifying a true known as unknown

    Returns:
        normalized error rate
    """
    is_true_unknown = y_true == -1
    is_pred_unknown = y_pred == -1

    num_mis_unknown = np.sum(is_true_unknown & ~is_pred_unknown)
    num_mis_as_unknown = np.sum(~is_true_unknown & is_pred_unknown)
    num_other = np.sum((y_true != y_pred) & ~is_true_unknown & ~is_pred_unknown)
    total = y_true.size
    return (
        num_other
        + cost_unknown_mis * num_mis_unknown
        + cost_mis_as_unknown * num_mis_as_unknown
    ) / total


@beartype.beartype
def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard classification error (unknown treated as error equally).
    """
    return classification_error_with_unknown(
        y_true, y_pred, cost_unknown_mis=1.0, cost_mis_as_unknown=1.0
    )


@beartype.beartype
def num_psc_decisions(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Number of poisonous species incorrectly predicted as non-poisonous.
    """
    is_true_poison = np.isin(y_true, POISONOUS_SPECIES)
    is_pred_poison = np.isin(y_pred, POISONOUS_SPECIES)
    return int(np.sum(is_true_poison & ~is_pred_poison))


@beartype.beartype
def num_esc_decisions(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Number of non-poisonous species incorrectly predicted as poisonous.
    """
    is_true_poison = np.isin(y_true, POISONOUS_SPECIES)
    is_pred_poison = np.isin(y_pred, POISONOUS_SPECIES)
    return int(np.sum(~is_true_poison & is_pred_poison))


@beartype.beartype
def psc_esc_cost_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_psc: float = 100.0,
    cost_esc: float = 1.0,
) -> float:
    """
    Weighted cost for poisonousness confusion per sample.
    """
    total = y_true.size
    psc = num_psc_decisions(y_true, y_pred)
    esc = num_esc_decisions(y_true, y_pred)
    return (cost_psc * psc + cost_esc * esc) / total


@beartype.beartype
def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute all four FungiCLEF custom metrics plus macro F1.

    Returns:
        dict with keys:
            - F1_macro
            - Classification_Error
            - PSC_ESC_Cost
            - User_Focused_Loss
            - Classification_Error_with_Unknown
    """
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro") * 100.0
    ce = classification_error(y_true, y_pred)
    psc = psc_esc_cost_score(y_true, y_pred)
    ce_unk = classification_error_with_unknown(y_true, y_pred)
    user_loss = ce + psc
    return {
        "f1_macro": f1,
        "ce": ce,
        "psc_esc_cost": psc,
        "user_loss": user_loss,
        "ce_unk": ce_unk,
    }
