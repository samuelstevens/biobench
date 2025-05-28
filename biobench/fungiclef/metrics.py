"""
FungiCLEF2023 custom metrics implementation.
Adapted from BohemianVRA's evaluate.py:
https://github.com/BohemianVRA/FGVC-Competitions/blob/main/FungiCLEF2023/evaluate.py
"""

import beartype
import numpy as np
import polars as pl
import sklearn.metrics
from jaxtyping import Float, Int, jaxtyped

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


@jaxtyped(typechecker=beartype.beartype)
def classification_error_with_unknown(
    y_true: Int[np.ndarray, "*batch n"],
    y_pred: Int[np.ndarray, "*batch n"],
    *,
    cost_unknown_mis: float = 10.0,
    cost_mis_as_unknown: float = 0.1,
) -> Float[np.ndarray, "*batch"]:
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
    *b, n = y_true.shape
    is_true_unknown = y_true == -1
    is_pred_unknown = y_pred == -1

    n_mis_unknown = np.sum(is_true_unknown & ~is_pred_unknown, axis=-1)
    n_mis_as_unknown = np.sum(~is_true_unknown & is_pred_unknown, axis=-1)
    n_other = np.sum((y_true != y_pred) & ~is_true_unknown & ~is_pred_unknown, axis=-1)

    err = (
        n_other
        + cost_unknown_mis * n_mis_unknown
        + cost_mis_as_unknown * n_mis_as_unknown
    ) / n

    return np.asarray(err)


@jaxtyped(typechecker=beartype.beartype)
def classification_error(
    y_true: Int[np.ndarray, "*batch n"], y_pred: Int[np.ndarray, "*batch n"]
) -> Float[np.ndarray, "*batch"]:
    """
    Standard classification error (unknown treated as error equally).
    """
    return classification_error_with_unknown(
        y_true, y_pred, cost_unknown_mis=1.0, cost_mis_as_unknown=1.0
    )


@beartype.beartype
def num_psc_decisions(
    y_true: Int[np.ndarray, "*batch n"], y_pred: Int[np.ndarray, "*batch n"]
) -> Int[np.ndarray, "*batch"]:
    """
    Number of poisonous species incorrectly predicted as non-poisonous.
    """
    is_true_poison = np.isin(y_true, POISONOUS_SPECIES)
    is_pred_poison = np.isin(y_pred, POISONOUS_SPECIES)
    return np.asarray(np.sum(is_true_poison & ~is_pred_poison, axis=-1))


@beartype.beartype
def num_esc_decisions(
    y_true: Int[np.ndarray, "*batch n"], y_pred: Int[np.ndarray, "*batch n"]
) -> Int[np.ndarray, "*batch"]:
    """
    Number of non-poisonous species incorrectly predicted as poisonous.
    """
    is_true_poison = np.isin(y_true, POISONOUS_SPECIES)
    is_pred_poison = np.isin(y_pred, POISONOUS_SPECIES)
    return np.asarray(np.sum(~is_true_poison & is_pred_poison, axis=-1))


@jaxtyped(typechecker=beartype.beartype)
def psc_esc_cost_score(
    y_true: Int[np.ndarray, "*batch n"],
    y_pred: Int[np.ndarray, "*batch n"],
    *,
    cost_psc: float = 100.0,
    cost_esc: float = 1.0,
) -> Float[np.ndarray, "*batch"]:
    """
    Weighted cost for poisonousness confusion per sample.
    """
    *batch, n = y_true.shape
    psc = num_psc_decisions(y_true, y_pred)
    esc = num_esc_decisions(y_true, y_pred)
    cost = (cost_psc * psc + cost_esc * esc) / n
    return np.asarray(cost)


@jaxtyped(typechecker=beartype.beartype)
def user_loss_score(
    y_true: Int[np.ndarray, "*batch n"], y_pred: Int[np.ndarray, "*batch n"]
) -> Float[np.ndarray, "*batch"]:
    ce_unk = classification_error_with_unknown(
        y_true, y_pred, cost_unknown_mis=10.0, cost_mis_as_unknown=0.1
    )
    psc = psc_esc_cost_score(y_true, y_pred, cost_psc=100.0, cost_esc=1.0)
    return np.asarray(ce_unk + psc)


@jaxtyped(typechecker=beartype.beartype)
def user_loss_score_normalized(
    y_true: Int[np.ndarray, "*batch n"], y_pred: Int[np.ndarray, "*batch n"]
) -> Float[np.ndarray, "*batch"]:
    cost_unknown_mis = 10.0
    cost_mis_as_unknown = 0.1
    ce_unk = classification_error_with_unknown(
        y_true,
        y_pred,
        cost_unknown_mis=cost_unknown_mis,
        cost_mis_as_unknown=cost_mis_as_unknown,
    )
    cost_psc = 100.0
    cost_esc = 1.0
    psc = psc_esc_cost_score(y_true, y_pred, cost_psc=cost_psc, cost_esc=cost_esc)

    n = y_true.size
    n_unknown = (y_true == -1).sum()
    n_poisonous = np.isin(y_true, POISONOUS_SPECIES).sum()

    # TODO: is this 1 supposed to be 0.1?
    worst_ce = (cost_unknown_mis - 1) * n_unknown / n + 1
    worst_psc = (cost_psc * n_poisonous + cost_esc * (n - n_poisonous)) / n

    score = 1 - (ce_unk + psc) / (worst_ce + worst_psc)
    return np.asarray(score)


@jaxtyped(typechecker=beartype.beartype)
def evaluate_metrics(
    y_true: Int[np.ndarray, "*batch n"], y_pred: Int[np.ndarray, "*batch n"]
) -> dict[str, Float[np.ndarray, "*batch"]]:
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
    f1 = np.asarray(sklearn.metrics.f1_score(y_true, y_pred, average="macro") * 100.0)
    ce = classification_error(y_true, y_pred)
    psc = psc_esc_cost_score(y_true, y_pred)
    ce_unk = classification_error_with_unknown(y_true, y_pred)
    user_loss = np.asarray(ce_unk + psc)
    return {
        "f1_macro": f1,
        "ce": ce,
        "psc_esc_cost": psc,
        "user_loss": user_loss,
        "ce_unk": ce_unk,
    }
