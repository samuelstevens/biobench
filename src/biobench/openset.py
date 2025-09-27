import logging

import beartype
import numpy as np
import sklearn.base
import sklearn.discriminant_analysis
import sklearn.utils.validation
from jaxtyping import Float, jaxtyped

from . import helpers

logger = logging.getLogger(__name__)


@beartype.beartype
class MahalanobisOpenSetClassifier(
    sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin
):
    """
    Wraps an arbitrary scikit-learn multiclass estimator with a Mahalanobis out-of-distribution detector.  Unknown samples are assigned `unknown_label`.

    @inproceedings{lee2018simple,
      title = {A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks},
      author = {Lee, Kimin and Lee, Kibok and Lee, Honglak and Shin, Jinwoo},
      year = 2018,
      booktitle = {Advances in Neural Information Processing Systems},
      publisher = {Curran Associates, Inc.},
      volume = 31,
      pages = {},
      url = {https://proceedings.neurips.cc/paper_files/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf},
      editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett}
    }


    Parameters
    ----------
    base_estimator : scikit-learn estimator
        Must implement fit / predict (e.g. a Pipeline with a classifier).
    alpha : float, default=0.95
        Confidence level for the chi-squared cutoff; 1-alpha is the tail mass declared OOD.
    unknown_label : int | str, default=-1
        Label given to detections outside the known set.
    """

    def __init__(
        self, base_estimator, alpha: float = 0.95, unknown_label: int | str = -1
    ):
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.unknown_label = unknown_label

    # ---------------- #
    # scikit-learn API #
    # ---------------- #
    def fit(self, X, y):
        X, y = sklearn.utils.validation.check_X_y(X, y, accept_sparse=False)
        self.classes_ = np.unique(y)

        self.clf_ = sklearn.base.clone(self.base_estimator).fit(X, y)
        logger.info("Fit base estimator.")

        self.lda_ = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", store_covariance=True
        )
        self.lda_.fit(X, y)
        logger.info("Fit LDA.")

        self.means_ = self.lda_.means_
        self.covariance_ = self.lda_.covariance_
        try:
            self.inv_covariance_ = np.linalg.inv(self.covariance_)
        except np.linalg.LinAlgError:
            self.inv_covariance_ = np.linalg.pinv(self.covariance_)
        logger.info("Inverted covariance matrix.")

        train_scores = min_mahalanobis_sq_batched(X, self.means_, self.inv_covariance_)
        self.tpr_ = np.percentile(train_scores, 95)

        logger.info("Fit.")
        return self

    def predict(self, X):
        scores = self.decision_function(X)

        pred_known = self.clf_.predict(X)
        pred = np.where(scores >= 0, pred_known, self.unknown_label)
        return pred

    def decision_function(self, X):
        sklearn.utils.validation.check_is_fitted(self, "clf_")
        X = sklearn.utils.validation.check_array(X, accept_sparse=False)

        d2 = min_mahalanobis_sq_batched(X, self.means_, self.inv_covariance_)
        return self.tpr_ - d2


@jaxtyped(typechecker=beartype.beartype)
def min_mahalanobis_sq(
    X: Float[np.ndarray, "n d"],
    mu: Float[np.ndarray, "classes d"],
    cov_inv: Float[np.ndarray, "d d"],
) -> Float[np.ndarray, " n"]:
    logger.info(
        "Calculating min_mahalanobis_sq. n=%d, d=%d, c=%d",
        X.shape[0],
        X.shape[1],
        mu.shape[0],
    )
    diff = X[:, None, :] - mu[None, :, :]
    logger.info("Got diff with shape %s", diff.shape)
    d2 = np.einsum("ncd,dd,ncd->nc", diff, cov_inv, diff)
    logger.info("Got d2 with shape %s", d2.shape)
    out = d2.min(axis=1)
    logger.info("Got out with shape %s", out.shape)
    return out.astype(np.float32)


@jaxtyped(typechecker=beartype.beartype)
def min_mahalanobis_sq_batched(
    X: Float[np.ndarray, "n d"],
    mu: Float[np.ndarray, "classes d"],
    cov_inv: Float[np.ndarray, "d d"],
    *,
    bsz: int = 256,
) -> Float[np.ndarray, " n"]:
    n, d = X.shape
    L = np.linalg.cholesky(cov_inv)
    logger.info("Calculated L.")

    Xw = X @ L  # (n, d)  whiten once
    muw = mu @ L  # (c, d)
    muw_norm_sq = np.square(muw).sum(axis=1).reshape(1, -1)  # (1, c)

    out = np.full(n, 0.0, dtype=np.float32)

    for start, end in helpers.progress(helpers.batched_idx(n, bsz), desc="mahalanobis"):
        Xwb = Xw[start:end]  # (bsz, d)
        Xwb_norm_sq = np.square(Xwb).sum(axis=1, keepdims=True)  # (bsz, 1)
        d_sq = Xwb_norm_sq + muw_norm_sq - 2.0 * Xwb @ muw.T  # (bsz,C)
        out[start:end] = d_sq.min(axis=1)

    return out.astype(np.float32)
