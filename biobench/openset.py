import beartype
import numpy as np
import scipy.stats
import sklearn.base
import sklearn.utils.validation


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
    covariance_estimator : {"empirical", "ledoit"}, default="ledoit"
        Strategy for covariance. "ledoit" is shrinkage-robust and invertible.
    unknown_label : int | str, default=-1
        Label given to detections outside the known set.
    """

    def __init__(
        self,
        base_estimator,
        alpha: float = 0.95,
        covariance_estimator: str = "ledoit",
        unknown_label: int | str = -1,
    ):
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.covariance_estimator = covariance_estimator
        self.unknown_label = unknown_label

    # ---------------- #
    # scikit-learn API #
    # ---------------- #
    def fit(self, X, y):
        X, y = sklearn.utils.validation.check_X_y(X, y, accept_sparse=False)
        self.classes_ = np.unique(y)

        # 1. fit the wrapped classifier
        self.clf_ = sklearn.base.clone(self.base_estimator).fit(X, y)

        # 2. compute per-class means
        self.means_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])

        # 3. shared covariance
        if self.covariance_estimator == "ledoit":
            cov = sklearn.covariance.LedoitWolf().fit(X)
            self.Sigma_inv_ = cov.precision_
        elif self.covariance_estimator == "empirical":
            cov = np.cov(X, rowvar=False)
            self.Sigma_inv_ = np.linalg.pinv(cov)
        else:
            raise ValueError("covariance_estimator must be 'empirical' or 'ledoit'")

        # 4. analytic chi-squared cutoff
        d = X.shape[1]
        self.tau_ = scipy.stats.chi2.ppf(self.alpha, df=d)
        return self

    def predict(self, X):
        sklearn.utils.validation.check_is_fitted(self, "clf_")
        X = sklearn.utils.validation.check_array(X, accept_sparse=False)

        # Mahalanobis distance to nearest class mean
        d2 = self._min_mahala_sq(X)
        is_in = d2 <= self.tau_

        pred_known = self.clf_.predict(X)
        pred = np.where(is_in, pred_known, self.unknown_label)
        return pred

    def decision_function(self, X):
        """Negative min-Mahalanobis distance (higher = more in-dist)."""
        sklearn.utils.validation.check_is_fitted(self, "clf_")
        X = sklearn.utils.validation.check_array(X, accept_sparse=False)
        return -self._min_mahala_sq(X)

    # ------------------ #
    # internal utilities #
    # ------------------ #
    def _min_mahala_sq(self, X) -> np.ndarray:
        """
        Vectorised squared Mahalanobis distance to the *nearest* class mean.
        """
        diff = X[:, None, :] - self.means_  # (n, C, d)
        # (n, C) -> min over C -> (n,)
        d2 = np.einsum("ncd,dd,ncd->nc", diff, self.Sigma_inv_, diff).min(axis=1)
        return d2
