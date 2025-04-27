"""
Implements normalized nearest-centroid classifiers, as described in [this paper](https://arxiv.org/abs/1911.04623).

If you use this work, be sure to cite the original work:

```
@article{wang2019simpleshot,
  title={Simpleshot: Revisiting nearest-neighbor classification for few-shot learning},
  author={Wang, Yan and Chao, Wei-Lun and Weinberger, Kilian Q and Van Der Maaten, Laurens},
  journal={arXiv preprint arXiv:1911.04623},
  year={2019}
}
```
"""

import beartype
import numpy as np
import sklearn.base
import sklearn.neighbors
import sklearn.utils.validation
import torch
from jaxtyping import Float, jaxtyped

from . import helpers


@beartype.beartype
class SimpleShotClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """
    scikit-learn wrapper for the "normalized nearest-centroid" classifier
    (a.k.a. SimpleShot, Wang et al. ICCV 2019).

    Parameters
    ----------
    device : {'cpu','cuda'} or torch.device, default='cpu'
        Used only during `predict`; centroids are pushed to this device for fast batched distance computation.
    """

    def __init__(self, batch_size: int = 2048, device: str | torch.device = "cpu"):
        self.batch_size = batch_size
        self.device = torch.device(device)

    def fit(self, X, y):
        x, y = sklearn.utils.validation.check_X_y(X, y, dtype=np.float32, order="C")
        # centre the cloud
        self.x_mean_ = x.mean(axis=0, keepdims=True)

        x = x - self.x_mean_
        x = l2_normalize(x)

        self.clf_ = sklearn.neighbors.NearestCentroid()
        self.clf_.fit(x, y)
        return self

    def predict(self, X):
        sklearn.utils.validation.check_is_fitted(self, ["clf_", "x_mean_"])
        x = sklearn.utils.validation.check_array(X, dtype=np.float32, order="C")

        x = x - self.x_mean_
        x = l2_normalize(x)

        # Do this next step on the GPU to make it fast.
        # Goes from 1 batch/sec to 77 batch/sec
        centroids = torch.from_numpy(self.clf_.centroids_).to(self.device)
        x = x.to(self.device)

        preds = []
        for start, stop in helpers.batched_idx(len(x), self.batch_size):
            x_batch = x[start:stop]
            distances = torch.linalg.vector_norm(x_batch[:, None] - centroids, axis=2)
            preds.append(torch.argmin(distances, dim=1))

        return np.concatenate(preds, dim=0)


@jaxtyped(typechecker=beartype.beartype)
def l2_normalize(
    features: Float[np.ndarray, "n_examples dim"],
) -> Float[np.ndarray, "n_examples dim"]:
    """L2-normalize a batch of features.

    Args:
        features: batch of $d$-dimensional vectors.

    Returns:
        batch of $d$-dimensional vectors with unit L2 norm.
    """
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / norms
