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

import collections.abc

import beartype
import numpy as np
import sklearn.neighbors
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
def l2_normalize(
    features: Float[Tensor, "n_examples dim"],
) -> Float[Tensor, "n_examples dim"]:
    """L2-normalize a batch of features.

    Args:
        features: batch of $d$-dimensional vectors.

    Returns:
        batch of $d$-dimensional vectors with unit L2 norm.
    """
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / norms


@beartype.beartype
def batched_idx(
    total_size: int, batch_size: int
) -> collections.abc.Iterator[tuple[int, int]]:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """
    for start in range(0, total_size, batch_size):
        stop = min(start + batch_size, total_size)
        yield start, stop


@jaxtyped(typechecker=beartype.beartype)
def simpleshot(
    x_train: Float[Tensor, "n_train dim"],
    y_train: Int[Tensor, " n_train"],
    x_test: Float[Tensor, "n_test dim"],
    y_test: Int[Tensor, " n_test"],
    batch_size: int,
    device: str,
) -> Float[Tensor, " n_test"]:
    """
    Applies simpleshot to features. Returns the list of scores for x_test.

    Args:
        ...

    Returns:
        A tensor of 0/1 scores, one for each test example, only considering exact match.
    """
    x_mean = x_train.mean(axis=0, keepdims=True)

    x_train = x_train - x_mean
    x_train = l2_normalize(x_train)

    x_test = x_test - x_mean
    x_test = l2_normalize(x_test)

    clf = sklearn.neighbors.NearestCentroid()
    clf.fit(x_train, y_train)

    # Do this next step on the GPU to make it fast.
    # Goes from 1 batch/sec to 77 batch/sec
    centroids = torch.from_numpy(clf.centroids_).to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    scores = []
    for start, stop in batched_idx(len(x_test), batch_size):
        x_batch = x_test[start:stop]
        y_batch = y_test[start:stop]
        distances = torch.linalg.vector_norm(x_batch[:, None] - centroids, axis=2)
        preds = torch.argmin(distances, dim=1)

        scores.append((preds == y_batch).type(torch.float32))

    return torch.cat(scores, axis=0)
