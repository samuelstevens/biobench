"""
Individual re-identification of African leopards (*Panthera pardus*) using [this LILA BC dataset](https://lila.science/datasets/leopard-id-2022/).

We use a simple but computationally expensive method, first proposed by Andrej Karpathy [in this notebook](https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb):

1. Embed all images using a vision backbone.
2. For each image, treat it as a test image and train a new SVC predicting the query image as positive and the other images as negative.
3. Choose the closest negative image based on the SVC's decision boundary as the returned image.
4. Give a score of 1.0 if this returned image is the same individual, otherwise 0.0.

Because this method requires training an SVM on every query image, it will train ~6800 SVMs.
However, this is an embarrassingly parallel task because none of the SVMs depend on each other.
We use the [joblib](https://joblib.readthedocs.io/en/stable/index.html) library and its `joblib.Parallel` class ([see this guide](https://joblib.readthedocs.io/en/stable/parallel.html)).

With 16 jobs, using multiprocessing, it takes about ~20 minutes on my lab's server.
With 16 jobs using *threading* it was predicted to take over 80 minutes.
I let it run for 10 minutes and it was stable in predicting 60+ minutes, so I settled on multiprocessing but with 24 jobs for more speed.
"""

import dataclasses
import logging
import os.path

import beartype
import joblib
import numpy as np
import sklearn.neighbors
import sklearn.preprocessing
import torch
import torchvision.datasets
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor

from .. import config, helpers, registry, reporting

logger = logging.getLogger("leopard")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Configuration for Leopard re-ID task."""

    batch_size: int = 256
    """Batch size for the vision backbone."""
    n_workers: int = 8
    """Number of dataloader workers."""
    log_every: int = 10
    """How often to log while getting features."""
    n_jobs: int = 16
    """How many SVMs to train in parallel."""
    # Computed at runtime.
    device: str = "cuda"
    """(computed at runtime) which kind of accelerator to use."""
    debug: bool = False
    """(computed at runtime) whether to run in debug mode."""
    n_train: int = -1
    """(computed at runtime) number of maximum training samples. Negative number means use all of them."""


@beartype.beartype
def benchmark(cfg: config.Experiment) -> tuple[config.Model, reporting.Report]:
    """
    Run the leopard re-ID benchmark. See this module's documentation for more details.
    """
    backbone = registry.load_vision_backbone(cfg.model)

    # Embed all images.
    features = get_features(cfg, backbone)
    # Convert string names into integer labels.
    encoder = sklearn.preprocessing.OrdinalEncoder(dtype=int)
    y = encoder.fit_transform(features.labels.reshape(-1, 1)).reshape(-1)

    @beartype.beartype
    def predict(i: int, image_id) -> reporting.Prediction:
        clf = sklearn.svm.LinearSVC(
            class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=0.1
        )
        svm_y = np.zeros(features.n)
        svm_y[i] = 1
        clf.fit(features.x, svm_y)
        sims = clf.decision_function(features.x)
        # The top result is always i, but we want the second-best result.
        pred_i = np.argsort(sims)[1]
        # TODO: we could also take the top k results and choose the most common.
        # Something like:
        #   pred_i = scipy.stats.mode(np.argsort(sims)[1:args.k+1]).mode

        example = reporting.Prediction(str(image_id), float(y[pred_i] == y[i]), {})
        return example

    examples = joblib.Parallel(n_jobs=cfg.n_jobs)(
        joblib.delayed(predict)(i, image_id)
        for i, image_id in enumerate(helpers.progress(features.ids, every=10))
    )

    return cfg.model, reporting.Report("LeopardID", examples)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    """
    A block of features.

    Note: In Jax, this could be a tuple of arrays, all with a leading dimension of `n`. Instead, in PyTorch, it's easier to make it its own class. Oh well.
    """

    x: Float[Tensor, "n dim"]
    """Input features; from a `biobench.registry.VisionBackbone`."""
    labels: Shaped[np.ndarray, " n"]
    """Individual name."""
    ids: Shaped[np.ndarray, " n"]
    """Array of image ids."""

    @property
    def n(self) -> int:
        return len(self.ids)


@beartype.beartype
@torch.no_grad
def get_features(args: Args, backbone: registry.VisionBackbone) -> Features:
    """
    Get a block of features from a vision backbone.

    Args:
        args: LeopardID arguments.
        backbone: visual backbone.
    """
    backbone_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    @jaxtyped(typechecker=beartype.beartype)
    def sample_transform(
        img, metadata: list[dict]
    ) -> tuple[Float[Tensor, "3 w h"], tuple[str, str]]:
        # tgt is always a list for some reason.
        metadata = metadata[0]
        x, y, w, h = metadata["bbox"]
        img = img.crop((x, y, x + w, y + h))
        return backbone_transform(img), (metadata["name"], str(metadata["image_id"]))

    if not os.path.isdir(args.datadir):
        msg = f"Path '{args.datadir}' doesn't exist. Did you download the leopard dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--leopard-args.datadir'; see --help for more."
        raise ValueError(msg)

    dataset = torchvision.datasets.CocoDetection(
        os.path.join(args.datadir, "leopard.coco", "images", "train2022"),
        os.path.join(
            args.datadir, "leopard.coco", "annotations", "instances_train2022.json"
        ),
        transforms=sample_transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
    )

    all_features, all_labels, all_ids = [], [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    for b in helpers.progress(range(total), every=args.log_every, desc="embed"):
        images, (labels, ids) = next(it)
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features

        all_features.append(features.cpu())
        all_labels.extend(labels)
        all_ids.extend(ids)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_labels = np.array(all_labels)
    all_ids = np.array(all_ids)

    return Features(all_features, all_labels, all_ids)
