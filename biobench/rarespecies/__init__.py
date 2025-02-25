import collections
import dataclasses
import logging
import math

import beartype
import numpy as np
import sklearn.neighbors
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped
from torch import Tensor

import datasets
from biobench import helpers, interfaces, registry

logger = logging.getLogger("rare-species")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    batch_size: int = 256
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of batches) to log progress."""
    # Computed at runtime.
    device: str = "cuda"
    """(computed at runtime) which kind of accelerator to use."""
    debug: bool = False
    """(computed at runtime) whether to run in debug mode."""
    n_train: int = -1
    """(computed at runtime) number of maximum training samples. Negative number means use all of them."""
    n_test: int = -1
    """(computed at runtime) number of test samples. Negative number means use all of them."""
    parallel: int = 1
    """(computed at runtime) number of parallel requests per second to MLLM service providers."""


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgsCvml
) -> tuple[interfaces.ModelArgsCvml, interfaces.TaskReport]:
    backbone = registry.load_vision_backbone(*model_args)
    features = get_features(args, backbone)

    train_i, test_i = make_split(features.y, k=1)

    scores = simpleshot(
        args,
        features.x[train_i],
        features.y[train_i],
        features.x[test_i],
        features.y[test_i],
    )
    examples = [
        interfaces.Prediction(str(id), float(score), {})
        for id, score in zip(features.ids[test_i], scores.tolist())
    ]
    return model_args, interfaces.TaskReport("RareSpecies", examples)


class LabelProcessor:
    def __init__(self):
        self._lookup = {}

    def transform(self, label) -> int:
        if label not in self._lookup:
            self._lookup[label] = len(self._lookup)
        return self._lookup[label]


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[Tensor, " n dim"]
    y: Int[Tensor, " n"]
    ids: Shaped[np.ndarray, " n"]


class Preprocess:
    def __init__(self, img_transform):
        self._img_transform = img_transform

    def __call__(self, example):
        example["image"] = example["image"].convert("RGB")
        example["image"] = self._img_transform(example["image"])
        example["label"] = "-".join(
            example[key]
            for key in [
                "kingdom",
                "phylum",
                "class",
                "order",
                "family",
                "genus",
                "species",
            ]
        )
        return example


@beartype.beartype
@torch.no_grad
def get_features(args: Args, backbone: interfaces.VisionBackbone) -> Features:
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    dataset = (
        datasets.load_dataset("imageomics/rare-species", split="train")
        .to_iterable_dataset(num_shards=args.n_workers)
        .map(Preprocess(img_transform))
        .with_format("torch")
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
        shuffle=False,  # We use dataset.shuffle instead
    )

    label_processor = LabelProcessor()

    all_features, all_labels, all_ids = [], [], []

    total = math.ceil(11984 / args.batch_size) if not args.debug else 2
    it = iter(dataloader)
    logger.debug("Need to embed %d batches of %d images.", total, args.batch_size)
    for b in helpers.progress(range(total), every=args.log_every, desc="embed"):
        batch = next(it)

        images = batch["image"].to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features

        all_features.append(features.cpu())

        labels = [label_processor.transform(label) for label in batch["label"]]
        all_labels.extend(labels)

        all_ids.extend(batch["rarespecies_id"])

    all_features = torch.cat(all_features, dim=0).cpu()
    all_ids = np.array(all_ids)
    all_labels = torch.tensor(all_labels)
    logger.info("Got features for %d images.", len(all_ids))

    return Features(all_features, all_labels, all_ids)


@jaxtyped(typechecker=beartype.beartype)
def simpleshot(
    args: Args,
    x_train: Float[Tensor, "n_train dim"],
    y_train: Int[Tensor, " n_train"],
    x_test: Float[Tensor, "n_test dim"],
    y_test: Int[Tensor, " n_test"],
) -> Float[Tensor, " n_test"]:
    """
    Applies simpleshot to the video clips. We assign each clip the majority label. Return the list of scores for x_test.
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
    centroids = torch.from_numpy(clf.centroids_).to(args.device)
    x_test = x_test.to(args.device)
    y_test = y_test.to(args.device)

    scores = []
    for start, stop in batched_idx(len(x_test), args.batch_size):
        x_batch = x_test[start:stop]
        y_batch = y_test[start:stop]
        distances = torch.linalg.vector_norm(x_batch[:, None] - centroids, axis=2)
        preds = torch.argmin(distances, dim=1)

        scores.append((preds == y_batch).type(torch.float32))

    return torch.cat(scores, axis=0)


@jaxtyped(typechecker=beartype.beartype)
def l2_normalize(
    features: Float[Tensor, "n_examples dim"],
) -> Float[Tensor, "n_examples dim"]:
    norms = torch.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / norms


@beartype.beartype
def batched_idx(
    total_size: int, batch_size: int
) -> collections.abc.Iterator[tuple[int, int]]:
    for start in range(0, total_size, batch_size):
        stop = min(start + batch_size, total_size)
        yield start, stop


@jaxtyped(typechecker=beartype.beartype)
def make_split(
    labels: Int[Tensor, " n_examples"], *, k: int
) -> tuple[Int[Tensor, " n_train"], Int[Tensor, " n_test"]]:
    classes = np.unique(labels)

    train_indices = np.array([], dtype=int)
    test_indices = np.array([], dtype=int)

    # Iterate through each class to select indices
    for cls in classes:
        # Indices corresponding to the current class
        cls_indices = np.where(labels == cls)[0]
        # Randomly shuffle the indices
        np.random.shuffle(cls_indices)
        # Select the first K indices for the train set
        cls_train_indices = cls_indices[:k]
        # The rest go into the test set
        cls_test_indices = cls_indices[k:]
        # Append the selected indices to the train/test arrays
        train_indices = np.concatenate((train_indices, cls_train_indices))
        test_indices = np.concatenate((test_indices, cls_test_indices))

    # Shuffle the indices to mix classes
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return torch.from_numpy(train_indices), torch.from_numpy(test_indices)
