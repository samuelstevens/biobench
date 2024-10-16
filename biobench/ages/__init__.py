"""
This task measures changes in performance with respect to the stage of life of a bird.
Specifically, we measure classification accuracy among 11 species in multiple settings:

1. Training images are adult, evaluation images are adult. This is the baseline.
2. Training images are juvenile, evaluation images are juvenile. Any drop in performance is likely a reflection on pre-training data distribution.
3. Training images are adult, evaluation images are juvenile. This measures whether model representations are robust to changes in stage of life, which is the opposite of what the original NeWT task measures. We report this number as the primary score.

We use the 11 juvenile vs adult tasks from NeWT, so if you use this task, be sure to cite that work (below).

To download the original data, follow the instructions in `biobench.newt.download`.

```
@inproceedings{van2021benchmarking,
  title={Benchmarking Representation Learning for Natural World Image Collections},
  author={Van Horn, Grant and Cole, Elijah and Beery, Sara and Wilber, Kimberly and Belongie, Serge and Mac Aodha, Oisin},
  booktitle={Computer Vision and Pattern Recognition},
  year={2021}
}
```
"""

import collections.abc
import dataclasses
import logging
import os

import beartype
import numpy as np
import polars as pl
import scipy.stats
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from biobench import helpers, interfaces, registry

logger = logging.getLogger("ages")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    """Ages task arguments."""

    batch_size: int = 256
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of batches) to log progress."""


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgs
) -> tuple[interfaces.ModelArgs, interfaces.TaskReport]:
    # 1. Load model
    backbone = registry.load_vision_backbone(*model_args)

    # 2. Get features.
    tasks = get_all_tasks(args, backbone)

    # 3. For each task outlined above, evaluate representation quality.
    splits = {}
    for name, train, test in tasks:
        x_mean = train.x.mean(axis=0, keepdims=True)

        x_train = train.x - x_mean
        x_train = l2_normalize(x_train)

        x_test = test.x - x_mean
        x_test = l2_normalize(x_test)

        svc = init_svc()

        svc.fit(x_train, train.y)
        y_pred = svc.predict(x_test)
        examples = [
            interfaces.Example(str(id), float(pred == true), {})
            for id, pred, true in zip(test.ids, y_pred, test.y)
        ]
        test_acc = np.mean(y_pred == test.y)
        splits[name] = test_acc.item()

    return model_args, interfaces.TaskReport("Ages", examples, splits=splits)


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    A dataset that returns `(example id, image tensor)` tuples.
    """

    def __init__(self, dir: str, df, transform):
        self.transform = transform
        self.image_ids = df.get_column("id").to_list()
        self.labels = df.get_column("species_label").to_list()
        self.dir = dir

    def __getitem__(self, i: int) -> tuple[str, Float[Tensor, "3 width height"], int]:
        image_id = self.image_ids[i]
        image = Image.open(os.path.join(self.dir, f"{image_id}.jpg"))
        if self.transform is not None:
            image = self.transform(image)
        return image_id, image, self.labels[i]

    def __len__(self) -> int:
        return len(self.image_ids)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, " n dim"]
    """Input features; from a `biobench.interfaces.VisionBackbone`."""
    y: Int[np.ndarray, " n"]
    """Class label."""
    ids: Shaped[np.ndarray, " n"]
    """Array of ids; could be strings, could be ints, etc."""


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_all_tasks(
    args: Args, backbone: interfaces.VisionBackbone
) -> collections.abc.Iterator[tuple[str, Features, Features]]:
    """ """
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(args.datadir, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(args.datadir, images_dir_name)

    if not os.path.isfile(labels_csv_path):
        msg = f"Path '{labels_csv_path}' doesn't exist. Did you download the Newt dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--ages-args.datadir'; see --help for more."
        raise RuntimeError(msg)

    df = pl.read_csv(labels_csv_path).with_row_index()
    # Only get tasks about age.
    df = df.filter(pl.col("task").str.contains("ml_age"))
    # Add integer label for species (0-indexed).
    df = df.with_columns(species_label=pl.col("task").rank("dense") - 1)

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    dataset = Dataset(images_dir_path, df, img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )

    all_features, all_labels, all_ids = [], [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    logger.debug("Need to embed %d batches of %d images.", total, args.batch_size)
    for b in helpers.progress(range(total), every=args.log_every, desc="Embedding"):
        ids, images, labels = next(it)
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features
            features = torch.nn.functional.normalize(features, dim=-1)
            all_features.append(features.cpu())

        all_ids.extend(ids)
        all_labels.extend(labels)

    all_features = torch.cat(all_features, dim=0).cpu().numpy()
    all_labels = torch.tensor(all_labels).numpy()
    all_ids = np.array(all_ids)
    logger.info("Got features for %d images.", len(all_ids))

    tasks = (("adult", "adult"), ("not_adult", "not_adult"), ("adult", "not_adult"))
    for train, test in tasks:
        train_i = (
            df.select((pl.col("split") == "train") & (pl.col("text_label") == train))
            .to_numpy()
            .squeeze()
        )
        test_i = (
            df.select((pl.col("split") == "test") & (pl.col("text_label") == test))
            .to_numpy()
            .squeeze()
        )

        yield (
            f"{train}/{test}",
            Features(all_features[train_i], all_labels[train_i], all_ids[train_i]),
            Features(all_features[test_i], all_labels[test_i], all_ids[test_i]),
        )


@jaxtyped(typechecker=beartype.beartype)
def l2_normalize(
    features: Float[np.ndarray, "batch dim"],
) -> Float[np.ndarray, "batch dim"]:
    """Normalizes a batch of vectors to have L2 unit norm."""
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / norms


def init_svc():
    """Create a new, randomly initialized SVM with a random hyperparameter search over kernel, C and gamma. It uses only 16 jobs in parallel to prevent overloading the CPUs on a shared machine."""
    return sklearn.model_selection.RandomizedSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.svm.SVC(C=1.0, kernel="rbf"),
        ),
        {
            "svc__C": scipy.stats.loguniform(a=1e-3, b=1e1),
            "svc__kernel": ["rbf", "linear", "sigmoid", "poly"],
            "svc__gamma": scipy.stats.loguniform(a=1e-4, b=1e-3),
        },
        n_iter=100,
        n_jobs=16,
        random_state=42,
    )
