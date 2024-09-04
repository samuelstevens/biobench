"""
# NeWT: Natural World Tasks

NeWT is a collection of 164 binary classification tasks related to visual understanding of the natural world ([CVPR 2021 paper](https://arxiv.org/abs/2103.16483), [code](https://github.com/visipedia/newt/tree/main)).

We evaluate a vision model by extracting visual features for each image, fitting a linear SVM to the training examples, and evaluating on the test data.
We aggregate scores across all 164 tasks.

If you use this evaluation, be sure to cite the original work:

```
@inproceedings{van2021benchmarking,
  title={Benchmarking Representation Learning for Natural World Image Collections},
  author={Van Horn, Grant and Cole, Elijah and Beery, Sara and Wilber, Kimberly and Belongie, Serge and Mac Aodha, Oisin},
  booktitle={Computer Vision and Pattern Recognition},
  year={2021}
}
```
"""

__all__ = ["Args", "benchmark"]

import collections.abc
import dataclasses
import logging
import os
import typing

import beartype
import numpy as np
import polars as pl
import scipy.stats
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import torch
import tqdm
from jaxtyping import Bool, Float, Int, jaxtyped
from PIL import Image
from torch import Tensor

from biobench import interfaces

logger = logging.getLogger("newt")


@beartype.beartype
@dataclasses.dataclass
class Args:
    seed: int = 42
    """random seed."""

    dataset_dir: str = ""
    """dataset directory; where you downloaded NEWT to."""
    batch_size: int = 256
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    device: typing.Literal["cpu", "cuda"] = "cuda"
    """(computed at runtime) which kind of accelerator to use."""


@beartype.beartype
def benchmark(
    backbone: interfaces.VisionBackbone, args: Args
) -> interfaces.BenchmarkReport:
    """
    Steps:
    1. Get features for all images.
    2. Select subsets of the features for fitting with SVMs.
    3. Evaluate SVMs and report.
    """

    # 2. Get features.
    all_task_features = get_all_task_specific_features(args, backbone)

    # Fit SVMs.
    results = []
    for task in tqdm.tqdm(all_task_features, desc="SVCs"):
        (x_train, y_train), (x_test, y_test) = task.splits

        x_mean = x_train.mean(axis=0, keepdims=True)

        x_train = x_train - x_mean
        x_train = l2_normalize(x_train)

        x_test = x_test - x_mean
        x_test = l2_normalize(x_test)

        svc = init_svc()

        svc.fit(x_train, y_train)

        train_acc = svc.score(x_train, y_train)
        test_acc = svc.score(x_test, y_test)

        results.append(
            {
                "task": task.name,
                "cluster": task.cluster,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )

    df = pl.DataFrame(results)

    for cluster, test_acc in (
        df.group_by("cluster").agg(pl.col("test_acc").mean()).iter_rows()
    ):
        logger.info("%10s %.1f", cluster, test_acc)

    test_acc = df.select("test_acc").mean().item()
    return interfaces.BenchmarkReport("NeWT", test_acc)


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir: str, df, transform):
        self.transform = transform
        self.image_ids = df.get_column("id").to_list()
        self.dir = dir

    def __getitem__(self, i: int) -> Float[Tensor, "3 width height"]:
        image_id = self.image_ids[i]
        image = Image.open(os.path.join(self.dir, f"{image_id}.jpg"))
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.image_ids)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    cluster: str
    features: Float[np.ndarray, "n_examples dim"]
    labels: Int[np.ndarray, " n_examples"]
    is_train: Bool[np.ndarray, " n_examples"]

    def __repr__(self) -> str:
        return f"Task(task={self.name}, cluster={self.cluster}, features={self.features.shape})"

    @property
    def splits(
        self,
    ) -> tuple[
        tuple[Float[np.ndarray, "n_train dim"], Int[np.ndarray, " n_train"]],
        tuple[Float[np.ndarray, "n_test dim"], Int[np.ndarray, " n_test"]],
    ]:
        x_train = self.features[self.is_train]
        y_train = self.labels[self.is_train]
        x_test = self.features[~self.is_train]
        y_test = self.labels[~self.is_train]

        return (x_train, y_train), (x_test, y_test)


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_all_task_specific_features(
    args: Args, backbone: interfaces.VisionBackbone
) -> collections.abc.Iterator[Task]:
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(args.dataset_dir, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(args.dataset_dir, images_dir_name)

    if not os.path.isfile(labels_csv_path):
        msg = f"Path '{labels_csv_path}' doesn't exist. Did you download the Newt dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with 'dataset-dir'; see --help for more."
        raise RuntimeError(msg)

    df = pl.read_csv(labels_csv_path).with_row_index()

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    dataset = Dataset(images_dir_path, df, img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    )

    all_features = []
    for images in tqdm.tqdm(dataloader, desc="img feats."):
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features
            features = torch.nn.functional.normalize(features, dim=-1)
            all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=0).cpu()
    for task in df.get_column("task").unique():
        task_df = df.filter(pl.col("task") == task)

        task_idx = task_df.get_column("index").to_numpy()
        features = all_features[task_idx].numpy()

        labels = task_df.get_column("label")
        is_train = task_df.select(pl.col("split") == "train").get_column("split")

        cluster = task_df.item(row=0, column="task_cluster")
        yield Task(task, cluster, features, labels.to_numpy(), is_train.to_numpy())


@jaxtyped(typechecker=beartype.beartype)
def l2_normalize(
    features: Float[np.ndarray, "n_examples dim"],
) -> Float[np.ndarray, "n_examples dim"]:
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / norms


def init_svc():
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
        n_jobs=-1,
        random_state=42,
    )
