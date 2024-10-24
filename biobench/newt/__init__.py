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

import collections.abc
import dataclasses
import itertools
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
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from biobench import helpers, interfaces, registry

logger = logging.getLogger("newt")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    """NeWT task arguments."""

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
    """
    The NeWT benchmark.
    First, get features for all images.
    Second, select the subsets of features that correspond to different tasks and train an SVM.
    Third, evaluate the SVM and report results.
    """
    # 1. Load model
    backbone = registry.load_vision_backbone(*model_args)

    # 2. Get features.
    all_task_features = get_all_task_specific_features(args, backbone)

    # Fit SVMs.
    results = []
    for task in all_task_features:
        (x_train, y_train), (x_test, y_test) = task.splits

        x_mean = x_train.mean(axis=0, keepdims=True)

        x_train = x_train - x_mean
        x_train = l2_normalize(x_train)

        x_test = x_test - x_mean
        x_test = l2_normalize(x_test)

        svc = init_svc()

        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        examples = [
            interfaces.Example(
                str(id),
                float(pred == true),
                {"cluster": task.cluster, "task": task.name},
            )
            for id, pred, true in zip(task.example_ids, y_pred, y_test)
        ]
        test_acc = np.mean(y_pred == y_test)

        results.append({
            "task": task.name,
            "cluster": task.cluster,
            "examples": examples,
            "test_acc": test_acc,
        })

    # Removes 'examples' from each dict in results
    examples = list(
        itertools.chain.from_iterable((result.pop("examples") for result in results))
    )

    return model_args, interfaces.TaskReport("NeWT", examples)


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    A dataset that returns `(example id, image tensor)` tuples.
    """

    def __init__(self, dir: str, df, transform):
        self.transform = transform
        self.image_ids = df.get_column("id").to_list()
        self.dir = dir

    def __getitem__(self, i: int) -> tuple[str, Float[Tensor, "3 width height"]]:
        image_id = self.image_ids[i]
        image = Image.open(os.path.join(self.dir, f"{image_id}.jpg"))
        if self.transform is not None:
            image = self.transform(image)
        return image_id, image

    def __len__(self) -> int:
        return len(self.image_ids)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Task:
    """
    Task is a group of features and labels for an SVM + a train/test split.
    """

    name: str
    cluster: str
    features: Float[np.ndarray, "batch dim"]
    labels: Int[np.ndarray, " batch"]
    is_train: Bool[np.ndarray, " batch"]
    example_ids: Shaped[np.ndarray, " batch"]  # Should be String[...]

    def __repr__(self) -> str:
        return f"Task(task={self.name}, cluster={self.cluster}, features={self.features.shape})"

    @property
    def splits(
        self,
    ) -> tuple[
        tuple[Float[np.ndarray, "n_train dim"], Int[np.ndarray, " n_train"]],
        tuple[Float[np.ndarray, "n_test dim"], Int[np.ndarray, " n_test"]],
    ]:
        """
        The features and labels for train and test splits.

        Returned as `(x_train, y_train), (x_test, y_test)`.
        """
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
    """ """
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(args.datadir, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(args.datadir, images_dir_name)

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
        pin_memory=False,
        persistent_workers=False,
    )

    all_features, all_ids = [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    for b in helpers.progress(range(total), every=args.log_every, desc="embed"):
        ids, images = next(it)
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features
            features = torch.nn.functional.normalize(features, dim=-1)
            all_features.append(features.cpu())

        all_ids.extend(ids)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_ids = np.array(all_ids)

    for task in df.get_column("task").unique():
        task_df = df.filter(pl.col("task") == task)

        task_idx = task_df.get_column("index").to_numpy()
        features = all_features[task_idx].numpy()
        ids = all_ids[task_idx]

        labels = task_df.get_column("label").to_numpy()
        is_train = task_df.select(pl.col("split") == "train").get_column("split")

        cluster = task_df.item(row=0, column="task_cluster")
        yield Task(task, cluster, features, labels, is_train.to_numpy(), ids)


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
