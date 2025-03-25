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
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from .. import config, helpers, registry, reporting

logger = logging.getLogger("newt")


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    """
    The NeWT benchmark.
    First, get features for all images.
    Second, select the subsets of features that correspond to different tasks and train an SVM.
    Third, evaluate the SVM and report results.
    """

    # Fit SVMs.
    all_preds = []
    for task in get_all_tasks(cfg):
        (x_train, y_train), (x_test, y_test) = task.splits

        x_mean = x_train.mean(axis=0, keepdims=True)

        x_train = x_train - x_mean
        x_train = l2_normalize(x_train)

        x_test = x_test - x_mean
        x_test = l2_normalize(x_test)

        svc = init_svc()

        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        preds = [
            reporting.Prediction(
                str(id),
                float(pred == true),
                {"cluster": task.cluster, "task": task.name},
            )
            for id, pred, true in zip(task.example_ids, y_pred, y_test)
        ]

        all_preds.extend(preds)

    return reporting.Report("newt", all_preds, cfg)


@jaxtyped(typechecker=beartype.beartype)
class Sample(typing.TypedDict):
    """A dictionary representing a single image sample with its metadata.

    Attributes:
        img_id: Unique identifier for the image.
        img: The image tensor with shape [3, width, height] (RGB channels first).
        label: Binary class label (0 or 1) for the image.
    """

    img_id: str
    img: Float[Tensor, "3 width height"]
    label: Int[Tensor, ""]


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """A dataset that returns ImageSample dictionaries."""

    def __init__(
        self,
        root: str,
        img_ids: Shaped[np.ndarray, " n"],
        labels: Int[np.ndarray, " n"],
        transform=None,
    ):
        """Initialize the dataset with image paths and labels.

        Args:
            root: Root directory containing the images.
            img_ids: Array of image IDs.
            labels: Array of binary labels corresponding to the images.
            transform: Optional transform to apply to the images.
        """
        self.transform = transform
        self.root = root
        self.img_ids = img_ids
        self.labels = labels

    def __getitem__(self, i: int) -> Sample:
        """Get a sample by its index.

        Args:
            i: Index of the sample to retrieve.

        Returns:
            A dictionary containing the image ID, image tensor, and label.
        """
        img_id = self.img_ids[i]
        img = Image.open(os.path.join(self.root, f"{img_id}.jpg"))
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[i]
        return {"img_id": img_id, "img": img, "label": label}

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The number of samples.
        """
        return len(self.img_ids)


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
def get_all_tasks(cfg: config.Experiment) -> collections.abc.Iterator[Task]:
    """ """
    rng = np.random.default_rng(seed=cfg.seed)

    # Load model
    backbone = registry.load_vision_backbone(cfg.model)
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(cfg.device))

    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(cfg.data.newt, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(cfg.data.newt, images_dir_name)

    if not os.path.isfile(labels_csv_path):
        msg = f"Path '{labels_csv_path}' doesn't exist. Did you download the Newt dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--data'; see --help for more."
        raise RuntimeError(msg)

    # Read the CSV and add row indices
    df = pl.read_csv(labels_csv_path).with_row_index()
    
    # Sample balanced training data for each task
    df = sample(rng, df, cfg.n_train)
    
    # Get all image IDs and labels
    all_data = df.select("id", "label").to_numpy(structured=True)
    all_ids, all_labels = all_data["id"], all_data["label"]
    
    # Create dataset with all samples
    dataset = Dataset(
        images_dir_path,
        all_ids,
        all_labels,
        img_transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )

    all_features, all_ids = [], []

    # Need to select just a subset of rows based on cfg.n_train.

    total = len(dataloader) if not cfg.debug else 2
    it = iter(dataloader)
    for b in helpers.progress(range(total), every=10, desc="embed"):
        ids, images = next(it)
        images = images.to(cfg.device)

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


@jaxtyped(typechecker=beartype.beartype)
def sample(
    rng: np.random.Generator, df: pl.DataFrame, n_train: int
) -> pl.DataFrame:
    """Sample a balanced subset of training data points for each task.

    Args:
        rng: Random number generator.
        df: NeWT dataframe.
        n_train: Number of training samples per task to return.

    Returns:
        A DataFrame with balanced training samples and all test samples.
    """
    if n_train <= 0:
        return df  # Return all data if n_train is not positive

    # Create a new dataframe to store the results
    result_dfs = []
    
    # Keep all test samples
    test_df = df.filter(pl.col("split") != "train")
    result_dfs.append(test_df)
    
    # Process each task separately
    for task in df.get_column("task").unique():
        task_df = df.filter((pl.col("task") == task) & (pl.col("split") == "train"))
        
        # Skip if the task has no training samples
        if task_df.height == 0:
            continue
            
        # Get samples for each class
        class0_df = task_df.filter(pl.col("label") == 0)
        class1_df = task_df.filter(pl.col("label") == 1)
        
        n0 = min(n_train // 2, class0_df.height)
        n1 = min(n_train - n0, class1_df.height)
        
        # If one class has fewer samples than needed, take more from the other class
        if n0 < n_train // 2 and class1_df.height > n1:
            n1 = min(n_train - n0, class1_df.height)
        elif n1 < (n_train - n_train // 2) and class0_df.height > n0:
            n0 = min(n_train - n1, class0_df.height)
        
        # Sample from each class
        if n0 > 0:
            indices0 = rng.choice(class0_df.height, size=n0, replace=False)
            sampled0 = class0_df.slice(indices0)
            result_dfs.append(sampled0)
            
        if n1 > 0:
            indices1 = rng.choice(class1_df.height, size=n1, replace=False)
            sampled1 = class1_df.slice(indices1)
            result_dfs.append(sampled1)
    
    # Combine all dataframes
    return pl.concat(result_dfs)
