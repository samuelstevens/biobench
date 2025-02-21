"""
This task measures changes in performance with respect to the stage of life of a bird.
Specifically, we measure classification accuracy among 11 species in multiple settings:

1. Training images are adult, evaluation images are adult. This is the baseline.
2. Training images are juvenile, evaluation images are juvenile. Any drop in performance is likely a reflection on pre-training data distribution.
3. Training images are adult, evaluation images are juvenile. This measures whether model representations are robust to changes in stage of life, which is the opposite of what the original NeWT task measures. We report this number as the primary score.

We use the 11 juvenile vs adult tasks from NeWT, so if you use this task, be sure to cite that work (below).
We use a multiclass SVM from scikit learn.

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

import asyncio
import collections.abc
import dataclasses
import difflib
import logging
import os
import random
import re

import beartype
import numpy as np
import polars as pl
import scipy.stats
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import torch
from jaxtyping import Float, Int, Integer, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from .. import helpers, interfaces, mllms, registry

logger = logging.getLogger("ages")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Ages task arguments."""

    data: str = ""
    """dataset directory; where you downloaded this task's data to."""
    batch_size_cv: int = 256
    """batch size for computer vision model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of batches) to log progress."""
    seed: int = 42
    """random seed."""
    max_examples: int = -1
    """Number of maximum training samples. Negative number means use all of them."""
    parallel: int = 5
    """Concurrent requests per second."""

    # Computed at runtime.
    device: str = "cuda"
    """(computed at runtime) which kind of accelerator to use."""
    debug: bool = False
    """(computed at runtime) whether to run in debug mode."""


@beartype.beartype
def benchmark_cvml(
    args: Args, model_args: interfaces.ModelArgsCvml
) -> tuple[interfaces.ModelArgsCvml, interfaces.TaskReport]:
    """
    Run benchmark.

    Args:
        args: configuration for age task.
        model_args: args to load vision backbone.

    Returns:
        A tuple of model_args and the report describing the results.
    """
    # 1. Load model
    backbone = registry.load_vision_backbone(*model_args)

    # 2. Get features.
    tasks = get_all_tasks_cvml(args, backbone)

    # 3. For each task outlined above, evaluate representation quality.
    splits = {}
    for name, train, test in tasks:
        clf = init_clf()

        clf.fit(train.x, train.y)
        y_pred = clf.predict(test.x)
        examples = [
            interfaces.Example(str(id), float(pred == true), {})
            for id, pred, true in zip(test.ids, y_pred, test.y)
        ]
        test_acc = np.mean(y_pred == test.y)
        splits[name] = test_acc.item()

    return model_args, interfaces.TaskReport("Ages", examples, splits=splits)


@beartype.beartype
def benchmark_mllm(
    args: Args, model_args: interfaces.ModelArgsMllm
) -> tuple[interfaces.ModelArgsMllm, interfaces.TaskReport]:
    rng = random.Random(args.seed)

    splits = {}

    if args.max_examples > 0:
        system = "You will be shown several example bird classifications followed by a test image to classify. For each image, respond only with the classification of the current image. Do not reclassify previous images."
    else:
        system = ""

    with asyncio.Runner() as loop:
        for task, dataset in get_all_tasks_mllm(args):
            limiter = mllms.RateLimiter(args.parallel)
            semaphore = asyncio.Semaphore(args.parallel)

            # We load all the training samples into memory right away because they will be re-used over and over again.
            # Test samples are loaded one by one on demand.
            i_train = rng.choices(task.train, k=args.max_examples)
            train_examples = [
                dataset[i].to_example(rng)
                for i in helpers.progress(i_train, desc="load train samples")
            ]

            @beartype.beartype
            async def run_one(i: int) -> interfaces.Prediction:
                async with semaphore:
                    # This is slow. If I could make this async/await, it would be faster.
                    test_example = dataset[i]
                    # Try to fit them into a prompt.
                    n_examples = 0
                    fewshot_examples = []
                    while mllms.fits(
                        model_args,
                        fewshot_examples,
                        test_example.image_b64,
                        test_example.make_user(rng),
                    ) and (args.max_examples < 0 or n_examples < args.max_examples):
                        # Add another example.
                        n_examples += 1
                        fewshot_examples = train_examples[:n_examples]

                    # Only shuffle once.
                    rng.shuffle(fewshot_examples)

                    await limiter.acquire()
                    assistant = await mllms.send(
                        model_args,
                        fewshot_examples,
                        test_example.image_b64,
                        test_example.make_user(rng),
                        system=system,
                    )
                    pred = test_example.parse_assistant(assistant)
                    return interfaces.Prediction(
                        test_example.image_id,
                        float(pred == test_example.classname),
                        info={
                            "task": task.name,
                            "gold": test_example.classname,
                            "pred": pred,
                            "assistant": assistant,
                        },
                    )

            @beartype.beartype
            async def run_all() -> list[interfaces.Prediction]:
                test_i = task.test
                if args.debug:
                    test_i = test_i[:10]
                jobs = [asyncio.create_task(run_one(i.item())) for i in test_i]

                preds = []
                for job in helpers.progress(jobs):
                    pred: interfaces.Prediction = await job
                    preds.append(pred)
                return preds

            preds = loop.run(run_all())
            test_acc = np.mean([pred.score for pred in preds]).item()
            splits[task.name] = test_acc

    return model_args, interfaces.TaskReport(
        "Ages", args.max_examples, preds, splits=splits
    )


#########
# CV/ML #
#########


@jaxtyped(typechecker=beartype.beartype)
class DatasetCvml(torch.utils.data.Dataset):
    """
    A dataset that returns `(example id, image tensor, integer label)` tuples.
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
    """Inputs and outputs for a given task."""

    x: Float[np.ndarray, " n dim"]
    """Input features; from a `biobench.interfaces.VisionBackbone`."""
    y: Int[np.ndarray, " n"]
    """Class label."""
    ids: Shaped[np.ndarray, " n"]
    """Array of ids; could be strings, could be ints, etc."""


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_all_tasks_cvml(
    args: Args, backbone: interfaces.VisionBackbone
) -> collections.abc.Iterator[tuple[str, Features, Features]]:
    """
    Gets train and test features for all the different tasks being evaluated.

    Args:
        args: configuration for the ages task.
        backbone: the particular vision backbone being evaluated.

    Returns:
        An iterator of (taskname, train features, test features) tuples, one for each task (described in this module's docstring).
    """
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(args.data, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(args.data, images_dir_name)

    if not os.path.isfile(labels_csv_path):
        msg = f"Path '{labels_csv_path}' doesn't exist. Did you download the Newt dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--ages-args.data'; see --help for more."
        raise RuntimeError(msg)

    df = pl.read_csv(labels_csv_path).with_row_index()
    # Only get tasks about age.
    df = df.filter(pl.col("task").str.contains("ml_age"))
    # Add integer label for species (0-indexed).
    df = df.with_columns(species_label=pl.col("task").rank("dense") - 1)

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    dataset = DatasetCvml(images_dir_path, df, img_transform)
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


def init_clf():
    """
    Create a new, randomly initialized SVM with a random hyperparameter search over kernel, C and gamma. It uses only 16 jobs in parallel to prevent overloading the CPUs on a shared machine.
    """
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


########
# MLLM #
########

RAW_TO_CLASSNAME = {
    "ml_age_coopers_hawk": "Cooper's hawk",
    "ml_age_black_bellied_plover": "black-bellied plover",
    "ml_age_semipalmated_plover": "semipalmated plover",
    "ml_age_whimbrel": "whimbrel",
    "ml_age_rough_legged_hawk": "rough-legged hawk",
    "ml_age_swainsons_hawk": "Swainson's hawk",
    "ml_age_bald_eagle": "bald eagle",
    "ml_age_sanderling": "sanderling",
    "ml_age_dunlin": "dunlin",
    "ml_age_western_sandpiper": "western sandpiper",
    "ml_age_least_sandpiper": "least sandpiper",
    "ml_age_sharp_shinned_hawk": "sharp-shinned hawk",
}

CLASSNAMES = list(RAW_TO_CLASSNAME.values())


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SampleMllm:
    image_id: str
    image_b64: str
    classname: str

    def make_user(self, rng: random.Random) -> str:
        classnames = rng.sample(CLASSNAMES, k=len(CLASSNAMES))
        return f"What is this a picture of: {', '.join(classnames[:-1])} or {classnames[-1]}? Respond with your answer in bold."

    @property
    def assistant(self) -> str:
        return f"**{self.classname}**"

    def parse_assistant(self, assistant: str) -> str:
        pattern = re.compile(r"\*\*(.*)\*\*")
        match = pattern.match(assistant)
        if match:
            # Return the closest classname in bold.
            pred = difflib.get_close_matches(match.group(1), CLASSNAMES, cutoff=0.0)[0]
        else:
            # Get the closest classname.
            pred = difflib.get_close_matches(assistant, CLASSNAMES, cutoff=0.0)[0]

        return pred

    def to_example(self, rng: random.Random) -> mllms.Example:
        return mllms.Example(
            image_b64=self.image_b64,
            user=self.make_user(rng),
            assistant=self.assistant,
        )


@jaxtyped(typechecker=beartype.beartype)
class DatasetMllm(torch.utils.data.Dataset):
    """
    A dataset that returns SampleMllms.
    """

    def __init__(self, root: str, df):
        self.root = root

        self.image_ids = df.get_column("id").to_list()
        self.text_labels = df.get_column("task").to_list()

    def __getitem__(self, i: int) -> SampleMllm:
        image_id = self.image_ids[i]
        classname = RAW_TO_CLASSNAME[self.text_labels[i]]

        image_b64 = helpers.load_image_b64(os.path.join(self.root, f"{image_id}.jpg"))

        return SampleMllm(image_id, image_b64, classname)

    def __len__(self) -> int:
        return len(self.image_ids)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class TaskMllm:
    """
    Task is a group of indices for a MLLM with a train/test split.
    """

    name: str
    train: Integer[np.ndarray, " n_train"]
    test: Integer[np.ndarray, " n_test"]

    def __repr__(self) -> str:
        return f"Task(name={self.name}, n_train={len(self.train)}, n_test={len(self.test)})"


@beartype.beartype
def get_all_tasks_mllm(
    args: Args,
) -> collections.abc.Iterator[tuple[TaskMllm, DatasetMllm]]:
    """
    Gets train and test features for all the different tasks being evaluated.

    Args:
        args: configuration for the ages task.
        backbone: the particular vision backbone being evaluated.

    Returns:
        An iterator of (taskname, train features, test features) tuples, one for each task (described in this module's docstring).
    """
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(args.data, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(args.data, images_dir_name)

    if not os.path.isfile(labels_csv_path):
        msg = f"Path '{labels_csv_path}' doesn't exist. Did you download the Newt dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--ages-args.data'; see --help for more."
        raise RuntimeError(msg)

    df = pl.read_csv(labels_csv_path).with_row_index()
    # Only get tasks about age.
    df = df.filter(pl.col("task").str.contains("ml_age"))
    # Add integer label for species (0-indexed).

    dataset = DatasetMllm(images_dir_path, df)

    tasks = (("adult", "adult"), ("not_adult", "not_adult"), ("adult", "not_adult"))
    for train, test in tasks:
        train_i = (
            df.select((pl.col("split") == "train") & (pl.col("text_label") == train))
            .to_numpy()
            .squeeze()
            .nonzero()[0]
        )
        test_i = (
            df.select((pl.col("split") == "test") & (pl.col("text_label") == test))
            .to_numpy()
            .squeeze()
            .nonzero()[0]
        )

        yield TaskMllm(f"{train}/{test}", train_i, test_i), dataset
