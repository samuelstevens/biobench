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

import asyncio
import collections.abc
import dataclasses
import difflib
import itertools
import logging
import os
import random
import re
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
from jaxtyping import Bool, Float, Int, Integer, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from .. import config, cvml, helpers, interfaces, mllms

logger = logging.getLogger("newt")


########
# CVML #
########


@beartype.beartype
def benchmark_cvml(cfg: config.Experiment) -> interfaces.Report:
    # 1. Load model
    backbone = cvml.load_vision_backbone(cfg.model)

    # 2. Get features.
    all_task_features = get_all_tasks_cvml(cfg, backbone)

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
            interfaces.Prediction(
                str(id),
                float(pred == true),
                {"cluster": task.cluster, "task": task.task},
            )
            for id, pred, true in zip(task.example_ids, y_pred, y_test)
        ]
        test_acc = np.mean(y_pred == y_test)

        results.append({
            "task": task.task,
            "cluster": task.cluster,
            "examples": examples,
            "test_acc": test_acc,
        })

    # Removes 'examples' from each dict in results
    examples = list(
        itertools.chain.from_iterable((result.pop("examples") for result in results))
    )

    # TODO: this is not right
    return interfaces.Report("NeWT", examples)


@jaxtyped(typechecker=beartype.beartype)
class ImageSample(typing.TypedDict):
    img_id: str
    img: Float[Tensor, "3 width height"]


@jaxtyped(typechecker=beartype.beartype)
class ImageDataset(torch.utils.data.Dataset):
    """
    A dataset that returns ImageSample dictionaries.
    """

    def __init__(self, root: str, img_ids: list[str], transform=None):
        self.transform = transform
        self.root = root
        self.img_ids = img_ids

    def __getitem__(self, i: int) -> ImageSample:
        img_id = self.img_ids[i]
        img = Image.open(os.path.join(self.root, f"{img_id}.jpg"))
        if self.transform is not None:
            img = self.transform(img)
        return {"img_id": img_id, "img": img}

    def __len__(self) -> int:
        return len(self.img_ids)


@beartype.beartype
class FeatureSample(typing.TypedDict):
    img_id: str
    x: Float[Tensor, " d"]
    y: Int[Tensor, ""]


@jaxtyped(typechecker=beartype.beartype)
class FeatureDataset(torch.utils.data.Dataset):
    """
    A dataset that returs Feature
    """

    def __init__(self):
        pass

    def __getitem__(self, i: int) -> FeatureSample:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_all_tasks_cvml(
    cfg: config.Experiment, backbone: cvml.VisionBackbone
) -> collections.abc.Iterator[tuple[FeatureDataset, FeatureDataset]]:
    """ """
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(cfg.newt_data, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(cfg.newt_data, images_dir_name)

    if not os.path.isfile(labels_csv_path):
        msg = f"Path '{labels_csv_path}' doesn't exist. Did you download the Newt dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--data'; see --help for more."
        raise RuntimeError(msg)

    df = pl.read_csv(labels_csv_path).with_row_index()

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(cfg.device))

    # Determine indices for training and testing based on configuration
    train_indices = []
    test_indices = []

    # Split indices based on train/test split in the dataset using native Polars methods
    train_indices = df.filter(pl.col("split") == "train").get_column("index").to_list()
    test_indices = df.filter(pl.col("split") != "train").get_column("index").to_list()

    # Apply n_train and n_test limits if specified
    if cfg.n_train > 0 and len(train_indices) > cfg.n_train:
        train_indices = random.sample(train_indices, cfg.n_train)

    if cfg.n_test > 0 and len(test_indices) > cfg.n_test:
        test_indices = random.sample(test_indices, cfg.n_test)

    if cfg.debug:
        n = cfg.batch_size * 2
        total = 2  # 2 batches

    # Use the appropriate indices based on whether we're processing training or testing data
    selected_indices = train_indices if is_train else test_indices

    if len(selected_indices) == 0:
        # No data for this split
        return FeaturesCvml(
            torch.zeros((0, 768), dtype=float), torch.zeros((0, 9), dtype=float), []
        )

    # Calculate total batches needed
    n = len(selected_indices)
    total = (n + cfg.batch_size - 1) // cfg.batch_size  # Ceiling division

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
        yield TaskCvml(task, cluster, features, labels, is_train.to_numpy(), ids)


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


########
# MLLM #
########


@beartype.beartype
def benchmark_mllm(cfg: config.Experiment) -> list[interfaces.Report]:
    rng = random.Random(cfg.seed)

    reports = []
    with asyncio.Runner() as loop:
        limiter = mllms.RateLimiter(cfg.parallel)
        semaphore = asyncio.Semaphore(cfg.parallel)

        for train_dataset, test_dataset in get_all_tasks_mllm(cfg):
            # We load all the training samples into memory right away because they will be re-used over and over again.
            # Test samples are loaded one by one on demand.
            train_indices = list(range(len(train_dataset)))
            i_train = rng.sample(train_indices, k=min(cfg.n_train, len(train_indices)))

            train_examples = [train_dataset[i].to_example(rng) for i in i_train]
            if train_examples:
                logger.info("Loaded %d training examples.", len(train_examples))

            @beartype.beartype
            async def run_one(i: int) -> interfaces.Prediction:
                async with semaphore:
                    example = test_dataset[i]

                    # Set up prompt.
                    n = 0
                    fewshot = []
                    while (
                        mllms.fits(
                            cfg, fewshot, example.img_b64, example.make_user(rng)
                        )
                    ) and (cfg.n_train < 0 or n < cfg.n_train):
                        # Add another example.
                        n += 1
                        fewshot = train_examples[:n]

                    # Only shuffle once.
                    rng.shuffle(fewshot)

                    await limiter.acquire()
                    assistant = await mllms.send(
                        cfg, fewshot, example.img_b64, example.make_user(rng)
                    )
                    pred_y, parsed = example.parse_assistant(assistant)

                    return interfaces.Prediction(
                        example.img_id,
                        float(pred_y == example.label),
                        len(fewshot),
                        info={
                            "task": test_dataset.task,
                            "cluster": test_dataset.cluster,
                            "subcluster": test_dataset.subcluster,
                            "parsed": parsed,
                        },
                    )

            @beartype.beartype
            async def run_all() -> list[interfaces.Prediction]:
                if cfg.debug:
                    logger.info(
                        "Using the first 10 examples out of %d.", len(test_dataset)
                    )
                    test_i = list(range(10))
                elif cfg.n_test >= 0 and cfg.n_test < len(test_dataset):
                    logger.info(
                        "Using %d random examples out of %d.",
                        cfg.n_test,
                        len(test_dataset),
                    )
                    test_i = rng.sample(range(len(test_dataset)), k=cfg.n_test)
                else:
                    logger.info(
                        "Using entire test set (%d examples).", len(test_dataset)
                    )
                    test_i = list(range(len(test_dataset)))

                jobs = [asyncio.create_task(run_one(i)) for i in test_i]
                preds = []
                for job in helpers.progress(jobs, desc=train_dataset.task):
                    pred: interfaces.Prediction = await job
                    preds.append(pred)
                return preds

            preds = loop.run(run_all())
            report = interfaces.Report(
                f"NeWT::{train_dataset.task}", preds, preds[0].n_train, cfg
            )
            reports.append(report)

    return reports


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SampleMllm:
    img_id: str
    img_b64: str
    label: int

    # TODO: these classnames are not being translated correctly.
    classnames: tuple[str, str]

    @property
    def classname(self) -> str:
        return self.classnames[self.label]

    def make_user(self, rng: random.Random) -> str:
        a, b = self.classnames
        if rng.random() > 0.5:
            a, b = b, a
        return f"What is this a picture of, '{a}' or '{b}'? Respond with your answer in bold."

    @property
    def assistant(self) -> str:
        return f"**{self.classname}**"

    def parse_assistant(self, assistant: str) -> tuple[int, bool]:
        """
        Returns a label, parsed to an integer. Also returns a boolean indicating whether parsing was successful.
        """
        pattern = re.compile(r"\*\*(.*)\*\*")
        match = pattern.match(assistant)
        if match:
            # Return the closest classname in bold.
            pred = difflib.get_close_matches(
                match.group(1), self.classnames, cutoff=0.0
            )[0]
            successful = True
        else:
            # Get the closest classname.
            pred = difflib.get_close_matches(assistant, self.classnames, cutoff=0.0)[0]
            successful = False

        for i, classname in enumerate(self.classnames):
            if classname == pred:
                return (i, successful)

        logger.warning(
            "Something is wrong in %s.parse_assistant.", self.__class__.__name__
        )
        return (0, False)

    def to_example(self, rng: random.Random) -> mllms.Example:
        return mllms.Example(
            img_b64=self.img_b64,
            user=self.make_user(rng),
            assistant=self.assistant,
        )


@jaxtyped(typechecker=beartype.beartype)
class DatasetMllm(torch.utils.data.Dataset):
    """
    A dataset that returns SampleMllms for a specific task.
    This dataset can be instantiated as either a training or test dataset.

    This dataset presents a clean 0-n indexing interface while internally
    mapping to the appropriate indices in the full dataset.
    """

    def __init__(
        self,
        task: str,
        cluster: str,
        subcluster: str | None,
        indices: Integer[np.ndarray, " n_samples"],
        root: str,
        df: pl.DataFrame,
        is_train: bool,
    ):
        self.task = task
        self.cluster = cluster
        self.subcluster = subcluster
        self.indices = indices
        self.root = root
        self.df = df
        self.is_train = is_train

        # Store the full dataset information
        self.img_ids = self.df.get_column("id").to_list()
        self.labels = self.df.get_column("label").to_list()
        self.tasks = self.df.get_column("task").to_list()

        # Create task-specific dataframes for efficient access
        self.task_df = self.df.filter(pl.col("task") == self.task)

    def __repr__(self) -> str:
        split = "train" if self.is_train else "test"
        return f"DatasetMllm(task={self.task}, cluster={self.cluster}, split={split}, n_samples={len(self.indices)})"

    def __getitem__(self, i: int) -> SampleMllm:
        """
        Get a sample by its index in this dataset (0 to len-1).
        Internally maps to the correct index in the full dataset.
        """
        if i >= len(self.indices):
            raise IndexError(
                f"Index {i} out of bounds for dataset with {len(self.indices)} samples"
            )

        # Map the local index to the global index
        global_idx = self.indices[i]

        img_id = self.img_ids[global_idx]
        label = self.labels[global_idx]
        task = self.tasks[global_idx]

        classnames = tuple(text_label_to_classname[task].keys())
        img_b64 = helpers.load_img_b64(os.path.join(self.root, f"{img_id}.jpg"))

        return SampleMllm(
            img_id,
            img_b64,
            label,
            classnames,
        )

    def __len__(self) -> int:
        """Return the number of samples in this dataset"""
        return len(self.indices)


@jaxtyped(typechecker=beartype.beartype)
def get_all_tasks_mllm(
    cfg: config.Experiment,
) -> collections.abc.Iterator[tuple[DatasetMllm, DatasetMllm]]:
    """
    Get all tasks as pairs of (train_dataset, test_dataset).
    Each dataset is a DatasetMllm instance.
    """
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(cfg.newt_data, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(cfg.newt_data, images_dir_name)

    if not os.path.isfile(labels_csv_path):
        msg = f"Path '{labels_csv_path}' doesn't exist. Did you download the Newt dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--data'; see --help for more."
        raise RuntimeError(msg)

    df = pl.read_csv(labels_csv_path).with_row_index()

    for task_name in df.get_column("task").unique():
        task_df = df.filter(pl.col("task") == task_name)

        # Get the cluster and sub-cluster for this task
        cluster = task_df.item(row=0, column="task_cluster")
        subcluster = task_df.item(row=0, column="task_subcluster")

        if not include_task(cfg.newt, task_name, cluster, subcluster):
            continue

        task_idx = task_df.get_column("index").to_numpy()
        is_train = (
            task_df.select(pl.col("split") == "train").get_column("split").to_numpy()
        )

        train_dataset = DatasetMllm(
            task=task_name,
            cluster=cluster,
            subcluster=subcluster,
            indices=task_idx[is_train],
            root=images_dir_path,
            df=df,
            is_train=True,
        )

        test_dataset = DatasetMllm(
            task=task_name,
            cluster=cluster,
            subcluster=subcluster,
            indices=task_idx[~is_train],
            root=images_dir_path,
            df=df,
            is_train=False,
        )

        yield (train_dataset, test_dataset)


text_label_to_classname = {
    # FGVCX
    "fgvcx_plant_pathology_healthy_vs_sick": {
        "healthy": "healthy plant",
        "sick": "sick plant",
    },
    "fgvcx_icassava_healthy_vs_sick": {
        "healthy": "healthy cassava leaf",
        "sick": "sick cassave leaf",
    },
    # ML Photo
    # (sam) we should change the openai templates for this task
    "ml_photo_rating_12_vs_45_v1": {
        "rating_4_or_5": "obvious bird",
        "rating_1_or_2": "obscured bird",
    },
    "ml_photo_rating_12_vs_45_v2": {
        "rating_4_or_5": "obvious bird",
        "rating_1_or_2": "obscured bird",
    },
    "ml_photo_rating_12_vs_45_v3": {
        "rating_4_or_5": "obvious bird",
        "rating_1_or_2": "obscured bird",
    },
    "ml_photo_rating_12_vs_45_v4": {
        "rating_4_or_5": "obvious bird",
        "rating_1_or_2": "obscured bird",
    },
    "ml_photo_rating_12_vs_45_v5": {
        "rating_4_or_5": "obvious bird",
        "rating_1_or_2": "obscured bird",
    },
    # ML Bio
    "ml_bio_has_red_eyes": {
        "has_red_eyes": "bird with red eyes",
        "not_red_eyes": "bird",
    },
    "ml_bio_high_contrast": {
        "high_contrast": "bird with high contrast in its colors",
        "not_high_contrast": "bird with low contrast in its colors",
    },
    "ml_bio_raptor_utility_pole": {
        "neg": "raptor in the wild",
        "raptor_on_pole": "raptor on a utility pole",
    },
    "ml_bio_is_at_flower": {
        "is_at_flower": "bird near a flower",
        "not_at_flower": "bird in the wild",
    },
    # ML Tag
    "ml_tag_back_of_camera": {
        "back_of_camera": "photo of a bird",
        "not_back_of_camera": "bird in the wild",
    },
    "ml_tag_copulation": {
        "not_copulation": "bird(s) in the wild",
        "copulation": "birds mating",
    },
    "ml_tag_feeding_young": {
        "not_feeding_young": "bird in the wild",
        "feeding_young": "bird feeding its young",
    },
    "ml_tag_egg": {
        "egg": "egg",
        "no_egg": "bird",
    },
    "ml_tag_watermark": {
        "no_watermark": "bird",
        "watermark": "bird with a watermark",
    },
    "ml_tag_field_notes_sketch": {
        "field_notes_sketch": "field drawing of a bird",
        "not_field_notes_sketch": "bird",
    },
    "ml_tag_nest": {
        "no_nest": "bird in the wild",
        "nest": "bird in its nest",
    },
    "ml_tag_molting_waterfowl": {
        "has_red_eyes": "molting waterfowl",
        "not_red_eyes": "regular waterfowl",
    },
    "ml_tag_molting_raptors": {
        "molting": "molting raptor",
        "not_molting": "regular raptor",
    },
    "ml_tag_vocalizing": {
        "not_vocalizing": "bird with its mouth closed",
        "vocalizing": "vocalizing bird",
    },
    "ml_tag_dead": {
        "not_dead": "living bird",
        "dead": "dead bird",
    },
    "ml_tag_in_hand": {
        "in_hand": "bird in a human hand",
        "not_in_hand": "bird in the wild",
    },
    "ml_tag_multiple_species": {
        "single_species": "one single species",
        "multiple_species": "multiple different species",
    },
    "ml_tag_carrying_food": {
        "not_carrying_food": "bird",
        "carrying_food": "bird carrying food",
    },
    # (sam) I used typical here instead of regular
    "ml_tag_foraging_waterfowl": {
        "not_waterfowl_foraging": "typical waterfowl",
        "waterfowl_foraging": "foraging waterfowl",
    },
    # I want the non_bird to not include the word bird in it.
    "ml_tag_non_bird": {
        "non_bird": "some thing",
        "not_non_bird": "bird",
    },
    # ML Age
    "ml_age_coopers_hawk": {
        "adult": "adult Cooper's hawk",
        "not_adult": "juvenile Cooper's hawk",
    },
    "ml_age_black_bellied_plover": {
        "not_adult": "juveile black-bellied plover",
        "adult": "adult black-bellied plover",
    },
    "ml_age_semipalmated_plover": {
        "adult": "adult semipalmated plover",
        "not_adult": "juvenile semipalmated plover",
    },
    "ml_age_whimbrel": {
        "not_adult": "juvenile whimbrel",
        "adult": "adult whimbrel",
    },
    "ml_age_rough_legged_hawk": {
        "adult": "adult rough-legged hawk",
        "not_adult": "juvenile rough-legged hawk",
    },
    "ml_age_swainsons_hawk": {
        "not_adult": "juvenile Swainson's hawk",
        "adult": "adult Swainson's hawk",
    },
    "ml_age_bald_eagle": {
        "not_adult": "juvenile bald eagle",
        "adult": "adult bald eagle",
    },
    "ml_age_sanderling": {
        "adult": "adult sanderling",
        "not_adult": "juvenile sanderling",
    },
    "ml_age_dunlin": {
        "adult": "adult dunlin",
        "not_adult": "juvenile dunlin",
    },
    "ml_age_western_sandpiper": {
        "not_adult": "juvenile western sandpiper",
        "adult": "adult wester sandpiper",
    },
    "ml_age_least_sandpiper": {
        "not_adult": "juvenile least sandpiper",
        "adult": "adult least sandpiper",
    },
    "ml_age_sharp_shinned_hawk": {
        "adult": "adult sharp-shinned hawk",
        "not_adult": "juvenile sharp-shinned hawk",
    },
    # NABirds
    "nabirds_species_classification_amekes_merlin": {
        "Merlin": "merlin",
        "American Kestrel": "American kestrel",
    },
    "nabirds_species_classification_botgra_grtgra": {
        "Boat-tailed Grackle": "boat-tailed grackle",
        "Great-tailed Grackle": "great-tailed grackle",
    },
    "nabirds_species_classification_easmea_wesmea": {
        "Eastern Meadowlark": "eastern meadowlark",
        "Western Meadowlark": "western meadowlark",
    },
    "nabirds_species_classification_orcwar_tenwar": {
        "Tennessee Warbler": "Tennessee warbler",
        "Orange-crowned Warbler": "orange-crowned warbler",
    },
    "nabirds_species_classification_houwre_winwre3": {
        "House Wren": "house wren",
        "Winter Wren": "winter wren",
    },
    "nabirds_species_classification_buhvir_casvir": {
        "Blue-headed Vireo": "blue-headed vireo",
        "Cassin's Vireo": "Cassin's vireo",
    },
    "nabirds_species_classification_cavswa_cliswa": {
        "Cave Swallow": "cave swallow",
        "Cliff Swallow": "cliff swallow",
    },
    "nabirds_species_classification_blkvul_turvul": {
        "Turkey Vulture": "turkey vulture",
        "Black Vulture": "black vulture",
    },
    "nabirds_species_classification_bkchum_rthhum": {
        "Black-chinned Hummingbird": "black-chinned hummingbird",
        "Ruby-throated Hummingbird": "ruby-throated hummingbird",
    },
    "nabirds_species_classification_gloibi_whfibi": {
        "Glossy Ibis": "glossy ibis",
        "White-faced Ibis": "white-faced ibis",
    },
    "nabirds_species_classification_brwhaw_reshaw": {
        "Red-shouldered Hawk": "red-shouldered hawk",
        "Broad-winged Hawk": "broad-winged hawk",
    },
    "nabirds_species_classification_bargol_comgol": {
        "Barrow's Goldeneye": "Barrow's goldeneye",
        "Common Goldeneye": "common goldeneye",
    },
    "nabirds_species_classification_amecro_comrav": {
        "American Crow": "American crow",
        "Common Raven": "common raven",
    },
    "nabirds_species_classification_coohaw_shshaw": {
        "Sharp-shinned Hawk": "sharp-shinned hawk",
        "Cooper's Hawk": "Cooper's hawk",
    },
    "nabirds_species_classification_savspa_sonspa": {
        "Song Sparrow": "song sparrow",
        "Savannah Sparrow": "savannah sparrow",
    },
    "nabirds_species_classification_linspa_sonspa": {
        "Lincoln's Sparrow": "Lincoln's sparrow",
        "Song Sparrow": "song sparrow",
    },
    "nabirds_species_classification_gresca_lessca": {
        "Greater Scaup": "greater scaup",
        "Lesser Scaup": "lesser scaup",
    },
    "nabirds_species_classification_eawpew_wewpew": {
        "Eastern Wood-Pewee": "eastern wood-pewee",
        "Western Wood-Pewee": "western wood-pewee",
    },
    "nabirds_species_classification_herthr_swathr": {
        "Hermit Thrush": "hermit thrush",
        "Swainson's Thrush": "Swainson's thrush",
    },
    "nabirds_species_classification_greyel_lesyel": {
        "Lesser Yellowlegs": "lesser yellowlegs",
        "Greater Yellowlegs": "greater yellowlegs",
    },
    "nabirds_species_classification_linspa_savspa": {
        "Lincoln's Sparrow": "Lincoln's sparrow",
        "Savannah Sparrow": "savannah sparrow",
    },
    "nabirds_species_classification_houfin_purfin": {
        "Purple Finch": "purple finch",
        "House Finch": "house finch",
    },
    "nabirds_species_classification_cacgoo1_cangoo": {
        "Canada Goose": "Canada goose",
        "Cackling Goose": "cackling goose",
    },
    "nabirds_species_classification_semsan_wessan": {
        "Semipalmated Sandpiper": "semipalmated sandpiper",
        "Western Sandpiper": "western sandpiper",
    },
    "nabirds_species_classification_canvas_redhea": {
        "Redhead": "redhead",
        "Canvasback": "canvasback",
    },
    "nabirds_species_classification_hergul_ribgul": {
        "Ring-billed Gull": "ring-billed gull",
        "Herring Gull": "herring gull",
    },
    "nabirds_species_classification_truswa_tunswa": {
        "Tundra Swan": "tundra swan",
        "Trumpeter Swan": "trumpeter swan",
    },
    "nabirds_species_classification_bkcchi_carchi": {
        "Carolina Chickadee": "Carolina chickadee",
        "Black-capped Chickadee": "black-capped chickadee",
    },
    "nabirds_species_classification_solsan_sposan": {
        "Spotted Sandpiper": "spotted sandpiper",
        "Solitary Sandpiper": "solitary sandpiper",
    },
    "nabirds_species_classification_rosgoo_snogoo": {
        "Snow Goose": "snow goose",
        "Ross's Goose": "Ross's goose",
    },
    "nabirds_species_classification_dowwoo_haiwoo": {
        "Hairy Woodpecker": "hairy woodpecker",
        "Downy Woodpecker": "downy woodpecker",
    },
    "nabirds_species_classification_buhvir_plsvir": {
        "Plumbeous Vireo": "plumbeous vireo",
        "Blue-headed Vireo": "blue-headed vireo",
    },
    "nabirds_species_classification_casvir_plsvir": {
        "Plumbeous Vireo": "plumbeous vireo",
        "Cassin's Vireo": "Cassin's vireo",
    },
    "nabirds_species_classification_comrav_fiscro": {
        "Fish Crow": "fish crow",
        "Common Raven": "common raven",
    },
    "nabirds_species_classification_rensap_yebsap": {
        "Yellow-bellied Sapsucker": "yellow-bellied sapsucker",
        "Red-naped Sapsucker": "red-naped sapsucker",
    },
    "nabirds_species_classification_sursco_whwsco2": {
        "Surf Scoter": "surf scoter",
        "White-winged Scoter": "white-winged scoter",
    },
    "nabirds_species_classification_commer_rebmer": {
        "Common Merganser": "common merganser",
        "Red-breasted Merganser": "red-breasted merganser",
    },
    "nabirds_species_classification_barswa_cliswa": {
        "Barn Swallow": "barn swallow",
        "Cliff Swallow": "cliff swallow",
    },
    "nabirds_species_classification_amecro_fiscro": {
        "American Crow": "American crow",
        "Fish Crow": "fish crow",
    },
    "nabirds_species_classification_louwat_norwat": {
        "Northern Waterthrush": "northern waterthrush",
        "Louisiana Waterthrush": "Louisiana waterthrush",
    },
    # iNat non-species
    "inat_non_species_dead_jackal": {
        "dead_coyote": "dead coyote",
        "dead_golden_jackal": "dead golden jackal",
    },
    "inat_non_species_white_american_robin": {
        "regular_robin": "regular robin",
        "white_robin": "white robin",
    },
    "inat_non_species_tagged_swan": {
        "not_tagged_swan": "regular swan",
        "tagged_swan": "tagged swan",
    },
    "inat_non_species_intersex_mallards": {
        "not_intersex": "regular mallard",
        "intersex": "intersex mallard",
    },
    "inat_non_species_birds_near_signs": {
        "bird_not_on_sign": "bird in the wild",
        "bird_on_sign": "bird on a man-made sign",
    },
    "inat_non_species_diseased_zebra_finch": {
        "regular_zebra_finch": "regular zebra finch",
        "diseased_zebra_finch": "diseased zebra finch",
    },
    "inat_non_species_mating_chauliognathus_pensylvanicus": {
        "mating": "mating Chauliognathus pensylvanicus",
        "not_mating": "non-mating Chauliognathus pensylvanicus",
    },
    "inat_non_species_mating_common_green_darner": {
        "mating": "mating common green darner",
        "not_mating": "non-mating common green darner",
    },
    "inat_non_species_black_eastern_gray_squirrel": {
        "black_squirrel": "black squirrel",
        "regular_squirrel": "regular squirrel",
    },
    "inat_non_species_dead_striped_skunk": {
        "dead_striped_skunk": "dead striped skunk",
        "dead_hog_nosed_skunk": "dead hog-nosed skunk",
    },
    "inat_non_species_dead_common_garter_snake": {
        "common_garter_snake": "dead common garter snake",
        "gopher_snake": "dead gopher snake",
    },
    "inat_non_species_diseased_leaves": {
        "mulberry_leaf_leaf": "diseased mulberry leaf",
        "red_dock_leaf": "red dock leaf",
    },
    "inat_non_species_mating_bagrada_hilaris": {
        "not_mating": "non-mating Bagrada hilaris",
        "mating": "mating Bagrada hilaris",
    },
    "inat_non_species_mating_hippodamia_convergens": {
        "not_mating": "non-mating Hippodamia convergens",
        "mating": "mating Hippodamia convergens",
    },
    "inat_non_species_mating_harmonia_axyridis": {
        "not_mating": "non-mating Harmonia axyridis",
        "mating": "mating Harmonia axyridis",
    },
    "inat_non_species_white_white_tailed_deer": {
        "regular_deer": "regular deer",
        "white_deer": "white-tailed deer",
    },
    "inat_non_species_mating_oncopeltus_fasciatus": {
        "mating": "mating Oncopeltus fasciatus",
        "not_mating": "non-mating Oncopeltus fasciatus",
    },
    "inat_non_species_mating_aligator_lizard": {
        "mating": "mating alligator lizard",
        "not_mating": "non-mating alligator lizard",
    },
    "inat_non_species_mating_toxomerus_marginatus": {
        "mating": "mating Toxomerus marginatus",
        "not_mating": "non-mating Toxomerus marginatus",
    },
    "inat_non_species_mating_danaus_plexippus": {
        "mating": "mating Danaus plexippus",
        "not_mating": "non-mating Danaus plexippus",
    },
    "inat_non_species_feather_california_scrub_jay_v_quail": {
        "quail_feather": "quail feather",
        "scrub_jay_feather": "scrub jay feather",
    },
    "inat_non_species_mating_argia_vivida": {
        "mating": "mating Argia vivida",
        "not_mating": "non-mating Argia vivida",
    },
    "inat_non_species_mammal_species": {
        "bobcat_feces": "bobcat feces",
        "black_bear_feces": "black bear feces",
    },
    "inat_non_species_deformed_beak": {
        "deformed_beak": "bird with a deformed beak",
        "regular_beak": "bird with a regular beak",
    },
    "inat_non_species_mating_terrapene_carolina": {
        "mating": "mating Terrapene carolina",
        "not_mating": "non-mating Terrapene carolina",
    },
    # iNat Observed
    "inat_observed_Yellow-backed_Spiny_Lizard_vs_Desert_Spiny_Lizard": {
        "Desert Spiny Lizard": "desert spiny lizard",
        "Yellow-backed Spiny Lizard": "yellow-backed spiny lizard",
    },
    "inat_observed_Orange_Jelly_Spot_vs_witch's_butter": {
        "witch's butter": "witch's butter",
        "Orange Jelly Spot": "orange jelly spot",
    },
    "inat_observed_Eastern_Meadowlark_vs_Western_Meadowlark": {
        "Eastern Meadowlark": "eastern meadowlark",
        "Western Meadowlark": "western meadowlark",
    },
    "inat_observed_Groove-billed_Ani_vs_Smooth-billed_Ani": {
        "Smooth-billed Ani": "smooth-billed ani",
        "Groove-billed Ani": "groove-billed ani",
    },
    "inat_observed_Pacific_Banana_Slug_vs_Button's_Banana_Slug": {
        "Button's Banana Slug": "Button's banana slug",
        "Pacific Banana Slug": "Pacific banana slug",
    },
    "inat_observed_Red_Belted_Conk_vs_Northern_Red_Belt": {
        "Northern Red Belt": "northern red belt",
        "Red Belted Conk": "red belted conk",
    },
    "inat_observed_Brown-lipped_Snail_vs_White-lipped_Snail": {
        "Brown-lipped Snail": "brown-lipped snail",
        "White-lipped Snail": "white-lipped snail",
    },
    "inat_observed_Cross_Orbweaver_vs_Hentz's_Orbweaver": {
        "Hentz's Orbweaver": "Hentz's orbweaver",
        "Cross Orbweaver": "cross orbweaver",
    },
    "inat_observed_Common_Grass_Yellow_vs_Three-spotted_Grass_Yellow": {
        "Three-spotted Grass Yellow": "three-spotted grass yellow",
        "Common Grass Yellow": "common grass yellow",
    },
    "inat_observed_southern_cattail_vs_lesser_reedmace": {
        "lesser reedmace": "lesser reedmace",
        "southern cattail": "southern cattail",
    },
    "inat_observed_Blue_Mussel_vs_California_Mussel": {
        "Blue Mussel": "blue mussel",
        "California Mussel": "California mussel",
    },
    "inat_observed_Northern_Two-lined_Salamander_vs_Southern_Two-lined_Salamander": {
        "Southern Two-lined Salamander": "southern two-lined salamander",
        "Northern Two-lined Salamander": "northern two-lined salamander",
    },
    "inat_observed_Belize_Crocodile_vs_American_Crocodile": {
        "American Crocodile": "American crocodile",
        "Belize Crocodile": "Belize crocodile",
    },
    "inat_observed_Jelly_Ear_vs_Ear_fungus": {
        "Jelly Ear": "jelly ear",
        "Ear fungus": "ear fungus",
    },
    "inat_observed_Desert_Blonde_Tarantula_vs_Desert_Tarantula": {
        "Desert Blonde Tarantula": "desert blonde tarantula",
        "Desert Tarantula": "desert tarantula",
    },
    "inat_observed_Northern_Cinnabar_Polypore_vs_Cinnabar_Bracket": {
        "Cinnabar Bracket": "cinnabar bracket",
        "Northern Cinnabar Polypore": "northern cinnabar polypore",
    },
    "inat_observed_Western_Mosquitofish_vs_Eastern_Mosquitofish": {
        "Western Mosquitofish": "western mosquitofish",
        "Eastern Mosquitofish": "eastern mosquitofish",
    },
    "inat_observed_Western_Grey_Kangaroo_vs_Eastern_Grey_Kangaroo": {
        "Western Grey Kangaroo": "western grey kangaroo",
        "Eastern Grey Kangaroo": "eastern grey kangaroo",
    },
    "inat_observed_Eastern_Cane_Toad_vs_Giant_Marine_Toad": {
        "Giant Marine Toad": "giant marine toad",
        "Eastern Cane Toad": "eastern cane Toad",
    },
    "inat_observed_Eastern_Oyster_vs_Pacific_Oyster": {
        "Pacific Oyster": "Pacific oyster",
        "Eastern Oyster": "Eastern oyster",
    },
    "inat_observed_Snakeskin_Chiton_vs_California_Spiny_Chiton": {
        "California Spiny Chiton": "California spiny chiton",
        "Snakeskin Chiton": "snakeskin chiton",
    },
    "inat_observed_Flea_Jumper_vs_Asiatic_Wall_Jumping_Spider": {
        "Asiatic Wall Jumping Spider": "Asiatic wall jumping spider",
        "Flea Jumper": "flea jumper",
    },
    "inat_observed_California_Sea_Lion_vs_Steller_Sea_Lion": {
        "Steller Sea Lion": "Steller sea lion",
        "California Sea Lion": "California sea lion",
    },
    "inat_observed_Southern_Cinnabar_Polypore_vs_Cinnabar_Bracket": {
        "Southern Cinnabar Polypore": "southern cinnabar polypore",
        "Cinnabar Bracket": "cinnabar bracket",
    },
    "inat_observed_Southern_Black_Widow_vs_Western_Black_Widow": {
        "Southern Black Widow": "Southern black widow",
        "Western Black Widow": "Western black widow",
    },
    "inat_observed_Eastern_Ribbonsnake_vs_Western_Ribbon_Snake": {
        "Eastern Ribbonsnake": "eastern ribbonsnake",
        "Western Ribbon Snake": "western ribbonsnake",
    },
    "inat_observed_Brown_House_Spider_vs_False_Black_Widow": {
        "False Black Widow": "false black widow",
        "Brown House Spider": "brown house spider",
    },
    "inat_observed_Allegheny_Mountain_Dusky_Salamander_vs_Dusky_Salamander": {
        "Dusky Salamander": "dusky salamander",
        "Allegheny Mountain Dusky Salamander": "Allegheny Mountain dusky salamander",
    },
    "inat_observed_Rough_Green_Snake_vs_Smooth_Greensnake": {
        "Rough Green Snake": "rough green snake",
        "Smooth Greensnake": "smooth greensnake",
    },
    "inat_observed_Common_Shiny_Woodlouse_vs_Rathke’s_Woodlouse": {
        "Rathke’s Woodlouse": "Rathke’s woodlouse",
        "Common Shiny Woodlouse": "common shiny woodlouse",
    },
    # iNat Unobserved
    "inat_unobserved_armillaria_luteobubalina_v_armillaria_novae-zelandiae": {
        "Armillaria novae-zelandiae": "Armillaria novae-zelandiae",
        "Armillaria luteobubalina": "Armillaria luteobubalina",
    },
    "inat_unobserved_phaeophyscia_orbicularis_v_phaeophyscia_rubropulchra": {
        "Phaeophyscia orbicularis": "Phaeophyscia orbicularis",
        "Phaeophyscia rubropulchra": "Phaeophyscia rubropulchra",
    },
    "inat_unobserved_corvus_orru_v_corvus_sinaloae": {
        "Corvus sinaloae": "Corvus sinaloae",
        "Corvus orru": "Corvus orru",
    },
    "inat_unobserved_lampsilis_cardium_v_lampsilis_siliquoidea": {
        "Lampsilis siliquoidea": "Lampsilis siliquoidea",
        "Lampsilis cardium": "Lampsilis cardium",
    },
    "inat_unobserved_diaea_dorsata_v_diaea_ambara": {
        "Diaea ambara": "Diaea ambara",
        "Diaea dorsata": "Diaea dorsata",
    },
    "inat_unobserved_polystichum_aculeatum_v_polystichum_setiferum": {
        "Polystichum setiferum": "Polystichum setiferum",
        "Polystichum aculeatum": "Polystichum aculeatum",
    },
    "inat_unobserved_pinus_clausa_v_pinus_mugo": {
        "Pinus mugo": "Pinus mugo",
        "Pinus clausa": "Pinus clausa",
    },
    "inat_unobserved_judolia_cordifera_v_judolia_cerambyciformis": {
        "Judolia cerambyciformis": "Judolia cerambyciformis",
        "Judolia cordifera": "Judolia cordifera",
    },
    "inat_unobserved_podarcis_virescens_v_podarcis_guadarramae": {
        "Podarcis virescens": "Podarcis virescens",
        "Podarcis guadarramae": "Podarcis guadarramae",
    },
    "inat_unobserved_thysanotus_tuberosus_v_thysanotus_patersonii": {
        "Thysanotus patersonii": "Thysanotus patersonii",
        "Thysanotus tuberosus": "Thysanotus tuberosus",
    },
    "inat_unobserved_amanita_flavorubens_v_amanita_xanthocephala": {
        "Amanita flavorubens": "Amanita flavorubens",
        "Amanita xanthocephala": "Amanita xanthocephala",
    },
    "inat_unobserved_otiorhynchus_ovatus_v_otiorhynchus_singularis": {
        "Otiorhynchus ovatus": "Otiorhynchus ovatus",
        "Otiorhynchus singularis": "Otiorhynchus singularis",
    },
    "inat_unobserved_tillandsia_balbisiana_v_tillandsia_bartramii": {
        "Tillandsia bartramii": "Tillandsia bartramii",
        "Tillandsia balbisiana": "Tillandsia balbisiana",
    },
    "inat_unobserved_oudemansiella_mucida_v_oudemansiella_furfuracea": {
        "Oudemansiella furfuracea": "Oudemansiella furfuracea",
        "Oudemansiella mucida": "Oudemansiella mucida",
    },
    "inat_unobserved_apodemus_sylvaticus_v_apodemus_agrarius": {
        "Apodemus agrarius": "Apodemus agrarius",
        "Apodemus sylvaticus": "Apodemus sylvaticus",
    },
    "inat_unobserved_lanius_bucephalus_v_lanius_meridionalis": {
        "Lanius meridionalis": "Lanius meridionalis",
        "Lanius bucephalus": "Lanius bucephalus",
    },
    "inat_unobserved_chloris_verticillata_v_chloris_cucullata": {
        "Chloris cucullata": "Chloris cucullata",
        "Chloris verticillata": "Chloris verticillata",
    },
    "inat_unobserved_turdus_torquatus_v_turdus_atrogularis": {
        "Turdus torquatus": "Turdus torquatus",
        "Turdus atrogularis": "Turdus atrogularis",
    },
    "inat_unobserved_panus_conchatus_v_panus_neostrigosus": {
        "Panus conchatus": "Panus conchatus",
        "Panus neostrigosus": "Panus neostrigosus",
    },
    "inat_unobserved_leucorrhinia_dubia_v_leucorrhinia_rubicunda": {
        "Leucorrhinia dubia": "Leucorrhinia dubia",
        "Leucorrhinia rubicunda": "Leucorrhinia rubicunda",
    },
    "inat_unobserved_cortinarius_austrovenetus_v_cortinarius_archeri": {
        "Cortinarius austrovenetus": "Cortinarius austrovenetus",
        "Cortinarius archeri": "Cortinarius archeri",
    },
    "inat_unobserved_emberiza_pusilla_v_emberiza_leucocephalos": {
        "Emberiza pusilla": "Emberiza pusilla",
        "Emberiza leucocephalos": "Emberiza leucocephalos",
    },
    "inat_unobserved_podarcis_liolepis_v_podarcis_bocagei": {
        "Podarcis bocagei": "Podarcis bocagei",
        "Podarcis liolepis": "Podarcis liolepis",
    },
    "inat_unobserved_serinus_canaria_v_serinus_canicollis": {
        "Serinus canaria": "Serinus canaria",
        "Serinus canicollis": "Serinus canicollis",
    },
    "inat_unobserved_cladonia_squamosa_v_cladonia_portentosa": {
        "Cladonia squamosa": "Cladonia squamosa",
        "Cladonia portentosa": "Cladonia portentosa",
    },
    "inat_unobserved_lactarius_torminosus_v_lactarius_turpis": {
        "Lactarius torminosus": "Lactarius torminosus",
        "Lactarius turpis": "Lactarius turpis",
    },
    "inat_unobserved_scopula_umbilicata_v_scopula_ornata": {
        "Scopula umbilicata": "Scopula umbilicata",
        "Scopula ornata": "Scopula ornata",
    },
    "inat_unobserved_aceria_negundi_v_aceria_cephalonea": {
        "Aceria negundi": "Aceria negundi",
        "Aceria cephalonea": "Aceria cephalonea",
    },
    "inat_unobserved_hippolais_icterina_v_hippolais_polyglotta": {
        "Hippolais polyglotta": "Hippolais polyglotta",
        "Hippolais icterina": "Hippolais icterina",
    },
    "inat_unobserved_cuphea_aequipetala_v_cuphea_hyssopifolia": {
        "Cuphea aequipetala": "Cuphea aequipetala",
        "Cuphea hyssopifolia": "Cuphea hyssopifolia",
    },
}


##########
# SHARED #
##########


@beartype.beartype
def include_task(
    cfg: config.Newt, task: str, cluster: str, subcluster: str | None
) -> bool:
    """
    Determine if a task should be included based on the configuration.

    Args:
        cfg: The newt configuration
        task: The name of the task
        cluster: The cluster the task belongs to
        subcluster: The sub-cluster the task belongs to (may be None)

    Returns:
        True if the task should be included, False otherwise
    """
    # Check explicit exclusions first
    if cfg.exclude_tasks and task in cfg.exclude_tasks:
        return False

    if cfg.exclude_clusters and cluster in cfg.exclude_clusters:
        return False

    if cfg.exclude_subclusters and subcluster and subcluster in cfg.exclude_subclusters:
        return False

    # Check inclusions - if any inclusion filter is specified, the task must match at least one
    has_inclusion_filter = False

    # Check specific tasks
    if cfg.tasks:
        has_inclusion_filter = True
        if task in cfg.tasks:
            return True

    # Check clusters
    if cfg.include_clusters:
        has_inclusion_filter = True
        if cluster in cfg.include_clusters:
            return True

    # Check subclusters
    if cfg.include_subclusters and subcluster:
        has_inclusion_filter = True
        if subcluster in cfg.include_subclusters:
            return True

    # If no inclusion filters were specified, include by default
    # Otherwise, exclude because it didn't match any inclusion filter
    return not has_inclusion_filter
