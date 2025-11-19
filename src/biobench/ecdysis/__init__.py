"""
# Ecdysis Foundation Insects

Trains a simple logistic regression classifier on species classification for the publicly available Ecdysis Foundation insect images.

This task is a true, real-world task.
"""

import collections.abc
import csv
import dataclasses
import functools
import logging
import os
import os.path
import typing as tp

import beartype
import numpy as np
import polars as pl
import torch
from jaxtyping import Float, Float16, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from .. import config, helpers, registry, reporting, linear_probing

logger = logging.getLogger(__name__)


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    # 1. Load model
    backbone = registry.load_vision_backbone(cfg.model)
    backbone = backbone.to(cfg.device)

    # 2. Load data.
    train_features = get_features(cfg, backbone, is_train=True)
    test_features = get_features(cfg, backbone, is_train=False)
    torch.cuda.empty_cache()

    # 3. Do classification.
    clf = linear_probing.LinearProbeClassifier(device=cfg.device)
    clf.fit(train_features.x, train_features.y)

    true_labels = test_features.y
    pred_labels = clf.predict(test_features.x)

    # Return benchmark report.
    preds = [
        reporting.Prediction(
            str(video_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for video_id, pred, true in zip(test_features.ids, pred_labels, true_labels)
    ]
    return reporting.Report("ecdysis", preds, cfg)


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
    assert df.get_column("task_name").unique().to_list() == ["ecdysis"]
    return reporting.bootstrap_scores_macro_f1(df, b=b, rng=rng)


@jaxtyped(typechecker=beartype.beartype)
class Sample(tp.TypedDict):
    """A dictionary representing a single image sample with its metadata.

    Attributes:
        img_id: Unique identifier for the image.
        img: The image tensor with shape [3, width, height] (RGB channels first).
        label: Binary class label (0 or 1) for the image.
    """

    img_id: str
    img: Float[Tensor, "3 width height"]
    label: Int[Tensor, ""]


@beartype.beartype
class Dataset(torch.utils.data.Dataset):
    """Ecdysis Foundation dataset."""

    def __init__(
        self,
        root: str,
        split: tp.Literal["train", "validation"],
        transform: tp.Callable | None = None,
    ):
        """Initialize the Ecdysis dataset.

        Args:
            root: Path to the dataset directory (e.g., /path/to/ecdysis). The actual data is in images/ and metadata.csv file.
            split: which split to use.
            transform: Optional image transform to apply.
        """
        # Extract split name from path (train or validation)
        self.split = split
        msg = f"Split must be train or validation, got {self.split}"
        assert self.split in ("train", "validation"), msg

        # Base directory is the parent (where metadata.csv and images/ are located)
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(self.root, "images")

        # Load and parse metadata
        metadata_fpath = os.path.join(self.root, "metadata.csv")
        if not os.path.exists(metadata_fpath):
            msg = f"Metadata file not found at {metadata_fpath}"
            raise FileNotFoundError(msg)

        # Read metadata and create samples
        label_to_id = {}
        samples_by_label = {}  # Group samples by label for per-species splitting

        with open(metadata_fpath) as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                # Skip rows with empty image paths
                if not row["image_thumbnail_large"]:
                    continue

                # Create unique label from taxonomic hierarchy: class|order|family|genus|species
                label_str = "|".join([
                    row["gbif_class"],
                    row["gbif_order"],
                    row["gbif_family"],
                    row["gbif_genus"],
                    row["gbif_species"],
                ])

                # Map label string to integer ID
                if label_str not in label_to_id:
                    label_to_id[label_str] = len(label_to_id)

                # Extract image filename from image_thumbnail_large path (e.g., specimen_images/foo.JPG -> foo.JPG)
                img_filename = os.path.basename(row["image_thumbnail_large"])
                img_fpath = os.path.join(self.images_dir, img_filename)

                # Use 'id' as the sample identifier
                sample_id = row["id"]

                # Store visit_date for temporal splitting
                visit_date = row["visit_date"]

                label_id = label_to_id[label_str]

                # Group by label
                if label_id not in samples_by_label:
                    samples_by_label[label_id] = []
                samples_by_label[label_id].append((img_fpath, label_id, sample_id, visit_date))

        # Create temporal train/validation split per species
        # For each label, sort by date and split 80/20
        train_samples = []
        val_samples = []

        for label_id, label_samples in samples_by_label.items():
            # Sort by visit_date (earliest to latest)
            label_samples.sort(key=lambda x: x[3])

            # Split: earliest 80% for train, latest 20% for validation
            split_idx = int(0.8 * len(label_samples))

            train_samples.extend([(fpath, label, sid) for fpath, label, sid, _ in label_samples[:split_idx]])
            val_samples.extend([(fpath, label, sid) for fpath, label, sid, _ in label_samples[split_idx:]])

        if self.split == "train":
            self.samples = train_samples
        else:  # validation
            self.samples = val_samples
        logger.info(
            "Loaded %d samples for %s split from %s",
            len(self.samples),
            self.split,
            self.root,
        )

    def __getitem__(self, index: int) -> Sample:
        img_fpath, label, sample_id = self.samples[index]

        # Load and transform image
        img = Image.open(img_fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return {
            "img_id": sample_id,
            "img": img,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> Int[np.ndarray, " n"]:
        return np.array([label for _, label, _ in self.samples])


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    img_transform = backbone.make_img_transform()
    backbone = backbone.to(cfg.device)

    split = "train" if is_train else "validation"
    dataset = Dataset(cfg.data.ecdysis, split, img_transform)

    if is_train and cfg.n_train > 0:
        i = helpers.balanced_random_sample(dataset.labels, cfg.n_train)
        assert len(i) == cfg.n_train
        dataset = torch.utils.data.Subset(dataset, i)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )

    def probe(batch):
        imgs = batch["img"].to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            # forward only
            _ = backbone.img_encode(imgs).img_features

    all_ids, all_features, all_labels = [], [], []
    # Set an upper limit. Otherwise we spend a lot of time picking an optimal batch size when we could just rip through the dataset.
    with helpers.auto_batch_size(dataloader, probe=probe, upper=512):
        backbone = torch.compile(backbone)

        for batch in helpers.progress(dataloader, every=10, desc=f"ecdysis/{split}"):
            imgs = batch["img"].to(cfg.device, non_blocking=True)

            with torch.amp.autocast(cfg.device):
                features = backbone.img_encode(imgs).img_features
                all_features.append(features.cpu())

            all_ids.extend(batch["img_id"])
            all_labels.extend(batch["label"])

    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    all_labels = np.array(all_labels)
    all_ids = np.array(all_ids)
    assert len(all_ids) == len(dataset)
    logger.info("Got features for %d images.", len(all_ids))

    return Features(all_features, all_labels, all_ids)
