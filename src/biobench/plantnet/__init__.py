"""
Pl@ntNet is a "dataset with high label ambiguity and a long-tailed distribution" from NeurIPS 2021.
We fit a ridge classifier from scikit-learn to a backbone's embeddings and evaluate on the validation split.


There are two pieces that make Pl@ntNet more than a simple classification task:

1. Because of the long tail, we use `class_weight='balanced'` which adjusts weights based on class frequency.
2. We use macro F1 both to choose the alpha parameter and to evaluate the final classifier rather than accuracy due to the massive class imbalance.

If you use this task, please cite the original paper:

@inproceedings{plantnet-300k,
    author={Garcin, Camille and Joly, Alexis and Bonnet, Pierre and Lombardo, Jean-Christophe and Affouard, Antoine and Chouet, Mathias and Servajean, Maximilien and Lorieul, Titouan and Salmon, Joseph},
    booktitle={NeurIPS Datasets and Benchmarks 2021},
    title={{Pl@ntNet-300K}: a plant image dataset with high label ambiguity and a long-tailed distribution},
    year={2021},
}
"""

import dataclasses
import logging
import os
import typing

import beartype
import numpy as np
import polars as pl
import sklearn.preprocessing
import torch
from jaxtyping import Float, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from .. import config, helpers, linear_probing, registry, reporting

logger = logging.getLogger("plantnet")


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    labels: Shaped[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]

    def y(self, encoder):
        return encoder.transform(self.labels.reshape(-1, 1)).reshape(-1)


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    """
    Steps:
    1. Get features for all images.
    2. Select lambda using cross validation splits.
    3. Report score on test data.
    """
    backbone = registry.load_vision_backbone(cfg.model)

    # 1. Get features
    val_features = get_features(cfg, backbone, split="val")
    train_features = get_features(cfg, backbone, split="train")
    torch.cuda.empty_cache()

    encoder = sklearn.preprocessing.OrdinalEncoder()
    all_labels = np.concatenate((val_features.labels, train_features.labels))
    encoder.fit(all_labels.reshape(-1, 1))

    # 2. Fit model.
    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y(encoder))

    true_labels = val_features.y(encoder)
    pred_labels = clf.predict(val_features.x)

    preds = [
        reporting.Prediction(
            str(img_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for img_id, pred, true in zip(val_features.ids, pred_labels, true_labels)
    ]

    return reporting.Report("plantnet", preds, cfg)


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
    assert df.get_column("task_name").unique().to_list() == ["plantnet"]
    return reporting.bootstrap_scores_macro_f1(df, b=b, rng=rng)


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    transform: typing.Any | None
    """Optional function function that transforms an image into a format expected by a neural network."""
    samples: list[tuple[str, str, str]]
    """List of all image ids, image paths, and classnames."""

    def __init__(self, root: str, transform):
        self.transform = transform
        self.samples = []
        if not os.path.exists(root) or not os.path.isdir(root):
            msg = f"Path '{root}' doesn't exist. Did you download the Pl@ntNet dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path as --dataset-dir PATH"
            raise RuntimeError(msg)

        for dirpath, dirnames, filenames in os.walk(root):
            img_class = os.path.relpath(dirpath, root)
            for filename in filenames:
                img_id = filename.removesuffix(".jpg")
                img_path = os.path.join(dirpath, filename)
                self.samples.append((img_id, img_path, img_class))

    def __getitem__(self, i: int) -> tuple[str, Float[Tensor, "3 width height"], str]:
        img_id, img_path, img_class = self.samples[i]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img_id, img, img_class

    def __len__(self) -> int:
        return len(self.samples)


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, split: str
) -> Features:
    imgs_dir_path = os.path.join(cfg.data.plantnet, "images", split)

    img_transform = backbone.make_img_transform()
    backbone = backbone.to(cfg.device)

    dataset = Dataset(imgs_dir_path, img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )

    all_ids, all_features, all_labels = [], [], []

    def probe(batch):
        _, imgs, _ = batch
        imgs = imgs.to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            backbone.img_encode(imgs).img_features  # forward only

    with helpers.auto_batch_size(dataloader, probe=probe):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        backbone = torch.compile(backbone)
        for b in helpers.progress(range(total), every=10, desc=f"plnt/{split}"):
            ids, imgs, labels = next(it)
            imgs = imgs.to(cfg.device)

            with torch.amp.autocast(cfg.device):
                features = backbone.img_encode(imgs).img_features
                all_features.append(features.cpu())

            all_ids.extend(ids)
            all_labels.extend(labels)

    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    all_labels = np.array(all_labels)
    all_ids = np.array(all_ids)

    assert len(all_ids) == len(dataset)
    if cfg.n_train >= 0:
        assert len(all_ids) == cfg.n_train
    logger.info("Got features for %d images.", len(all_ids))

    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    clf = linear_probing.LinearProbeClassifier(device=cfg.device)
    return clf
