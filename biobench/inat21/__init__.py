"""
Trains a simple ridge regression classifier on visual representations for the iNat21 challenge.
In the challenge, there are 10K different species (classes).
We use the mini training set with 50 images per species, and test on the validation set, which has 10 images per species.

This task is a benchmark: it should help you understand how general a vision backbone's representations are.
This is not a true, real-world task.

If you use this task, be sure to cite the original iNat21 dataset paper:

```
@misc{inat2021,
  author={Van Horn, Grant and Mac Aodha, Oisin},
  title={iNat Challenge 2021 - FGVC8},
  publisher={Kaggle},
  year={2021},
  url={https://kaggle.com/competitions/inaturalist-2021}
}
```
"""

import dataclasses
import logging
import os

import beartype
import numpy as np
import polars as pl
import torch
import torchvision.datasets
from jaxtyping import Float, Float16, Int, Shaped, jaxtyped

from .. import config, helpers, linear_probing, registry, reporting

logger = logging.getLogger("inat21")

n_classes = 10_000


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float16[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    """
    Steps:
    1. Get features for all images.
    2. Select lambda using validation data.
    3. Report score on test data.
    """
    backbone = registry.load_vision_backbone(cfg.model)

    # 1. Get features
    val_features = get_features(cfg, backbone, is_train=False)
    train_features = get_features(cfg, backbone, is_train=True)

    # 2. Fit model.
    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)

    true_labels = val_features.y
    pred_labels = clf.predict(val_features.x)

    preds = [
        reporting.Prediction(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(val_features.ids, pred_labels, true_labels)
    ]

    return reporting.Report("inat21", preds, cfg)


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
    assert df.get_column("task_name").unique().to_list() == ["inat21"]

    # For some reason, one of my models only has 49.3K predictions, so I only use that many. Compared to 50K it's probably fine.
    n = 100_000

    if b > 0:
        assert rng is not None, "must provide rng argument"
        i_bs = rng.integers(0, n, size=(b, n), dtype=np.int32)

    scores = {}

    correct_buf = np.zeros((b, n), dtype=bool)

    for model_ckpt in df.get_column("model_ckpt").unique().sort().to_list():
        # pull y_true and y_pred for *one* model
        y_pred = (
            df.filter(pl.col("model_ckpt") == model_ckpt)
            .select("img_id", "y_pred")
            .unique()
            .sort("img_id")
            .get_column("y_pred")
            .cast(pl.Int32)
            .to_numpy()
        )

        if len(y_pred) == 0:
            continue

        y_true = (
            df.filter(pl.col("model_ckpt") == model_ckpt)
            .select("img_id", "y_true")
            .unique()
            .sort("img_id")
            .get_column("y_true")
            .cast(pl.Int32)
            .to_numpy()
        )
        assert y_true.size == y_pred.size

        if b > 0:
            # bootstrap resample into pre-allocated buffers
            np.take(y_pred == y_true, i_bs, axis=0, out=correct_buf)
            scores[model_ckpt] = correct_buf.mean(axis=1)
        else:
            scores[model_ckpt] = np.array([(y_pred == y_true).mean()])

    return scores


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torchvision.datasets.ImageFolder):
    """
    Subclasses ImageFolder so that `__getitem__` includes the path, which we use as the ID.
    """

    def __getitem__(self, index: int) -> tuple[str, object, object]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (path, sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target

    @property
    def labels(self) -> Int[np.ndarray, " n"]:
        return np.array([label for _, label in self.samples])


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    img_transform = backbone.make_img_transform()
    backbone = backbone.to(cfg.device)
    split = "train_mini" if is_train else "val"

    root = os.path.join(cfg.data.inat21, split)

    if not os.path.isdir(root):
        msg = f"Path '{root}' doesn't exist. Did you download the iNat21 dataset?"
        raise ValueError(msg)

    dataset = Dataset(root, img_transform)

    if is_train and cfg.n_train > 0:
        i = helpers.balanced_random_sample(dataset.labels, cfg.n_train)
        assert len(i) == cfg.n_train
        dataset = torch.utils.data.Subset(dataset, i)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=True,
    )

    all_ids, all_features, all_labels = [], [], []

    def probe(batch):
        ids, imgs, labels = batch
        imgs = imgs.to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            backbone.img_encode(imgs).img_features

    with helpers.auto_batch_size(dataloader, probe=probe):
        backbone = torch.compile(backbone)
        for ids, images, labels in helpers.progress(dataloader, desc=f"inat21/{split}"):
            images = images.to(cfg.device, non_blocking=True)

            with torch.amp.autocast(cfg.device):
                features = backbone.img_encode(images).img_features

            all_features.append(features.cpu())
            all_labels.extend(labels)
            all_ids.extend(ids)

    all_features = torch.cat(all_features, dim=0).cpu().to(torch.float16).numpy()
    all_ids = np.array(all_ids)
    all_labels = torch.tensor(all_labels).numpy()
    if is_train and cfg.n_train > 0:
        assert len(all_ids) == cfg.n_train

    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    clf = linear_probing.LinearProbeClassifier(device=cfg.device)
    return clf
