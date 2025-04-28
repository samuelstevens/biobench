"""
Herbarium19: classify specimens into species based on the 2019 FGVC6 competition.

```
@article{tan2019herbarium,
  title={The herbarium challenge 2019 dataset},
  author={Tan, Kiat Chuan and Liu, Yulong and Ambrose, Barbara and Tulig, Melissa and Belongie, Serge},
  journal={arXiv preprint arXiv:1906.05372},
  year={2019}
}
```
"""

import dataclasses
import logging
import os
import typing

import beartype
import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
import torchvision.datasets
from jaxtyping import Float, Int, Shaped, jaxtyped
from torch import Tensor

from .. import config, helpers, registry, reporting

logger = logging.getLogger("herbarium19")


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


@beartype.beartype
class Dataset(torchvision.datasets.ImageFolder):
    """ImageFolder but returns Sample."""

    def __getitem__(self, index) -> Sample:
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return {"img_id": path, "img": img, "label": label}

    @property
    def labels(self) -> Int[np.ndarray, " n"]:
        return np.array([label for _, label in self.samples])


@beartype.beartype
@torch.no_grad()
def benchmark(cfg: config.Experiment) -> reporting.Report:
    backbone = registry.load_vision_backbone(cfg.model)

    train_feats = get_features(cfg, backbone, is_train=True)
    test_feats = get_features(cfg, backbone, is_train=False)

    clf = init_clf(cfg)
    clf.fit(train_feats.x, train_feats.y)

    if hasattr(clf, "best_params_"):
        helpers.write_hparam_sweep_plot("herbarium19", cfg.model.ckpt, clf)

    preds = clf.predict(test_feats.x)
    examples = [
        reporting.Prediction(
            img_id, float(p == t), {"y_pred": p.item(), "y_true": t.item()}
        )
        for img_id, p, t in zip(test_feats.ids, preds, test_feats.y)
    ]
    return reporting.Report("herbarium19", examples, cfg)


@beartype.beartype
def score(preds: list[reporting.Prediction]) -> float:
    return reporting.macro_f1(preds)


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
    split = "train" if is_train else "validation"
    images_dir_path = os.path.join(cfg.data.herbarium19, split)

    img_transform = backbone.make_img_transform()
    dataset = Dataset(images_dir_path, img_transform)

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

    backbone = torch.compile(backbone.to(cfg.device))

    def probe(batch):
        imgs = batch["img"].to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            _ = backbone.img_encode(imgs).img_features  # forward only

    all_ids, all_features, all_labels = [], [], []
    with helpers.auto_batch_size(dataloader, probe=probe):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for b in helpers.progress(range(total), every=10, desc=f"hb19/{split}"):
            batch = next(it)
            imgs = batch["img"].to(cfg.device)

            with torch.amp.autocast("cuda"):
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


@beartype.beartype
def init_clf(cfg: config.Experiment):
    alpha = np.pow(2.0, np.arange(-15, 5))
    if cfg.debug:
        alpha = np.pow(2.0, np.arange(-2, 2))

    if 0 < cfg.n_train <= 300:
        return sklearn.linear_model.RidgeClassifier()

    return sklearn.model_selection.HalvingGridSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.RidgeClassifier(1.0),
        ),
        {"ridgeclassifier__alpha": alpha},
        n_jobs=16,
        verbose=2,
        factor=3,
        random_state=cfg.seed,
        scoring="f1_macro",
    )
