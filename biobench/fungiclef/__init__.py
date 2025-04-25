"""
FungiCLEF2023: classify fungal species using Danish Fungi preprocessed images.

Citations:
```
@inproceedings{BohemianVRA2023,
  title={FungiCLEF 2023 challenge evaluation},
  author={BohemianVRA},
  booktitle={ImageCLEF},
  year={2023}
}
```
"""

import dataclasses
import logging
import os
import typing

import beartype
import numpy as np
import polars as pl
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped
from PIL import Image

from .. import config, helpers, openset, registry, reporting
from .metrics import user_loss_score

logger = logging.getLogger("fungiclef")


USER_LOSS_SCORER = sklearn.metrics.make_scorer(user_loss_score, greater_is_better=False)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
class FungiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        image_names: np.ndarray,
        labels: np.ndarray,
        transform,
    ):
        self.root = root
        self.names = list(image_names)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict[str, object]:
        name = self.names[idx]
        label = int(self.labels[idx])
        # try exact case, then lowercase
        p1 = os.path.join(self.root, name)
        if os.path.exists(p1):
            path = p1
        else:
            p2 = os.path.join(self.root, name.lower())
            if os.path.exists(p2):
                path = p2
                logger.debug("Using lowercase image path for %s", name)
            else:
                raise FileNotFoundError(
                    f"Image '{name}' not found in {self.root} (tried '{p1}', '{p2}')"
                )
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"img_id": name, "img": img, "label": label}


@beartype.beartype
def get_features(
    cfg: config.Experiment,
    backbone: registry.VisionBackbone,
    *,
    split: typing.Literal["train", "val"],
) -> Features:
    # read metadata
    df = pl.read_csv(
        os.path.join(
            cfg.data.fungiclef,
            f"FungiCLEF2023_{split}_metadata_PRODUCTION.csv",
        )
    )
    img_names = df.get_column("image_path").to_numpy()
    labels = df.get_column("class_id").to_numpy().astype(int)

    # subsample for train
    if split == "train" and cfg.n_train > 0:
        idxs = helpers.balanced_random_sample(labels, cfg.n_train)
        img_names = img_names[idxs]
        labels = labels[idxs]

    img_dir = os.path.join(
        cfg.data.fungiclef,
        "DF20_300" if split == "train" else "DF21_300",
    )
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"Image directory not found: {img_dir}")

    transform = backbone.make_img_transform()
    dataset = FungiDataset(img_dir, img_names, labels, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        pin_memory=True,
    )

    backbone = torch.compile(backbone.to(cfg.device))
    feats_list, labs_list, ids_list = [], [], []

    total = len(dataloader) if not cfg.debug else 2

    it = iter(dataloader)
    for b in helpers.progress(range(total), every=10, desc=f"fungi/{split}"):
        batch = next(it)
        imgs = batch["img"].to(cfg.device)
        with torch.amp.autocast("cuda"):
            out = backbone.img_encode(imgs).img_features.cpu().numpy()
        feats_list.append(out)
        labs_list.append(np.array(batch["label"], dtype=int))
        ids_list.extend(batch["img_id"])

    x = np.concatenate(feats_list, axis=0)
    y = np.concatenate(labs_list, axis=0)
    ids = np.array(ids_list)
    return Features(x, y, ids)


@beartype.beartype
@torch.no_grad()
def benchmark(cfg: config.Experiment) -> reporting.Report:
    backbone = registry.load_vision_backbone(cfg.model)
    train_feats = get_features(cfg, backbone, split="train")
    val_feats = get_features(cfg, backbone, split="val")

    clf = init_clf(cfg)
    clf = openset.MahalanobisOpenSetClassifier(clf)
    clf.fit(train_feats.x, train_feats.y)

    preds = clf.predict(val_feats.x)
    # Calculate a set of train and test classes using np.unique and set(), then add a field to the info dict in Prediction that marks whether it is an OOD example (y_true not in train classes). AI!
    examples = [
        reporting.Prediction(
            img_id,
            float(p == t),
            {"y_pred": int(p), "y_true": int(t)},
        )
        for img_id, p, t in zip(val_feats.ids, preds, val_feats.y)
    ]
    return reporting.Report("fungiclef", examples, cfg)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    alphas = np.pow(2.0, np.arange(-15, 5))
    if cfg.debug:
        alphas = np.pow(2.0, np.arange(-2, 2))
    if 0 < cfg.n_train < 2_713 * 5 * 2:
        return sklearn.linear_model.RidgeClassifier()

    pipe = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.RidgeClassifier(1.0),
    )
    return sklearn.model_selection.HalvingGridSearchCV(
        pipe,
        {"ridgeclassifier__alpha": alphas},
        scoring=USER_LOSS_SCORER,
        n_jobs=16,
        verbose=2,
        factor=3,
        random_state=cfg.seed,
    )
