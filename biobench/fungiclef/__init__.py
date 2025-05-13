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
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped
from PIL import Image

from .. import config, helpers, openset, registry, reporting, simpleshot
from . import metrics

logger = logging.getLogger("fungiclef")


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]
    """image IDs for validation, observation IDs for training."""


@beartype.beartype
class FungiDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: typing.Literal["train", "val"], transform):
        img_dpath = os.path.join(root, "DF20_300" if split == "train" else "DF21_300")
        if not os.path.isdir(img_dpath):
            raise RuntimeError(f"Image directory not found: {img_dpath}")

        csv_fpath = os.path.join(root, f"FungiCLEF2023_{split}_metadata_PRODUCTION.csv")
        if not os.path.isfile(csv_fpath):
            raise RuntimeError(f"CSV not found: {csv_fpath}")

        df = pl.read_csv(csv_fpath)
        self.img_names = df.get_column("image_path").to_numpy()
        self.labels = df.get_column("class_id").to_numpy().astype(int)
        self.obs_ids = df.get_column("observationID").to_numpy()

        self.img_dpath = img_dpath
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx) -> dict[str, object]:
        img_name = self.img_names[idx]
        label = self.labels[idx].item()
        # try exact case, then lowercase
        p1 = os.path.join(self.img_dpath, img_name)
        if os.path.exists(p1):
            path = p1
        else:
            p2 = os.path.join(self.img_dpath, img_name.lower())
            if os.path.exists(p2):
                path = p2
                # logger.debug("Using lowercase image path for %s", img_name)
            else:
                raise FileNotFoundError(
                    f"Image '{img_name}' not found in {self.img_dpath} (tried '{p1}', '{p2}')"
                )
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return {
            "img_id": img_name,
            "img": img,
            "label": label,
            "obs_id": self.obs_ids[idx].item(),
        }


@beartype.beartype
@torch.inference_mode()
def get_features(
    cfg: config.Experiment,
    backbone: registry.VisionBackbone,
    *,
    is_train: bool,
    pool: bool,
) -> Features:
    transform = backbone.make_img_transform()

    split = "train" if is_train else "val"
    dataset = FungiDataset(cfg.data.fungiclef, split, transform)

    # subsample for train
    if is_train and cfg.n_train > 0:
        idxs = helpers.balanced_random_sample(dataset.labels, cfg.n_train)
        dataset = torch.utils.data.Subset(dataset, idxs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        pin_memory=False,
    )

    backbone = backbone.to(cfg.device)
    feats_list, labs_list, img_ids_list, obs_ids_list = [], [], [], []

    @beartype.beartype
    def debug_cuda_mem(tag: str):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s: %s", tag, torch.cuda.memory_summary())

    def probe(batch):
        imgs = batch["img"].to(cfg.device)
        with torch.amp.autocast(cfg.device):
            backbone.img_encode(imgs)

    with helpers.auto_batch_size(dataloader, probe=probe):
        backbone = torch.compile(backbone)
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for _ in helpers.progress(range(total), desc=f"fungi/{split}"):
            debug_cuda_mem("loop start")
            batch = next(it)
            debug_cuda_mem("after batch")
            imgs = batch["img"].to(cfg.device)
            debug_cuda_mem("imgs.to(device)")
            with torch.amp.autocast(cfg.device):
                out = backbone.img_encode(imgs).img_features
            debug_cuda_mem("after forward pass")
            feats_list.append(out.cpu().numpy())
            debug_cuda_mem("appended feats")

            # I was getting some CUDA OOM errors due to PyTorch reserving/allocating and memory fragmentation. Rather than adjust the recommended env variable, I added this line manual GC.
            del out
            torch.cuda.empty_cache()

            debug_cuda_mem("emptied cache")
            labs_list.extend(batch["label"].tolist())
            img_ids_list.extend(batch["img_id"])
            obs_ids_list.extend(batch["obs_id"].tolist())
            debug_cuda_mem("appended metadata")

    x = np.concatenate(feats_list, axis=0)
    y = np.array(labs_list)
    img_ids = np.array(img_ids_list)
    obs_ids = np.array(obs_ids_list)

    if pool:
        # for each unique obs take mean of its image features
        uniq, inv = np.unique(obs_ids, return_inverse=True)
        pooled = np.empty((len(uniq), x.shape[1]), dtype=x.dtype)
        for k, u in enumerate(helpers.progress(uniq, every=10_000, desc="groupby")):
            pooled[k] = x[inv == k].mean(axis=0)
        # labels should be identical within an observation
        pooled_y = np.array([y[inv == k][0] for k in range(len(uniq))], dtype=int)
        return Features(pooled, pooled_y, uniq)

    # image-level output
    return Features(x, y, img_ids)


@beartype.beartype
@torch.inference_mode()
def benchmark(cfg: config.Experiment) -> reporting.Report:
    backbone = registry.load_vision_backbone(cfg.model)
    train_feats = get_features(cfg, backbone, is_train=True, pool=False)
    val_feats = get_features(cfg, backbone, is_train=False, pool=True)

    torch.cuda.empty_cache()  # be nice to others on the machine.

    clf = init_clf(cfg)
    clf.fit(train_feats.x, train_feats.y)

    logger.info("Classifying %d examples.", len(val_feats.x))
    preds = clf.predict(val_feats.x)
    logger.info("Classified %d examples.", len(val_feats.x))

    # Identify train and test classes
    train_classes = set(np.unique(train_feats.y))

    examples = [
        reporting.Prediction(
            str(img_id),
            float(p == t),
            {"y_pred": int(p), "y_true": int(t), "ood": t not in train_classes},
        )
        for img_id, p, t in zip(val_feats.ids, preds, val_feats.y)
    ]
    return reporting.Report("fungiclef", examples, cfg)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    return openset.MahalanobisOpenSetClassifier(
        simpleshot.SimpleShotClassifier(device="cuda:0")
    )


@beartype.beartype
def score(preds: list[reporting.Prediction]) -> float:
    """
    Return the **User-Focused Loss** used in FungiCLEF:
        user_loss = classification_error + PSC/ESC cost

    Notes
    -----
    * `info['y_true']` and `info['y_pred']` are ints; unknown is -1.
    * The helper in metrics.py already combines CE and PSC/ESC.
    """
    y_true = np.fromiter(
        (-1 if p.info["ood"] else p.info["y_true"] for p in preds), dtype=int
    )
    y_pred = np.fromiter((p.info["y_pred"] for p in preds), dtype=int)
    user_loss = metrics.user_loss_score(y_true, y_pred)

    n = y_true.size
    n_unknown = (y_true == -1).sum()
    n_poisonous = np.isin(y_true, metrics.POISONOUS_SPECIES).sum()

    cost_unknown_mis = 10.0
    cost_psc = 100.0
    cost_esc = 1.0

    worst_ce = (cost_unknown_mis - 1) * n_unknown / n + 1
    worst_psc = (cost_psc * n_poisonous + cost_esc * (n - n_poisonous)) / n

    score = 1 - user_loss / (worst_ce + worst_psc)
    return score
