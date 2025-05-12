"""
Individual re-identification of Beluga whales (*Delphinapterus leucas*) using [this LILA BC dataset](https://lila.science/datasets/beluga-id-2022/).

We use a very simple method:

1. Embed all images using a vision backbone.
2. For each image, treat it as a test image and find its nearest neighbor (k=1).
3. Give a score of 1.0 if the nearest neighbor is the same individual, otherwise 0.0.

You could improve this with nearest centroid classification, k>1, or any number of fine-tuning techniques.
But we are simply interested in seeing if models embed images of the same individual closer together in representation space.

If you use this task, please cite the original dataset paper and the paper that proposed this evaluation method:

```
@article{algasov2024understanding,
  title={Understanding the Impact of Training Set Size on Animal Re-identification},
  author={Algasov, Aleksandr and Nepovinnykh, Ekaterina and Eerola, Tuomas and K{\"a}lvi{\"a}inen, Heikki and Stewart, Charles V and Otarashvili, Lasha and Holmberg, Jason A},
  journal={arXiv preprint arXiv:2405.15976},
  year={2024}
}

@inproceedings{vcermak2024wildlifedatasets,
  title={WildlifeDatasets: An open-source toolkit for animal re-identification},
  author={{\v{C}}erm{\'a}k, Vojt{\v{e}}ch and Picek, Lukas and Adam, Luk{\'a}{\v{s}} and Papafitsoros, Kostas},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5953--5963},
  year={2024}
}
```
"""

import dataclasses
import logging
import os.path

import beartype
import numpy as np
import sklearn.neighbors
import torch
import torchvision.datasets
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor

from .. import config, helpers, registry, reporting

logger = logging.getLogger("beluga")


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    """Run the BelugaID benchmark."""
    backbone = registry.load_vision_backbone(cfg.model)

    # Embed all images.
    features = get_features(cfg, backbone)
    # Convert string names into integer labels.
    encoder = sklearn.preprocessing.OrdinalEncoder(dtype=int)
    y = encoder.fit_transform(features.labels.reshape(-1, 1)).reshape(-1)

    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform")
    clf.fit(features.x, y)
    y_hat = clf.predict(None)

    preds = [
        reporting.Prediction(
            str(img_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for img_id, pred, true in zip(features.ids, y_hat, y)
    ]

    return reporting.Report("beluga", preds, cfg)


@beartype.beartype
def score(preds: list[reporting.Prediction]) -> float:
    return reporting.macro_f1(preds)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    """A block of features."""

    x: Float[Tensor, "n dim"]
    """Input features; from a `biobench.registry.VisionBackbone`."""
    labels: Shaped[np.ndarray, " n"]
    """Individual name."""
    ids: Shaped[np.ndarray, " n"]
    """Array of image ids."""

    def y(self, encoder):
        return encoder.transform(self.labels.reshape(-1, 1)).reshape(-1)

    @property
    def n(self) -> int:
        return len(self.ids)


@beartype.beartype
def collate_fn(batch):
    imgs = torch.stack([img for img, _ in batch])
    metadata = [meta for _, meta in batch]
    return imgs, metadata


@beartype.beartype
@torch.no_grad()
def get_features(cfg: config.Experiment, backbone: registry.VisionBackbone) -> Features:
    """
    Get a block of features from a vision backbone.

    Args:
        args: BelugaID arguments.
        backbone: visual backbone.
    """
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(cfg.device))

    if not os.path.isdir(cfg.data.beluga):
        msg = f"Path '{cfg.data.beluga}' doesn't exist. Did you download the Beluga dataset?"
        raise ValueError(msg)

    dataset = torchvision.datasets.CocoDetection(
        os.path.join(cfg.data.beluga, "beluga.coco", "images", "train2022"),
        os.path.join(
            cfg.data.beluga, "beluga.coco", "annotations", "instances_train2022.json"
        ),
        img_transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_features, all_labels, all_ids = [], [], []

    def probe(batch):
        imgs, _ = batch
        imgs = imgs.to(cfg.device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            backbone.img_encode(imgs).img_features

    with helpers.auto_batch_size(dataloader, probe=probe):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for b in helpers.progress(range(total), desc="beluga"):
            imgs, metadata = next(it)
            imgs = imgs.to(cfg.device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                features = backbone.img_encode(imgs).img_features

            assert all(len(meta) == 1 for meta in metadata)
            labels = [meta[0]["name"] for meta in metadata]
            ids = [str(meta[0]["image_id"]) for meta in metadata]

            all_features.append(features.cpu())
            all_labels.extend(labels)
            all_ids.extend(ids)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_ids = np.array(all_ids)
    all_labels = np.array(all_labels)

    return Features(all_features, all_labels, all_ids)
