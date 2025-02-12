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

from biobench import helpers, interfaces, registry

logger = logging.getLogger("beluga")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    """Configuration for BelugaID task."""

    batch_size: int = 256
    """Batch size for the vision backbone."""
    n_workers: int = 8
    """Number of dataloader workers."""
    log_every: int = 10
    """How often to log while getting features."""


@beartype.beartype
def benchmark_cvml(
    args: Args, model_args: interfaces.ModelArgsCvml
) -> tuple[interfaces.ModelArgsCvml, interfaces.TaskReport]:
    """
    Run the BelugaID benchmark. See this module's documentation for more details.
    """
    backbone = registry.load_vision_backbone(*model_args)

    # Embed all images.
    features = get_features(args, backbone)
    # Convert string names into integer labels.
    encoder = sklearn.preprocessing.OrdinalEncoder(dtype=int)
    y = encoder.fit_transform(features.labels.reshape(-1, 1)).reshape(-1)

    clf = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    clf.fit(features.x, y)
    preds = clf.kneighbors(return_distance=False)

    logger.info("Constructing examples.")
    examples = [
        interfaces.Example(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(
            helpers.progress(features.ids, every=1_000), preds, y
        )
    ]
    logger.info("%d examples done.", len(examples))

    return model_args, interfaces.TaskReport("BelugaID", examples)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    """
    A block of features.

    Note: In Jax, this could be a tuple of arrays, all with a leading dimension of `n`. Instead, in PyTorch, it's easier to make it its own class. Oh well.
    """

    x: Float[Tensor, "n dim"]
    """Input features; from a `biobench.interfaces.VisionBackbone`."""
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
@torch.no_grad
def get_features(args: Args, backbone: interfaces.VisionBackbone) -> Features:
    """
    Get a block of features from a vision backbone.

    Args:
        args: BelugaID arguments.
        backbone: visual backbone.
    """
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    if not os.path.isdir(args.datadir):
        msg = f"Path '{args.datadir}' doesn't exist. Did you download the Beluga dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--beluga-args.datadir'; see --help for more."
        raise ValueError(msg)

    dataset = torchvision.datasets.CocoDetection(
        os.path.join(args.datadir, "beluga.coco", "images", "train2022"),
        os.path.join(
            args.datadir, "beluga.coco", "annotations", "instances_train2022.json"
        ),
        img_transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
        shuffle=True,  # We use dataset.shuffle instead
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    all_features, all_labels, all_ids = [], [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    for b in helpers.progress(range(total), every=args.log_every, desc="embed"):
        images, metadata = next(it)
        images = torch.stack(images).to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features

        labels = [meta[0]["name"] for meta in metadata]
        ids = [str(meta[0]["image_id"]) for meta in metadata]

        all_features.append(features.cpu())
        all_labels.extend(labels)
        all_ids.extend(ids)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_ids = np.array(all_ids)
    all_labels = np.array(all_labels)

    return Features(all_features, all_labels, all_ids)
