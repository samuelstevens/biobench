"""
Classification of phytoplankton using ridge classifiers.
This task is particularly challenging because the image distribution is very different to typical pre-training datasets; it's all microscopic images in mono-channel (black and white).

If you use this task, please cite the original paper to propose this train/test split and the original datasets as well:

Paper:

```
@article{kaisa2022towards,
    author={Kraft, Kaisa  and Velhonoja, Otso  and Eerola, Tuomas  and Suikkanen, Sanna  and Tamminen, Timo  and Haraguchi, Lumi  and Ylöstalo, Pasi  and Kielosto, Sami  and Johansson, Milla  and Lensu, Lasse  and Kälviäinen, Heikki  and Haario, Heikki  and Seppälä, Jukka },
    title={Towards operational phytoplankton recognition with automated high-throughput imaging, near-real-time data processing, and convolutional neural networks},
    journal={Frontiers in Marine Science},
    volume={9},
    year={2022},
    url={https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.867695},
    doi={10.3389/fmars.2022.867695},
    issn={2296-7745},
}
```

Training data:

```
@misc{kaisa2022syke
    doi = {10.23728/B2SHARE.ABF913E5A6AD47E6BAA273AE0ED6617A},
    url = {https://b2share.eudat.eu/records/abf913e5a6ad47e6baa273ae0ed6617a},
    author = {Kraft, Kaisa and Velhonoja, Otso and Seppälä, Jukka and Hällfors, Heidi and Suikkanen, Sanna and Ylöstalo, Pasi and Anglès, Sílvia and Kielosto, Sami and Kuosa, Harri and Lehtinen, Sirpa and Oja, Johanna and Tamminen, Timo},
    keywords = {3.1.21 -> Biology -> Marine biology, phytoplankton image data set, imaging flow cytometry, Imaging FlowCytobot, IFCB, phytoplankton, Baltic Sea, image data, SYKE, Finnish Environment Institute, Marine Research Centre, Marine Ecological Research Laboratory, plankton image data, FINMARI},
    title = {SYKE-plankton_IFCB_2022},
    publisher = {https://b2share.eudat.eu},
    year = {2022},
    copyright = {open}
}
```

Evaluation data:

```
@misc{kaisa2021syke,
  doi = {10.23728/B2SHARE.7C273B6F409C47E98A868D6517BE3AE3},
  url = {https://b2share.eudat.eu/records/7c273b6f409c47e98a868d6517be3ae3},
  author = {Kraft, Kaisa and Haraguchi, Lumi and Velhonoja, Otso and Seppälä, Jukka},
  keywords = {3.1.21 -> Biology -> Marine biology, phytoplankton image data set, imaging flow cytometry, Imaging FlowCytobot, IFCB, Baltic Sea, image data, SYKE, Finnish Environment Institute, Marine Research Centre, Marine Ecological Research Laboratory, plankton image data, FINMARI, phytoplankton},
  title = {SYKE-plankton_IFCB_Utö_2021},
  publisher = {https://b2share.eudat.eu},
  year = {2022},
  copyright = {open}
}
```

This task was added because of interesting conversations with [Ekaterina Nepovinnykh](https://scholar.google.com/citations?user=lmYki4gAAAAJ) and [Heikki Kälviäinen](https://www.lut.fi/en/profiles/heikki-kalviainen).
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
from torch import Tensor

from .. import config, helpers, linear_probing, registry, reporting

logger = logging.getLogger("plankton")


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


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
    train_features = get_features(cfg, backbone, is_train=True)
    val_features = get_features(cfg, backbone, is_train=False)

    torch.cuda.empty_cache()  # Be nice to others on the machine.

    # 2. Fit model.
    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)

    # 3. Predict.
    pred_labels = clf.predict(val_features.x)
    logger.info("Predicted classes for %d examples.", len(val_features.x))
    true_labels = val_features.y

    preds = [
        reporting.Prediction(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(val_features.ids, pred_labels, true_labels)
    ]

    return reporting.Report("plankton", preds, cfg)


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
    assert df.get_column("task_name").unique().to_list() == ["plankton"]
    return reporting.bootstrap_scores_macro_f1(df, b=b, rng=rng)


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
            msg = f"Path '{root}' doesn't exist. Did you download the plankton dataset? See the docstring at the top of this file for instructions."
            raise RuntimeError(msg)

        class_to_int = {}
        for dirname in sorted(os.listdir(root)):
            class_to_int[dirname] = len(class_to_int)

        for dirpath, dirnames, filenames in os.walk(root):
            img_class = os.path.relpath(dirpath, root)
            for filename in filenames:
                if not filename.endswith(".png"):
                    continue
                img_id = filename.removesuffix(".png")
                img_path = os.path.join(dirpath, filename)
                self.samples.append((img_id, img_path, class_to_int[img_class]))

    def __getitem__(self, i) -> Sample:
        img_id, img_path, label = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"img_id": img_id, "img": img, "label": label}

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> Int[np.ndarray, " n_samples"]:
        return np.array([label for _, _, label in self.samples])


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    split = "train" if is_train else "val"
    images_dir_path = os.path.join(cfg.data.plankton, split)

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(cfg.device))

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

    def probe(batch):
        imgs = batch["img"].to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            _ = backbone.img_encode(imgs).img_features  # forward only

    all_ids, all_features, all_labels = [], [], []

    with helpers.auto_batch_size(dataloader, probe=probe):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for b in helpers.progress(range(total), every=10, desc=f"plk/{split}"):
            batch = next(it)
            imgs = batch["img"].to(cfg.device)

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


@beartype.beartype
def init_clf(cfg: config.Experiment):
    clf = linear_probing.LinearProbeClassifier(device=cfg.device)
    return clf
