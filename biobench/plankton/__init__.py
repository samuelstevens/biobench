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
    keywords = {3.1.21 → Biology → Marine biology, phytoplankton image data set, imaging flow cytometry, Imaging FlowCytobot, IFCB, phytoplankton, Baltic Sea, image data, SYKE, Finnish Environment Institute, Marine Research Centre, Marine Ecological Research Laboratory, plankton image data, FINMARI},
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
  keywords = {3.1.21 → Biology → Marine biology, phytoplankton image data set, imaging flow cytometry, Imaging FlowCytobot, IFCB, Baltic Sea, image data, SYKE, Finnish Environment Institute, Marine Research Centre, Marine Ecological Research Laboratory, plankton image data, FINMARI, phytoplankton},
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
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import torch
from jaxtyping import Float, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from biobench import helpers, interfaces, registry

logger = logging.getLogger("plankton")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    """Plankton task arguments."""

    batch_size: int = 256
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of batches) to log progress."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    labels: Shaped[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]

    def y(self, encoder):
        return encoder.transform(self.labels.reshape(-1, 1)).reshape(-1)


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgs
) -> tuple[interfaces.ModelArgs, interfaces.TaskReport]:
    """
    Steps:
    1. Get features for all images.
    2. Select lambda using cross validation splits.
    3. Report score on test data.
    """
    backbone = registry.load_vision_backbone(*model_args)

    # 1. Get features
    train_features = get_features(args, backbone, split="train")
    val_features = get_features(args, backbone, split="val")

    encoder = sklearn.preprocessing.OrdinalEncoder()
    all_labels = np.concatenate((val_features.labels, train_features.labels))
    encoder.fit(all_labels.reshape(-1, 1))

    # 2. Fit model.
    clf = init_clf(args)
    clf.fit(train_features.x, train_features.y(encoder))

    # 3. Predict.
    pred_labels = clf.predict(val_features.x)
    logger.info("Predicted classes for %d examples.", len(val_features.x))
    true_labels = val_features.y(encoder)

    examples = [
        interfaces.Example(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(
            helpers.progress(val_features.ids, desc="Making examples", every=1_000),
            pred_labels,
            true_labels,
        )
    ]

    return model_args, interfaces.TaskReport("Plankton", examples)


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
            msg = f"Path '{root}' doesn't exist. Did you download the plankton dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path as --plankton-args.datadir PATH."
            raise RuntimeError(msg)

        for dirpath, dirnames, filenames in os.walk(root):
            # TODO: there are random PDFs in these directories. You have to be careful to only get directories that are actually full of images.
            # Also need to assign the same integers to the same classnames.
            image_class = os.path.relpath(dirpath, root)
            for filename in filenames:
                if not filename.endswith(".png"):
                    continue
                image_id = filename.removesuffix(".png")
                image_path = os.path.join(dirpath, filename)
                self.samples.append((image_id, image_path, image_class))

    def __getitem__(self, i: int) -> tuple[str, Float[Tensor, "3 width height"], str]:
        image_id, image_path, image_class = self.samples[i]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image_id, image, image_class

    def __len__(self) -> int:
        return len(self.samples)


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, *, split: str
) -> Features:
    images_dir_path = os.path.join(args.datadir, split)

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    dataset = Dataset(images_dir_path, img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )

    all_ids, all_features, all_labels = [], [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    for b in helpers.progress(
        range(total), every=args.log_every, desc=f"Embed {split}"
    ):
        ids, images, labels = next(it)
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features
            all_features.append(features.cpu())

        all_ids.extend(ids)

        all_labels.extend(labels)

    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    all_labels = np.array(all_labels)
    all_ids = np.array(all_ids)

    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(args: Args):
    """
    Make a grid search cross-validation version of a RidgeClassifier.
    """
    alpha = np.pow(2.0, np.arange(-20, 11))
    if args.debug:
        alpha = np.pow(2.0, np.arange(-2, 2))

    return sklearn.model_selection.GridSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.RidgeClassifier(1.0, class_weight="balanced"),
        ),
        {"ridgeclassifier__alpha": alpha},
        n_jobs=16,
        verbose=2,
    )
