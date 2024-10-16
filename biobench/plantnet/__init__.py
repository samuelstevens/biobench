import dataclasses
import logging
import os
import typing

import beartype
import numpy as np
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from biobench import helpers, interfaces, registry

logger = logging.getLogger("plantnet")

n_classes = 1081


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
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
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


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
    val_features = get_features(args, backbone, split="val")
    train_features = get_features(args, backbone, split="train")

    # 2. Fit model.
    clf = init_clf(args)
    clf.fit(train_features.x, train_features.y)

    true_labels = val_features.y
    pred_labels = clf.predict(val_features.x)

    examples = [
        interfaces.Example(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(val_features.ids, pred_labels, true_labels)
    ]

    report = interfaces.TaskReport(
        "Pl@ntNet", examples, calc_mean_score=calc_macro_top1
    )
    return model_args, report


def calc_macro_top1(examples: list[interfaces.Example]) -> float:
    """
    Macro top-1 accuracy.
    """
    cls_examples = {}
    for example in examples:
        true_cls = example.info["y_true"]
        if true_cls not in cls_examples:
            cls_examples[true_cls] = []

        cls_examples[true_cls].append(example)

    cls_accs = []
    for examples in cls_examples.values():
        cls_accs.append(np.mean([example.score for example in examples]))
    return np.mean(cls_accs).item()


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
            image_class = os.path.relpath(dirpath, root)
            for filename in filenames:
                image_id = filename.removesuffix(".jpg")
                image_path = os.path.join(dirpath, filename)
                self.samples.append((image_id, image_path, image_class))

    def __getitem__(self, i: int) -> tuple[str, Float[Tensor, "3 width height"], str]:
        image_id, image_path, image_class = self.samples[i]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image_id, image, image_class

    def __len__(self) -> int:
        return len(self.samples)


class LabelProcessor:
    def __init__(self):
        self._lookup = {}

    def transform(self, label) -> int:
        if label not in self._lookup:
            self._lookup[label] = len(self._lookup)
        return self._lookup[label]


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, *, split: str
) -> Features:
    images_dir_path = os.path.join(args.datadir, "images", split)

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
    label_processor = LabelProcessor()

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

        labels = [label_processor.transform(label) for label in labels]
        all_labels.extend(labels)

    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    all_labels = torch.tensor(all_labels).numpy()
    all_ids = np.array(all_ids)

    logger.info("Got features for %d images.", len(all_ids))
    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(args: Args):
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
        # This uses sklearn.metrics.f1_score with average="macro"
        scoring="f1_macro",
    )
