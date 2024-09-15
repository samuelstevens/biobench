__all__ = ["Args", "benchmark"]

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

from biobench import interfaces, registry

logger = logging.getLogger("plantnet")

n_classes = 1081


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    batch_size: int = 256
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, " n dim"]
    y: Int[np.ndarray, " n n_classes"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
class MeanScoreCalculator:
    """
    Macro top-1 accuracy.
    """

    def __call__(self, examples: list[interfaces.Example]) -> float:
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


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgs
) -> tuple[interfaces.ModelArgs, interfaces.TaskReport]:
    """
    Steps:
    1. Get features for all images.
    2. Select lambda using validation data.
    3. Report score on test data.
    """
    backbone = registry.load_vision_backbone(*model_args)

    # 1. Get features
    val_features = get_features(args, backbone, split="val")
    train_features = get_features(args, backbone, split="train")

    # 2. Fit model.
    model = init_ridge()
    model.fit(train_features.x, train_features.y)

    true_labels = val_features.y.argmax(axis=1)
    pred_labels = model.predict(val_features.x).argmax(axis=1)

    examples = [
        interfaces.Example(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(val_features.ids, pred_labels, true_labels)
    ]

    splits = {
        "micro-acc@1": (pred_labels == true_labels).mean().item(),
    }

    return interfaces.TaskReport("Pl@ntNet", examples, splits, calc_mean_score)


def calc_mean_score(examples: list[interfaces.Example]) -> float:
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
    all_ids, all_features, all_labels = [], [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    logger.debug("Need to embed %d batches of %d images.", total, args.batch_size)
    for b in range(total):
        ids, images, labels = next(it)
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features
            all_features.append(features.cpu())

        all_ids.extend(ids)
        all_labels.extend(labels)
        logger.debug("Embedded batch %d", b)

    # Convert labels to one single np.ndarray
    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    # Convert ids to np.ndarray of strings
    all_ids = np.array(all_ids)

    # Make one-hot encoding np.ndarray
    ##################################

    # First make a label mapping from label_str to label_int
    label_lookup = {}
    for label in sorted(set(all_labels)):
        assert label not in label_lookup
        label_lookup[label] = len(label_lookup)
    all_labels = torch.tensor([label_lookup[label] for label in all_labels])

    n_examples = len(all_labels)
    assert n_classes == len(label_lookup)

    # Then one-hot encode the labels
    all_onehots = torch.full((n_examples, n_classes), -1, dtype=all_labels.dtype)
    index = all_labels[:, None]
    # .scatter_(): all_onehots[i][index[i][j]] = src[i][j] for dim=1
    all_onehots.scatter_(dim=1, index=index, src=torch.ones_like(index))
    assert (all_onehots == 1).sum() == n_examples
    all_onehots = all_onehots.numpy()

    return Features(all_features, all_onehots, all_ids)


def init_ridge():
    return sklearn.model_selection.GridSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.Ridge(1.0),
        ),
        {"ridge__alpha": np.pow(2.0, np.arange(-20, 11))},
        n_jobs=-1,
        verbose=2,
    )
