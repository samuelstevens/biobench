"""
Fits a linear classifier that is trained using cross-entropy on the training set of iWildCam 2020.


"""

import dataclasses
import logging
import os.path

import beartype
import numpy as np
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
import wilds
import wilds.common.data_loaders
from jaxtyping import Float, Int, Shaped, jaxtyped

from biobench import interfaces, registry

logger = logging.getLogger("iwildcam")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    batch_size: int = 2048
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
    def __call__(self, examples: list[interfaces.Example]) -> float:
        y_pred = np.array([example.info["y_pred"] for example in examples])
        y_true = np.array([example.info["y_true"] for example in examples])
        score = sklearn.metrics.f1_score(
            y_true, y_pred, average="macro", labels=np.unique(y_true)
        )
        return score.item()


@beartype.beartype
def benchmark(
    model_args: tuple[str, str], args: Args
) -> tuple[tuple[str, str], interfaces.TaskReport]:
    backbone = registry.load_vision_backbone(*model_args)
    # 1. Load dataloaders.
    transform = backbone.make_img_transform()
    if not os.path.exists(args.datadir) or not os.path.isdir(args.datadir):
        msg = f"Path '{args.datadir}' doesn't exist. Did you download the iWildCam dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path as --datadir PATH"
        raise RuntimeError(msg)
    dataset = wilds.get_dataset(
        dataset="iwildcam", download=False, root_dir=args.datadir
    )

    test_data = dataset.get_subset("test", transform=transform)
    test_dataloader = wilds.common.data_loaders.get_eval_loader(
        "standard",
        test_data,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    test_features = get_features(args, backbone, test_dataloader)

    train_dataset = dataset.get_subset("train", transform=transform)
    train_dataloader = wilds.common.data_loaders.get_train_loader(
        "standard",
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    train_features = get_features(args, backbone, train_dataloader)

    # 2. Fit model.
    model = init_ridge(args)
    model.fit(train_features.x, train_features.y)

    true_labels = test_features.y.argmax(axis=1)
    pred_labels = model.predict(test_features.x).argmax(axis=1)

    # TODO: I don't know why this is so slow. 42K examples should be faster than 40 seconds.
    logger.info("Constructing examples.")
    examples = [
        interfaces.Example(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip((test_features.ids), pred_labels, true_labels)
    ]
    logger.info("%d examples done.", len(examples))

    return model_args, interfaces.TaskReport(
        "iWildCam", examples, {}, MeanScoreCalculator()
    )


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, dataloader
) -> Features:
    backbone = torch.compile(backbone.to(args.device))

    all_features, all_labels, all_ids = [], [], []

    # I don't do `for ... in dataloader` because early breaks were throwing exceptions.
    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    logger.debug("Need to embed %d batches of %d images.", total, args.batch_size)
    for b in range(total):
        images, labels, _ = next(it)
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features
            all_features.append(features.cpu())

        all_labels.append(labels)

        ids = (np.arange(len(labels)) + b * args.batch_size).astype(str)
        all_ids.append(ids)
        logger.debug("Embedded batch %d", b)

    # Convert labels to one single np.ndarray
    all_features = torch.cat(all_features, axis=0).cpu().numpy()

    # Leave as Tensor so we can reference the dtype later.
    all_labels = torch.cat(all_labels, axis=0)

    # Convert ids to np.ndarray of strings
    all_ids = np.concatenate(all_ids, axis=0)

    # Make one-hot encoding np.ndarray
    ##################################
    n_examples = len(all_labels)
    n_classes = dataloader.dataset.n_classes

    # First one-hot encode the labels
    all_onehots = torch.full((n_examples, n_classes), -1, dtype=all_labels.dtype)
    index = all_labels[:, None]
    # .scatter_(): all_onehots[i][index[i][j]] = src[i][j] for dim=1
    all_onehots.scatter_(dim=1, index=index, src=torch.ones_like(index))
    assert (all_onehots == 1).sum() == n_examples
    all_onehots = all_onehots.numpy()

    return Features(all_features, all_onehots, all_ids)


def init_ridge(args: Args):
    alpha = np.pow(2.0, np.arange(-20, 11))
    if args.debug:
        alpha = np.pow(2.0, np.arange(-2, 2))
    return sklearn.model_selection.GridSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.Ridge(1.0),
        ),
        {"ridge__alpha": alpha},
        n_jobs=-1,
        verbose=2,
    )
