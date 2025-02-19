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

from biobench import helpers, interfaces, registry

logger = logging.getLogger("iwildcam")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Arguments for the iWildCam task."""

    batch_size: int = 2048
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of batches) to log progress."""
    # Computed at runtime.
    max_examples: int = -1
    """(computed at runtime) Number of maximum training samples. Negative number means use all of them."""
    device: str = "cuda"
    """(computed at runtime) Which kind of accelerator to use."""
    debug: bool = False
    """(computed at runtime) Whether to run in debug mode."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
class MeanScoreCalculator:
    def __call__(self, examples: list[interfaces.Prediction]) -> float:
        y_pred = np.array([example.info["y_pred"] for example in examples])
        y_true = np.array([example.info["y_true"] for example in examples])
        score = sklearn.metrics.f1_score(
            y_true, y_pred, average="macro", labels=np.unique(y_true)
        )
        return score.item()


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgsCvml
) -> tuple[interfaces.ModelArgsCvml, interfaces.TaskReport]:
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
    logger.info("Got test features.")

    train_dataset = dataset.get_subset("train", transform=transform)
    train_dataloader = wilds.common.data_loaders.get_train_loader(
        "standard",
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    train_features = get_features(args, backbone, train_dataloader)
    logger.info("Got train features.")

    # 2. Fit model.
    clf = init_clf(args)
    clf.fit(train_features.x, train_features.y)

    helpers.write_hparam_sweep_plot("iwildcam", model_args, clf)
    alpha = clf.best_params_["ridgeclassifier__alpha"].item()
    logger.info("alpha=%.2g scored %.3f.", alpha, clf.best_score_.item())

    true_labels = test_features.y
    pred_labels = clf.predict(test_features.x)

    examples = [
        interfaces.Prediction(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(
            helpers.progress(test_features.ids, desc="making examples", every=1_000),
            pred_labels,
            true_labels,
        )
    ]

    return model_args, interfaces.TaskReport(
        "iWildCam", examples, calc_mean_score=MeanScoreCalculator()
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
    for b in helpers.progress(range(total), every=args.log_every):
        images, labels, _ = next(it)
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features
            all_features.append(features.cpu())

        all_labels.extend(labels)

        ids = (np.arange(len(labels)) + b * args.batch_size).astype(str)
        all_ids.append(ids)

    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    all_labels = torch.tensor(all_labels).numpy()
    all_ids = np.concatenate(all_ids, axis=0)

    return Features(all_features, all_labels, all_ids)


def init_clf(args: Args):
    alpha = np.pow(2.0, np.arange(-15, 5))
    if args.debug:
        alpha = np.pow(2.0, np.arange(-2, 2))

    return sklearn.model_selection.GridSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.RidgeClassifier(1.0),
        ),
        {"ridgeclassifier__alpha": alpha},
        n_jobs=16,
        verbose=2,
        # This uses sklearn.metrics.f1_score with average="macro", just like our final score calculator.
        scoring="f1_macro",
    )
