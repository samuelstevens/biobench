"""
Fits a linear classifier that is trained using cross-entropy on the training set of iWildCam 2020.


"""

import dataclasses
import logging
import typing

import beartype
import numpy as np
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
import tqdm
import wilds
import wilds.common.data_loaders
from jaxtyping import Float, Int, Shaped, jaxtyped

from biobench import interfaces

logger = logging.getLogger("iwildcam")


@beartype.beartype
@dataclasses.dataclass
class Args:
    seed: int = 42
    """random seed."""
    # Data
    dataset_dir: str = ""
    """dataset directory; where you downloaded iWildCam to."""
    batch_size: int = 2048
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    # Computed at runtime.
    device: typing.Literal["cpu", "cuda"] = "cuda"
    """(computed at runtime) which kind of accelerator to use."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, " n dim"]
    y: Int[np.ndarray, " n n_classes"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
def benchmark(
    backbone: interfaces.VisionBackbone, args: Args
) -> interfaces.BenchmarkReport:
    # 1. Load dataloaders.
    transform = backbone.make_img_transform()
    dataset = wilds.get_dataset(
        dataset="iwildcam", download=False, root_dir=args.dataset_dir
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
    model = init_ridge()
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
        for image_id, pred, true in zip(
            tqdm.tqdm(test_features.ids), pred_labels, true_labels
        )
    ]
    logger.info("%d examples done.", len(examples))

    @beartype.beartype
    def _calc_mean_score(examples: list[interfaces.Example]) -> float:
        # Use the dataset.eval to evaluate a particular set of examples
        all_y_pred = torch.tensor([example.info["y_pred"] for example in examples])
        all_y_true = torch.tensor([example.info["y_true"] for example in examples])
        metrics, _ = dataset.eval(all_y_pred, all_y_true, None)
        return metrics["F1-macro_all"]

    return interfaces.BenchmarkReport("iWildCam", examples, {}, _calc_mean_score)


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, dataloader
) -> Features:
    backbone = torch.compile(backbone.to(args.device))

    all_features, all_labels = [], []
    for images, labels, metadata in tqdm.tqdm(dataloader, desc="Embedding images"):
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features
            all_features.append(features.cpu())

        all_labels.append(labels)

    # Convert labels to one single np.ndarray
    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    # Convert ids to np.ndarray of strings
    all_ids = np.array([str(i) for i in range(len(dataloader.dataset))])

    # Leave as Tensor so we can reference the dtype later.
    all_labels = torch.cat(all_labels, axis=0)

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
