""" """

import dataclasses
import logging
import math

import beartype
import datasets
import numpy as np
import sklearn.experimental.enable_halving_search_cv
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped

from biobench import config, helpers, registry, reporting

logger = logging.getLogger("imagenet1k")


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    backbone = registry.load_vision_backbone(cfg.model)
    test_features = get_features(cfg, backbone, is_train=False)
    train_features = get_features(cfg, backbone, is_train=True)

    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)

    helpers.write_hparam_sweep_plot("imagenet1k", cfg.model.ckpt, clf)
    alpha = clf.best_params_["ridgeclassifier__alpha"].item()
    logger.info("alpha=%.2g scored %.3f.", alpha, clf.best_score_.item())

    true_labels = test_features.y
    pred_labels = clf.predict(test_features.x)

    preds = [
        reporting.Prediction(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(
            helpers.progress(test_features.ids, desc="building exs", every=1_000),
            pred_labels,
            true_labels,
        )
    ]

    return reporting.Report("imagenet1k", preds)


class Transform:
    def __init__(self, img_transform):
        self._img_transform = img_transform

    def __call__(self, example, index: int):
        example["image"] = example["image"].convert("RGB")
        example["image"] = self._img_transform(example["image"])
        example["id"] = str(index)
        return example


@beartype.beartype
@torch.no_grad
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(cfg.device))
    split = "train" if is_train else "validation"

    dataset = datasets.load_dataset("ILSVRC/imagenet-1k", split=split)
    n_examples = dataset.num_rows

    if is_train:
        i = helpers.balanced_random_sample(np.array(dataset["label"]), cfg.n_train)
    else:
        i = np.arange(n_examples)

    # Map
    dataset = (
        dataset.select(i)
        .to_iterable_dataset(num_shards=min(len(i), cfg.n_workers))
        .map(Transform(img_transform), with_indices=True)
        .shuffle(seed=cfg.seed)
        .with_format("torch")
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=False,
    )

    all_features, all_labels, all_ids = [], [], []

    total = math.ceil(n_examples / cfg.batch_size) if not cfg.debug else 20
    it = iter(dataloader)
    logger.debug("Need to embed %d batches of %d images.", total, cfg.batch_size)
    for b in helpers.progress(range(total), every=10, desc=f"Embedding {split}"):
        batch = next(it)

        images = batch["image"].to(cfg.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features

        all_features.append(features.cpu())
        all_labels.extend(batch["label"])
        all_ids.extend(batch["id"])

    all_features = torch.cat(all_features, dim=0).cpu().numpy()
    all_ids = np.array(all_ids)
    all_labels = torch.tensor(all_labels).numpy()
    logger.info("Got features for %d images.", len(all_ids))

    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    alpha = np.pow(2.0, np.arange(-15, 5))
    if cfg.debug:
        alpha = np.pow(2.0, np.arange(-2, 2))

    return sklearn.model_selection.HalvingGridSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.RidgeClassifier(1.0),
        ),
        {"ridgeclassifier__alpha": alpha},
        n_jobs=16,
        verbose=2,
        factor=3,
    )
