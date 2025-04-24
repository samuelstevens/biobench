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

from biobench import config, helpers, registry, reporting

logger = logging.getLogger("iwildcam")


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
def score(preds: list[reporting.Prediction]) -> float:
    return reporting.macro_f1(preds)


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    backbone = registry.load_vision_backbone(cfg.model)

    # 1. Load dataloaders.
    test_features = get_features(cfg, backbone, is_train=False)
    logger.info("Got test features.")

    train_features = get_features(cfg, backbone, is_train=True)
    logger.info("Got train features.")

    # 2. Fit model.
    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)

    if hasattr(clf, "best_params_"):
        helpers.write_hparam_sweep_plot("iwildcam", cfg.model.ckpt, clf)
        alpha = clf.best_params_["ridgeclassifier__alpha"].item()
        logger.info("alpha=%.2g scored %.3f.", alpha, clf.best_score_.item())

    true_labels = test_features.y
    pred_labels = clf.predict(test_features.x)

    preds = [
        reporting.Prediction(
            str(img_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for img_id, pred, true in zip(test_features.ids, pred_labels, true_labels)
    ]

    return reporting.Report("iwildcam", preds, cfg)


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    if not os.path.exists(cfg.data.iwildcam) or not os.path.isdir(cfg.data.iwildcam):
        msg = f"Path '{cfg.data.iwildcam}' doesn't exist. Did you download the iWildCam dataset? See the docstring at the top of this file for instructions."
        raise RuntimeError(msg)

    dataset = wilds.get_dataset(
        dataset="iwildcam", download=False, root_dir=cfg.data.iwildcam
    )

    transform = backbone.make_img_transform()
    if is_train:
        dataset = dataset.get_subset("train", transform=transform)
        if cfg.n_train > 0:
            i = helpers.balanced_random_sample(dataset.y_array.numpy(), cfg.n_train)
            assert len(i) == cfg.n_train
            dataset = torch.utils.data.Subset(dataset, i)
            # When we create a Subset, it doesn't inherit the collate method from the original dataset. The WILDS dataloader expects this attribute to be present as it uses it for the collate_fn parameter. We need to copy it from the original dataset to avoid AttributeError.
            dataset.collate = dataset.dataset.collate
        dataloader = wilds.common.data_loaders.get_train_loader(
            "standard",
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_workers,
        )
    else:
        dataset = dataset.get_subset("test", transform=transform)
        dataloader = wilds.common.data_loaders.get_eval_loader(
            "standard",
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_workers,
        )

    backbone = torch.compile(backbone.to(cfg.device))
    all_features, all_labels, all_ids = [], [], []

    # I don't do `for ... in dataloader` because early breaks were throwing exceptions.
    total = len(dataloader) if not cfg.debug else 2
    it = iter(dataloader)
    for b in helpers.progress(range(total), every=10):
        imgs, labels, _ = next(it)
        imgs = imgs.to(cfg.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(imgs).img_features
            all_features.append(features.cpu())

        all_labels.extend(labels)

        ids = (np.arange(len(labels)) + b * cfg.batch_size).astype(str)
        all_ids.append(ids)

    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    all_labels = torch.tensor(all_labels).numpy()
    all_ids = np.concatenate(all_ids, axis=0)

    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    alpha = np.pow(2.0, np.arange(-15, 5))
    if cfg.debug:
        alpha = np.pow(2.0, np.arange(-2, 2))

    if 0 < cfg.n_train <= 300:
        return sklearn.linear_model.RidgeClassifier()

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
