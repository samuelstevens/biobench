"""
Fits a linear classifier that is trained using cross-entropy on the training set of iWildCam 2020.

Please cite both the Wilds paper (provides the great package code) and the original iWildCam dataset:

```
@inproceedings{koh2021wilds,
  title={Wilds: A benchmark of in-the-wild distribution shifts},
  author={Koh, Pang Wei and Sagawa, Shiori and Marklund, Henrik and Xie, Sang Michael and Zhang, Marvin and Balsubramani, Akshay and Hu, Weihua and Yasunaga, Michihiro and Phillips, Richard Lanas and Gao, Irena and others},
  booktitle={International conference on machine learning},
  pages={5637--5664},
  year={2021},
  organization={PMLR}
}

@article{beery2020iwildcam,
    title={The iWildCam 2020 Competition Dataset},
    author={Beery, Sara and Cole, Elijah and Gjoka, Arvi},
    journal={arXiv preprint arXiv:2004.10340},
    year={2020}
}
```
"""

import dataclasses
import logging
import os.path

import beartype
import numpy as np
import polars as pl
import torch
import wilds
import wilds.common.data_loaders
from jaxtyping import Float, Int, Shaped, jaxtyped

from .. import config, helpers, linear_probing, registry, reporting

logger = logging.getLogger("iwildcam")


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    backbone = registry.load_vision_backbone(cfg.model)

    # 1. Load features.
    test_features = get_features(cfg, backbone, is_train=False)
    logger.info("Got test features.")

    train_features = get_features(cfg, backbone, is_train=True)
    logger.info("Got train features.")

    torch.cuda.empty_cache()  # Be nice to others on the machine.

    # 2. Fit model.
    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)

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
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
    assert df.get_column("task_name").unique().to_list() == ["iwildcam"]
    return reporting.bootstrap_scores_macro_f1(df, b=b, rng=rng)


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

    backbone = backbone.to(cfg.device)
    transform = backbone.make_img_transform()

    if is_train:
        subset = "train"
        dataset = dataset.get_subset(subset, transform=transform)
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
        subset = "test"
        dataset = dataset.get_subset(subset, transform=transform)
        dataloader = wilds.common.data_loaders.get_eval_loader(
            "standard",
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_workers,
        )

    all_features, all_labels, all_ids = [], [], []

    def probe(batch):
        imgs, labels, _ = batch
        imgs = imgs.to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            _ = backbone.img_encode(imgs).img_features

    with helpers.auto_batch_size(
        dataloader,
        probe=probe,
        # Set an upper limit that's around 1/40 of the dataset size. Otherwise we spend a lot of time picking an optimal batch size when we could just rip through the dataset. And naturally we want a power of 2.
        upper=2 ** np.log2(len(dataset) / 40).astype(int).item(),
    ):
        backbone = torch.compile(backbone)
        for batch in helpers.progress(dataloader, desc=f"iwildcam/{subset}"):
            imgs, labels, _ = batch
            imgs = imgs.to(cfg.device, non_blocking=True)

            with torch.amp.autocast(cfg.device):
                features = backbone.img_encode(imgs).img_features
                all_features.append(features.cpu())

            all_labels.extend(labels)

            ids = [str(i + len(all_ids)) for i in range(len(labels))]
            all_ids.extend(ids)

    all_features = torch.cat(all_features, axis=0).cpu().numpy()
    all_labels = torch.tensor(all_labels).numpy()
    all_ids = np.array(all_ids)

    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    clf = linear_probing.LinearProbeClassifier(device=cfg.device)
    return clf
