"""
Trains a simple ridge regression classifier on visual representations for the iNat21 challenge.
In the challenge, there are 10K different species (classes).
We use the mini training set with 50 images per species, and test on the validation set, which has 10 images per species.

This task is a benchmark: it should help you understand how general a vision backbone's representations are.
This is not a true, real-world task.

If you use this task, be sure to cite the original iNat21 dataset paper:

```
@misc{inat2021,
  author={Van Horn, Grant and Mac Aodha, Oisin},
  title={iNat Challenge 2021 - FGVC8},
  publisher={Kaggle},
  year={2021},
  url={https://kaggle.com/competitions/inaturalist-2021}
}
```
"""

import dataclasses
import logging
import os

import beartype
import numpy as np
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
import torchvision.datasets
from jaxtyping import Float, Int, Shaped, jaxtyped

from biobench import config, helpers, registry, reporting

logger = logging.getLogger("inat21")

n_classes = 10_000


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
    2. Select lambda using validation data.
    3. Report score on test data.
    """
    backbone = registry.load_vision_backbone(cfg.model)

    # 1. Get features
    val_features = get_features(cfg, backbone, is_train=False)
    train_features = get_features(cfg, backbone, is_train=True)

    # 2. Fit model.
    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)

    if hasattr(clf, "best_params_"):
        helpers.write_hparam_sweep_plot("imagenet1k", cfg.model.ckpt, clf)
        alpha = clf.best_params_["ridgeclassifier__alpha"].item()
        logger.info("alpha=%.2g scored %.3f.", alpha, clf.best_score_.item())

    true_labels = val_features.y
    pred_labels = clf.predict(val_features.x)

    preds = [
        reporting.Prediction(
            str(image_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for image_id, pred, true in zip(val_features.ids, pred_labels, true_labels)
    ]

    return reporting.Report("inat21", preds, cfg)


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torchvision.datasets.ImageFolder):
    """
    Subclasses ImageFolder so that `__getitem__` includes the path, which we use as the ID.
    """

    def __getitem__(self, index: int) -> tuple[str, object, object]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (path, sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target

    @property
    def labels(self) -> Int[np.ndarray, " n"]:
        return np.array([label for _, label in self.samples])


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(cfg.device))
    split = "train_mini" if is_train else "val"

    root = os.path.join(cfg.data.inat21, split)

    if not os.path.isdir(root):
        msg = f"Path '{root}' doesn't exist. Did you download the iNat21 dataset?"
        raise ValueError(msg)

    dataset = Dataset(root, img_transform)

    if is_train and cfg.n_train > 0:
        i = helpers.balanced_random_sample(dataset.labels, cfg.n_train)
        assert len(i) == cfg.n_train
        dataset = torch.utils.data.Subset(dataset, i)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=True,
    )

    all_ids, all_features, all_labels = [], [], []

    def probe(batch):
        ids, imgs, labels = batch
        imgs = imgs.to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            backbone.img_encode(imgs).img_features

    with helpers.auto_batch_size(dataloader, probe=probe):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for b in helpers.progress(range(total), desc=f"inat21/{split}"):
            ids, images, labels = next(it)
            images = images.to(cfg.device, non_blocking=True)

            with torch.amp.autocast(cfg.device):
                features = backbone.img_encode(images).img_features

            all_features.append(features.cpu())
            all_labels.extend(labels)
            all_ids.extend(ids)

    all_features = torch.cat(all_features, dim=0).cpu().numpy()
    all_ids = np.array(all_ids)
    all_labels = torch.tensor(all_labels).numpy()
    if is_train and cfg.n_train > 0:
        assert len(all_ids) == cfg.n_train

    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    alpha = np.pow(2.0, np.arange(-15, 5))
    if cfg.debug:
        alpha = np.pow(2.0, np.arange(-2, 2))

    clf = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.RidgeClassifier(solver="saga"),
    )
    if 0 < cfg.n_train < 20_000:
        clf

    return sklearn.model_selection.HalvingGridSearchCV(
        clf,
        {"ridgeclassifier__alpha": alpha},
        n_jobs=16,
        verbose=2,
        factor=3,
    )
