"""
# MammalNet

We frame behavior recognition as a classification task.
Given a short video segment, embed the video via some frame-sampling strategy and associate that embedding with a label.
We train a simple nearest-centroid classifier [which works well with few-shot tasks](https://arxiv.org/abs/1911.04623) over these representation-label pairs.

You must use torchcodec 0.2 with torch 2.6.
If you have torch 2.7, then use torchcodec 0.3.

If you use this benchmark, please cite the original work:

```
@inproceedings{chen2023mammalnet,
  title={Mammalnet: A large-scale video benchmark for mammal recognition and behavior understanding},
  author={Chen, Jun and Hu, Ming and Coker, Darren J and Berumen, Michael L and Costelloe, Blair and Beery, Sara and Rohrbach, Anna and Elhoseiny, Mohamed},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={13052--13061},
  year={2023}
}
```
"""

import collections.abc
import csv
import dataclasses
import functools
import logging
import os
import os.path

import beartype
import numpy as np
import polars as pl
import scipy.stats
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
from jaxtyping import Float, Float16, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from .. import config, helpers, registry, reporting

logger = logging.getLogger(__name__)


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    # 1. Load model
    backbone = registry.load_vision_backbone(cfg.model)
    backbone = backbone.to(cfg.device)

    # 2. Load data.
    test_features = get_features(cfg, backbone, is_train=False)
    train_features = get_features(cfg, backbone, is_train=True)
    torch.cuda.empty_cache()

    # 4. Do simpleshot.
    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)

    true_labels = test_features.y
    pred_labels = clf.predict(test_features.x)

    # Return benchmark report.
    preds = [
        reporting.Prediction(
            str(video_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for video_id, pred, true in zip(test_features.ids, pred_labels, true_labels)
    ]
    return reporting.Report("mammalnet", preds, cfg)


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
    assert df.get_column("task_name").unique().to_list() == ["mammalnet"]
    return reporting.bootstrap_scores_macro_f1(df, b=b, rng=rng)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float16[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    """
    Gets all model features and true labels for all frames and all examples in the dataloader.

    Returns it as a pair of big tensors; other tasks use a dedicated class for this, but here it's just a tuple.

    Args:
        args: KABR task arguments.
        backbone: Vision backbone.
        is_train: Whether it's training data or not.

    Returns:
    """
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone)
    split = "train" if is_train else "test"

    dataset = Dataset(cfg.data.mammalnet, split=split, transform=img_transform)

    if is_train and cfg.n_train > 0:
        i = helpers.balanced_random_sample(dataset.labels, cfg.n_train)
        assert len(i) == cfg.n_train
        dataset = torch.utils.data.Subset(dataset, i)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, cfg.batch_size // 32),
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
    )

    def probe(batch):
        with torch.amp.autocast(cfg.device):
            frames, _, _ = batch
            frames = frames.to(cfg.device, non_blocking=True)
            bsz, n_frames, c, h, w = frames.shape
            frames = frames.view(bsz * n_frames, c, h, w)
            outputs = backbone.img_encode(frames)
            features = outputs.img_features.view(bsz, n_frames, -1)
            features = aggregate_frames(features.to(torch.float16))

    all_feats, all_labels, all_ids = [], [], []

    with helpers.auto_batch_size(dataloader, probe=probe, backoff=1):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for b in helpers.progress(range(total), desc=f"mammal/{split}"):
            with torch.amp.autocast(cfg.device):
                frames, labels, ids = next(it)
                frames = frames.to(cfg.device, non_blocking=True)
                # conv2d doesn't support multiple batch dimensions, so we have to view() before and after the model.img_encode() call.
                bsz, n_frames, c, h, w = frames.shape
                frames = frames.view(bsz * n_frames, c, h, w)
                outputs = backbone.img_encode(frames)
                features = outputs.img_features.view(bsz, n_frames, -1)

                features = aggregate_frames(features.to(torch.float16))
                all_feats.append(features.cpu())

            all_labels.extend(labels)
            all_ids.extend(ids)

    all_feats = torch.cat(all_feats, dim=0).cpu().numpy()
    all_labels = np.array(all_labels)
    all_ids = np.array(all_ids)
    assert len(all_ids) == len(dataset)

    if is_train and cfg.n_train >= 0:
        assert len(all_ids) == len(dataset) == cfg.n_train

    return Features(all_feats, all_labels, all_ids)


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Video:
    frame_fpaths: list[str]
    """Full filepaths to the frames."""
    video_id: str
    """Unique identifier for the video clip."""
    species_id: int
    """Numeric ID representing the animal species in the video."""
    behavior_id: int
    """Numeric ID representing the behavior category in the video."""


@beartype.beartype
def find_videos(
    root: str, *, split: str, composition: str = "composition", n_frames: int = 32
) -> collections.abc.Iterator[Video]:
    if not os.path.exists(root) or not os.path.isdir(root):
        msg = f"Path '{root}' doesn't exist. Did you download the MammalNet dataset?"
        raise RuntimeError(msg)

    with open(os.path.join(root, "annotation", composition, f"{split}.csv")) as fd:
        reader = csv.reader(fd, delimiter=" ")
        for rel_path, behavior_id, species_id in reader:
            # the CSV prefixes "trimmed_videos/..."
            _, fname = rel_path.split("/")
            video_id, ext = os.path.splitext(fname)
            assert ext == ".mp4"

            frames_dpath = os.path.join(root, "frames", video_id)
            if not os.path.isdir(frames_dpath):
                msg = ("Missing frames for clip '%s' in split '%s'; skipping",)
                logger.warn(msg, frames_dpath, split)
                continue

            frame_fpaths = [
                os.path.join(frames_dpath, f"frame_{f + 1:02}.jpg")
                for f in range(n_frames)
            ]
            frame_fpaths = [fpath for fpath in frame_fpaths if os.path.isfile(fpath)]

            if len(frame_fpaths) < n_frames:
                msg = "Missing %d frames for clip '%s' in split '%s; skipping"
                logger.warn(msg, n_frames - len(frame_fpaths), video_id, split)
                continue

            yield Video(frame_fpaths, video_id, int(species_id), int(behavior_id))


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    _videos: list[Video]

    def __init__(self, root: str, *, split: str, transform):
        self._videos = list(
            helpers.progress(
                find_videos(root, split=split), every=500, desc=f"videos/{split}"
            )
        )
        self._transform = transform
        logger.info("Loaded %d videos for split '%s'.", len(self), split)

    def __len__(self) -> int:
        return len(self._videos)

    def __getitem__(
        self, i
    ) -> tuple[Float[Tensor, "n_frames channels width height"], int, str]:
        video = self._videos[i]

        frames = [self._transform(Image.open(fpath)) for fpath in video.frame_fpaths]

        return torch.stack(frames, dim=0), video.behavior_id, video.video_id

    @functools.cached_property
    def labels(self) -> Int[np.ndarray, " n"]:
        return np.array([video.behavior_id for video in self._videos])


@jaxtyped(typechecker=beartype.beartype)
def aggregate_frames(
    features: Float16[Tensor, "batch n_frames dim"],
) -> Float16[Tensor, "batch dim"]:
    return torch.max(features, dim=1).values


@beartype.beartype
def init_clf(cfg: config.Experiment):
    clf = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.LogisticRegression(max_iter=200),
    )
    if 0 < cfg.n_train <= 1000:
        return clf

    return sklearn.model_selection.RandomizedSearchCV(
        clf,
        {"logisticregression__C": scipy.stats.loguniform(a=1e-2, b=1e2)},
        n_iter=30,
        n_jobs=8,
        verbose=3,
        random_state=cfg.seed,
        scoring="f1_macro",
    )
