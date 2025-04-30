"""
# Kenyan Animal Behavior Recognition (KABR)

KABR is a video recognition task ([paper](https://openaccess.thecvf.com/content/WACV2024W/CV4Smalls/papers/Kholiavchenko_KABR_In-Situ_Dataset_for_Kenyan_Animal_Behavior_Recognition_From_Drone_WACVW_2024_paper.pdf), [website](https://kabrdata.xyz/), [Huggingface](https://huggingface.co/datasets/imageomics/KABR)) where the model predicts Kenyan animal behavior in short video segments.

This can be framed as a classification task: given a short video segment of a single animal, which behavior is most common within the segment?

While specialized architectures exist, we train a simple nearest-centroid classifier [which works well with few-shot tasks](https://arxiv.org/abs/1911.04623) over video representations.
We get video representations by embedding each frame of the video and taking the mean over the batch dimension.

## Data

To download the data, you need to use the dataset download script:

1. Copy-paste the [download script](https://huggingface.co/datasets/imageomics/KABR/raw/main/download.py) to your data directory, like `/scratch/KABR/download.py`.
2. Run `python download.py`. It doesn't have any requirements beyond the Python standard library.
"""

import csv
import dataclasses
import logging
import os

import beartype
import numpy as np
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from .. import config, helpers, registry, reporting, simpleshot

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Video:
    """A single video instance as a sequence of frames."""

    video_id: int
    """Video ID."""
    frames: list[str]
    """Paths to actual frame images."""
    labels: list[int]
    """Frame-level labels."""

    def __post_init__(self):
        err_msg = f"Video {self.video_id} has a different number of frames ({len(self.frames)} and labels ({len(self.labels)})."
        assert len(self.frames) == len(self.labels), err_msg


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    Clips of at most 90 frames in Charades format with each frame stored as an image.
    """

    def __init__(self, path, split: str, transform=None, seed: int = 42):
        self.path = path
        self.split = split
        self.transform = transform
        self.seed = seed

        self.rng = np.random.default_rng(seed=seed)

        self.n_frames = 16
        self.n_every = 5

        # Load videos
        #############

        frames: dict[int, list[str]] = {}
        labels: dict[int, list[int]] = {}

        if not os.path.exists(self.path) or not os.path.isdir(self.path):
            msg = f"Path '{self.path}' doesn't exist. Did you download the KABR dataset? See the docstring at the top of this file for instructions."
            raise RuntimeError(msg)

        with open(os.path.join(self.path, "annotation", f"{split}.csv")) as fd:
            reader = csv.reader(fd, delimiter=" ")
            next(reader)  # skip headers
            for _, video_id, frame_id, path, label in reader:
                video_id = int(video_id)
                frame_id = int(frame_id)
                label = int(label)

                if video_id not in frames:
                    frames[video_id] = []
                if video_id not in labels:
                    labels[video_id] = []

                if frame_id > len(frames[video_id]) + 1:
                    raise ValueError(f"Video {video_id} is missing a frame.")

                path = os.path.join(self.path, "dataset", "image", path)
                frames[video_id].append(path)
                labels[video_id].append(label)

        self.videos = [
            Video(video_id, frames[video_id], labels[video_id])
            for video_id in frames.keys()
            if len(frames[video_id]) >= self.n_frames
        ]

    def __getitem__(
        self, i: int
    ) -> tuple[list[Float[Tensor, "3 width height"]], list[int], str]:
        """
        Returns 16 frames and their labels sampled every 5 frames from a clip. The start of the clip is uniformly sampled. If there are fewer
        """
        n_every = self.n_every

        video = self.videos[i]

        while len(video.frames) < ((self.n_frames - 1) * n_every + 1):
            n_every -= 1

        if n_every <= 0:
            print(n_every, len(video.frames), ((self.n_frames - 1) * n_every + 1))
        assert n_every >= 1

        # margin is the number of extra frames on either size of the 16x5 sampled frames.
        margin = len(video.frames) - ((self.n_frames - 1) * n_every + 1)

        # Pick a random start, then pick n_frames frames every n_every frames.
        # (sam) This is likely not clear and there are probably better ways to express this in Python that is more clear to other video ML devs. Please open a PR if you know a better way!
        start = self.rng.integers(0, margin + 1)
        frames = video.frames[start:None:n_every][: self.n_frames]
        labels = video.labels[start:None:n_every][: self.n_frames]

        images = [Image.open(frame) for frame in frames]

        if self.transform is not None:
            images = [self.transform(image) for image in images]

        return images, labels, str(i)

    def __len__(self) -> int:
        return len(self.videos)


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    """Runs KABR benchmark."""
    # 1. Load model
    backbone = registry.load_vision_backbone(cfg.model)
    backbone = backbone.to(cfg.device)

    # 2. Load data.
    test_features = get_features(cfg, backbone, is_train=False)
    train_features = get_features(cfg, backbone, is_train=True)

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
    return reporting.Report("kabr", preds, cfg)


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
        dataloader: Dataloader for whatever data you want to get features for.

    Returns:
        tuple of model features and true labels. See signature for shape.
    """
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone)
    split = "train" if is_train else "val"

    dataset = Dataset(cfg.data.kabr, split, transform=img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        drop_last=False,
        shuffle=False,
    )

    all_feats, all_labels, all_ids = [], [], []

    def probe(batch):
        frames, _, _ = batch
        frames = torch.stack(frames, dim=0)
        frames = frames.to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            n_frames, bsz, c, h, w = frames.shape
            frames = frames.view(bsz * n_frames, c, h, w)
            outputs = backbone.img_encode(frames)
            features = outputs.img_features.view(n_frames, bsz, -1)
            features = aggregate_frames(features)

    with helpers.auto_batch_size(dataloader, probe=probe):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for b in helpers.progress(range(total), desc=f"kabr/{split}"):
            frames, labels, ids = next(it)
            frames = torch.stack(frames, dim=0)
            labels = torch.stack(labels, dim=0)
            frames = frames.to(cfg.device, non_blocking=True)

            with torch.amp.autocast(cfg.device):
                # conv2d doesn't support multiple batch dimensions, so we have to view() before and after the model.img_encode() call.
                n_frames, bsz, c, h, w = frames.shape
                frames = frames.view(bsz * n_frames, c, h, w)
                outputs = backbone.img_encode(frames)
                features = outputs.img_features.view(n_frames, bsz, -1)

                features = aggregate_frames(features)
                all_feats.append(features.cpu())

            labels = aggregate_labels(labels)
            all_labels.append(labels.cpu())

            logger.debug("Embedded batch %d/%d", b + 1, total)
            all_ids.extend(ids)

    all_feats = torch.cat(all_feats, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    all_ids = np.array(all_ids)

    return Features(all_feats, all_labels, all_ids)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    return simpleshot.SimpleShotClassifier(device="cuda:0")


@jaxtyped(typechecker=beartype.beartype)
def aggregate_labels(
    labels: Int[Tensor, "n_frames n_examples"],
) -> Int[Tensor, " n_examples"]:
    """Aggregate per-frame labels to a per-video label. Uses the most common label (mode)."""
    return torch.mode(labels, dim=0).values


@jaxtyped(typechecker=beartype.beartype)
def aggregate_frames(
    features: Float[Tensor, "n_frames n_examples dim"],
) -> Float[Tensor, "n_examples dim"]:
    return torch.max(features, dim=0).values
