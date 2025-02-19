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
import typing

import beartype
import numpy as np
import torch
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor

from biobench import interfaces, registry, simpleshot

logger = logging.getLogger("kabr")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Arguments for the KABR task."""

    data: str = ""
    """dataset directory; where you downloaded this task's data to."""
    batch_size_cv: int = 256
    """batch size for computer vision model."""
    n_workers: int = 4
    """Number of dataloader worker processes."""
    frame_agg: typing.Literal["mean", "max"] = "mean"
    """How to aggregate features across time dimension."""
    seed: int = 42
    """random seed."""
    parallel: int = 5
    """Concurrent requests per second."""

    # Computed at runtime.
    max_examples: int = -1
    """(computed at runtime) Number of maximum training samples. Negative number means use all of them."""
    device: str = "cuda"
    """(computed at runtime) Which kind of accelerator to use."""
    debug: bool = False
    """(computed at runtime) Whether to run in debug mode."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Video:
    """A single video instance as a sequence of frames."""

    video_id: int
    frames: list[str]
    """Paths to actual frame images."""
    labels: list[int]
    """Frame-level labels."""

    def __post_init__(self):
        err_msg = f"Video {self.video_id} has a different number of frames ({len(self.frames)} and labels ({len(self.labels)})."
        assert len(self.frames) == len(self.labels), err_msg


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
            msg = f"Path '{self.path}' doesn't exist. Did you download the KABR dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path as --dataset-dir PATH"
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
    ) -> tuple[list[Float[Tensor, "3 width height"]], list[int]]:
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

        return images, labels

    def __len__(self) -> int:
        return len(self.videos)


@torch.no_grad()
@jaxtyped(typechecker=beartype.beartype)
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, dataloader
) -> tuple[
    Float[Tensor, "n_frames n_examples dim"], Int[Tensor, "n_frames n_examples"]
]:
    """
    Gets all model features and true labels for all frames and all examples in the dataloader.

    Returns it as a pair of big tensors; other tasks like `biobench.birds525` use a dedicated class for this, but here it's just a tuple.

    Args:
        args: KABR task arguments.
        backbone: Vision backbone.
        dataloader: Dataloader for whatever data you want to get features for.

    Returns:
        tuple of model features and true labels. See signature for shape.
    """
    backbone = torch.compile(backbone)
    all_features, all_labels = [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    logger.debug("Need to embed %d batches of %d images.", total, args.batch_size * 16)
    for b in range(total):
        frames, labels = next(it)
        frames = torch.stack(frames, dim=0)
        labels = torch.stack(labels, dim=0)
        frames = frames.to(args.device)

        with torch.amp.autocast("cuda"):
            # conv2d doesn't support multiple batch dimensions, so we have to view() before and after the model.img_encode() call.
            n_frames, bsz, c, h, w = frames.shape
            frames = frames.view(bsz * n_frames, c, h, w)
            outputs = backbone.img_encode(frames)
            features = outputs.img_features.view(n_frames, bsz, -1)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

        logger.debug("Embedded batch %d/%d", b + 1, total)

    all_features = torch.cat(all_features, dim=1).cpu()
    all_labels = torch.cat(all_labels, dim=1).cpu()

    return all_features, all_labels


@jaxtyped(typechecker=beartype.beartype)
def aggregate_labels(
    args: Args, labels: Int[Tensor, "n_frames n_examples"]
) -> Int[Tensor, " n_examples"]:
    """Aggregate per-frame labels to a per-video label. Uses the most common label (mode)."""
    return torch.mode(labels, dim=0).values


@jaxtyped(typechecker=beartype.beartype)
def aggregate_frames(
    args: Args, features: Float[Tensor, "n_frames n_examples dim"]
) -> Float[Tensor, "n_examples dim"]:
    if args.frame_agg == "mean":
        return torch.mean(features, dim=0)
    elif args.frame_agg == "max":
        return torch.max(features, dim=0).values
    else:
        typing.assert_never(args.frame_agg)


@beartype.beartype
def benchmark_cvml(
    args: Args, model_args: interfaces.ModelArgsCvml
) -> tuple[interfaces.ModelArgsCvml, interfaces.TaskReport]:
    """Runs KABR benchmark."""
    # 1. Load model
    backbone = registry.load_vision_backbone(*model_args)
    img_transform = backbone.make_img_transform()
    backbone = backbone.to(args.device)

    # 2. Load data.
    train_dataset = Dataset(args.datadir, "train", transform=img_transform)
    val_dataset = Dataset(args.datadir, "val", transform=img_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
    )

    # 3. Get features
    val_features, val_labels = get_features(args, backbone, val_dataloader)
    val_features = aggregate_frames(args, val_features)
    val_labels = aggregate_labels(args, val_labels)

    train_features, train_labels = get_features(args, backbone, train_dataloader)
    train_features = aggregate_frames(args, train_features)
    train_labels = aggregate_labels(args, train_labels)

    # 4. Do simpleshot.
    scores = simpleshot.simpleshot(
        args, train_features, train_labels, val_features, val_labels
    )

    # Return benchmark report.
    video_ids = [video.video_id for video in val_dataset.videos]
    examples = [
        interfaces.Prediction(str(id), float(score), {})
        for id, score in zip(video_ids, scores.tolist())
    ]
    # TODO: include example-specific info (class? something else)
    return model_args, interfaces.TaskReport("KABR", examples)
