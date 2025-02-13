"""
# MammalNet

MammalNet is built around a biological mammal taxonomy spanning 17 orders, 69 families and 173 mammal categories, and includes 12 common high-level mammal behaviors (e.g. hunt, groom).
We adopt the compositional low-shot animal and behavior recognition benchmark.

While specialized architectures exist, we train a simple nearest-centroid classifier [which works well with few-shot tasks](https://arxiv.org/abs/1911.04623) over video representations.
We get video representations by embedding each frame of the video and taking the mean over the batch dimension.

If you use this evaluation, be sure to cite the original work:

```
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Jun and Hu, Ming and Coker, Darren J. and Berumen, Michael L. and Costelloe, Blair and Beery, Sara and Rohrbach, Anna and Elhoseiny, Mohamed},
    title     = {MammalNet: A Large-Scale Video Benchmark for Mammal Recognition and Behavior Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {13052-13061}
}
```

This task was contributed by [Jianyang Gu](https://vimar-gu.github.io/).
"""

import csv
import dataclasses
import logging
import os
import typing
from tqdm import tqdm

import beartype
import numpy as np
import torch
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor
import torchvision.io as io
from torchvision import transforms as T

from biobench import interfaces, registry, simpleshot

logger = logging.getLogger("mammalnet")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    """Arguments for the MammalNet task."""

    batch_size: int = 16
    """Batch size for deep model. Note that this is multiplied by 16 (number of frames)"""
    n_workers: int = 8
    """Number of dataloader worker processes."""
    frame_agg: typing.Literal["mean", "max"] = "mean"
    """How to aggregate features across time dimension."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Video:
    """A single video instance as a sequence of frames."""

    video_id: int
    file_name: str
    """Path to actual video file."""
    label_behave: int
    """Label for animal behavior."""
    label_species: int
    """Label for animal species."""


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    Each video has two labels for behavior and species, respectively.
    """

    def __init__(self, path, split: str, transform=None, seed: int = 42):
        self.path = path
        self.split = split
        self.transform = transform
        self.seed = seed

        self.rng = np.random.default_rng(seed=seed)

        self.n_frames = 16

        # Load videos
        #############

        file_name: dict[int, str] = {}
        labels_behave: dict[int, int] = {}
        labels_species: dict[int, int] = {}

        if not os.path.exists(self.path) or not os.path.isdir(self.path):
            msg = f"Path '{self.path}' doesn't exist. Did you download the MammalNet dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path as --dataset-dir PATH"
            raise RuntimeError(msg)

        with open(os.path.join(self.path, "annotation", "composition", f"{split}.csv")) as fd:
            video_files = fd.readlines()
            for video_id, video_file in enumerate(video_files):
                path, label_behave, label_species = video_file.strip().split(" ")
                video_id = int(video_id)
                label_behave = int(label_behave)
                label_species = int(label_species)

                path = os.path.join(self.path, path[:13] + path[14:])
                file_name[video_id] = path
                labels_behave[video_id] = label_behave
                labels_species[video_id] = label_species

        self.videos = [
            Video(video_id, file_name[video_id], labels_behave[video_id], labels_species[video_id])
            for video_id in file_name.keys()
        ]

    def __getitem__(
        self, i: int
    ) -> tuple[list[Float[Tensor, "3 width height"]], int, int]:
        """
        Returns 16 frames and their labels evenly sampled from a clip.
        """
        video = self.videos[i]
        video_file = video.file_name
        label_behave = video.label_behave
        label_species = video.label_species

        frames, _, _ = io.read_video(video_file, pts_unit="sec")

        # Sample n_sample frames between the start and end with equal interval.
        indices = torch.linspace(0, len(frames) - 1, self.n_frames).long()
        frames = frames[indices]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0

        if self.transform is not None:
            frames = torch.stack([self.transform(frame) for frame in frames])

        return frames, label_behave, label_species

    def __len__(self) -> int:
        return len(self.videos)


@torch.no_grad()
@jaxtyped(typechecker=beartype.beartype)
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, dataloader
) -> tuple[
    Float[Tensor, "n_examples n_frames dim"], Int[Tensor, "n_examples"], Int[Tensor, "n_examples"]
]:
    """
    Gets all model features and true labels for all frames and all examples in the dataloader.

    Returns it as a pair of big tensors; other tasks like `biobench.birds525` use a dedicated class for this, but here it's just a tuple.

    Args:
        args: MammalNet task arguments.
        backbone: Vision backbone.
        dataloader: Dataloader for whatever data you want to get features for.

    Returns:
        tuple of model features and true labels. See signature for shape.
    """
    backbone = torch.compile(backbone)
    all_features, all_labels_behave, all_labels_species = [], [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    logger.debug("Need to embed %d batches of %d images.", total, args.batch_size * 16)
    for b in tqdm(range(total)):
        frames, labels_behave, labels_species = next(it)
        frames = frames.to(args.device)

        with torch.amp.autocast("cuda"):
            # conv2d doesn't support multiple batch dimensions, so we have to view() before and after the model.img_encode() call.
            bsz, n_frames, c, h, w = frames.shape
            frames = frames.view(bsz * n_frames, c, h, w)
            outputs = backbone.img_encode(frames)
            features = outputs.img_features.view(bsz, n_frames, -1)
            all_features.append(features.cpu())
            all_labels_behave.append(labels_behave.cpu())
            all_labels_species.append(labels_species.cpu())

        logger.debug("Embedded batch %d/%d", b + 1, total)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_labels_behave = torch.cat(all_labels_behave).cpu()
    all_labels_species = torch.cat(all_labels_species).cpu()

    return all_features, all_labels_behave, all_labels_species


@jaxtyped(typechecker=beartype.beartype)
def aggregate_frames(
    args: Args, features: Float[Tensor, "n_examples n_frames dim"]
) -> Float[Tensor, "n_examples dim"]:
    if args.frame_agg == "mean":
        return torch.mean(features, dim=1)
    elif args.frame_agg == "max":
        return torch.max(features, dim=1).values
    else:
        typing.assert_never(args.frame_agg)


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgs
) -> tuple[interfaces.ModelArgs, interfaces.TaskReport]:
    """Runs MammalNet benchmark."""
    # 1. Load model
    backbone = registry.load_vision_backbone(*model_args)
    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
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
    val_features, val_labels_behave, val_labels_species = get_features(args, backbone, val_dataloader)
    val_features = aggregate_frames(args, val_features)

    train_features, train_labels_behave, train_labels_species = get_features(args, backbone, train_dataloader)
    train_features = aggregate_frames(args, train_features)

    # 4. Do simpleshot.
    scores_behave = simpleshot.simpleshot(
        train_features, train_labels_behave,
        val_features, val_labels_behave, args.batch_size, args.device
    )
    scores_species = simpleshot.simpleshot(
        train_features, train_labels_species,
        val_features, val_labels_species, args.batch_size, args.device
    )

    # Return benchmark report.
    scores = scores_behave.long() & scores_species.long()
    video_ids = [video.video_id for video in val_dataset.videos]
    examples = [
        interfaces.Example(str(id), float(score), {})
        for id, score in zip(video_ids, scores.tolist())
    ]
    # TODO: include example-specific info (class? something else)
    return model_args, interfaces.TaskReport("MammalNet", examples)
