import dataclasses
import logging
import os.path

import beartype
import numpy as np
import torch
import torchvision.datasets
from jaxtyping import Float, Int, Shaped, jaxtyped
from torch import Tensor

from biobench import interfaces, registry, simpleshot

__all__ = ["Args", "benchmark"]
logger = logging.getLogger("birds525")


dataset_url = (
    "https://www.kaggle.com/datasets/gpiosenka/100-bird-species/croissant/download"
)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    batch_size: int = 256
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of batches) to log progress."""
    n_repeats: int = 100
    """number of times to do 1-shot training."""


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgs
) -> tuple[interfaces.ModelArgs, interfaces.TaskReport]:
    backbone = registry.load_vision_backbone(*model_args)
    train_features = get_features(args, backbone, is_train=True)
    test_features = get_features(args, backbone, is_train=False)

    all_scores = []
    for r in range(args.n_repeats):
        i = choose_k_per_class(train_features.y, k=1)

        scores = simpleshot.simpleshot(
            train_features.x[i],
            train_features.y[i],
            test_features.x,
            test_features.y,
            args.batch_size,
            args.device,
        )
        all_scores.append(torch.mean(scores).cpu())
        if (r + 1) % args.log_every == 0:
            logger.info(
                "%d/%d simpleshot finished (%.1f%%)",
                r + 1,
                args.n_repeats,
                (r + 1) / args.n_repeats * 100,
            )

    all_scores = np.array(all_scores)

    # Just choose the last sampled result's examples.
    examples = [
        interfaces.Example(str(id), float(score), {})
        for id, score in zip(test_features.ids, scores.tolist())
    ]

    # We sort of cheat here. We run simpleshot n_repeats (100) times, then when we want to calculate the confidence intervals, we just choose the score of one of these simpleshot runs, regardless of what examples are passed.
    return model_args, interfaces.TaskReport(
        "Birds525-1shot", examples, {}, ChooseRandomCachedResult(args.seed, all_scores)
    )


@jaxtyped(typechecker=beartype.beartype)
class ChooseRandomCachedResult:
    def __init__(self, seed, scores: Float[np.ndarray, " n"]):
        self._scores = scores
        self._rng = np.random.default_rng(seed=seed)

    def __call__(self, examples: list[interfaces.Example]) -> float:
        return self._rng.choice(self._scores).item()


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[Tensor, " n dim"]
    y: Int[Tensor, " n"]
    ids: Shaped[np.ndarray, " n"]


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int) -> tuple[str, object, object]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target


@beartype.beartype
@torch.no_grad
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, *, is_train: bool
) -> Features:
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    split = "train" if is_train else "valid"

    root = os.path.join(args.datadir, split)
    if not os.path.isdir(root):
        msg = f"Path '{root}' doesn't exist. Did you download the Birds525 dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--datadir'; see --help for more."
        raise ValueError(msg)
    dataset = Dataset(os.path.join(args.datadir, split), img_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=False,
        shuffle=False,  # We use dataset.shuffle instead
    )

    all_features, all_labels, all_ids = [], [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    logger.debug("Need to embed %d batches of %d images.", total, args.batch_size)
    for b in range(total):
        ids, images, labels = next(it)

        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features = backbone.img_encode(images).img_features

        all_features.append(features.cpu())
        all_labels.extend(labels)
        all_ids.extend(ids)

        if (b + 1) % args.log_every == 0:
            logger.info("%d/%d", b + 1, total)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_ids = np.array(all_ids)
    all_labels = torch.tensor(all_labels)
    logger.info("Got features for %d images.", len(all_ids))

    return Features(all_features, all_labels, all_ids)


@jaxtyped(typechecker=beartype.beartype)
def choose_k_per_class(labels: Int[Tensor, " n"], *, k: int) -> Int[Tensor, " n_train"]:
    classes = np.unique(labels)

    train_indices = np.array([], dtype=int)

    # Iterate through each class to select indices
    for cls in classes:
        # Indices corresponding to the current class
        cls_indices = np.where(labels == cls)[0]
        # Randomly shuffle the indices
        np.random.shuffle(cls_indices)
        # Select the first K indices for the train set
        cls_train_indices = cls_indices[:k]
        # Append the selected indices to the train array
        train_indices = np.concatenate((train_indices, cls_train_indices))

    # Shuffle the indices to mix classes
    np.random.shuffle(train_indices)

    return torch.from_numpy(train_indices)
